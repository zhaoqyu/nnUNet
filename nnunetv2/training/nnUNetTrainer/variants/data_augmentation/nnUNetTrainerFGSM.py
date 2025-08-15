import torch
import torch.nn.functional as F
import numpy as np
from typing import Union, Tuple, List, Dict, Any
from copy import deepcopy
from time import time
from torch import autocast, distributed as dist
from batchgenerators.utilities.file_and_folder_operations import join

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.helpers import dummy_context


class nnUNetTrainerFGSM(nnUNetTrainer):
    """
    nnUNetTrainer variant that integrates FGSM (Fast Gradient Sign Method) for adversarial training.
    
    This trainer makes the model robust against adversarial examples by:
    1. Generating adversarial examples using FGSM for a range of epsilon values
    2. Training the model on both clean and adversarial examples
    3. Using a curriculum learning approach where epsilon increases over training
    
    Based on the paper: https://onlinelibrary.wiley.com/doi/full/10.1155/2021/5595026
    """
    
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda'),
):
        """
        Initialize FGSM trainer.
        
        Args:
            fgsm_epsilon_range: Range of epsilon values for FGSM (min, max)
            fgsm_alpha: Step size for FGSM (usually epsilon/255)
            fgsm_steps: Number of FGSM steps (1 for standard FGSM)
            adversarial_weight: Weight for adversarial loss vs clean loss
            curriculum_epochs: Number of epochs to gradually increase epsilon
        """
        # fgsm_epsilon_range: Tuple[float, float] = (0.01, 0.1),
        # fgsm_alpha: float = 0.3,
        # fgsm_steps: int = 1,
        # adversarial_weight: float = 0.5,
        # curriculum_epochs: int = 50
        # self.fgsm_epsilon_range = fgsm_epsilon_range
        # self.fgsm_alpha = fgsm_alpha
        # self.fgsm_steps = fgsm_steps
        # self.adversarial_weight = adversarial_weight
        # self.curriculum_epochs = curriculum_epochs

        self.fgsm_epsilon_range = (0.01, 0.1)
        self.fgsm_alpha = 0.3
        self.fgsm_steps = 1
        self.adversarial_weight = 0.5
        self.curriculum_epochs = 50


 


        super().__init__(plans, configuration, fold, dataset_json, device)
        
        # Log FGSM parameters
        self.print_to_log_file(f"FGSM Training Parameters:")
        self.print_to_log_file(f"  Epsilon range: {self.fgsm_epsilon_range}")
        self.print_to_log_file(f"  Alpha: {self.fgsm_alpha}")
        self.print_to_log_file(f"  Steps: {self.fgsm_steps}")
        self.print_to_log_file(f"  Adversarial weight: {self.adversarial_weight}")
        self.print_to_log_file(f"  Curriculum epochs: {self.curriculum_epochs}")
    
    def _get_current_epsilon(self) -> float:
        """
        Get current epsilon value based on curriculum learning.
        Epsilon increases linearly from min to max over curriculum_epochs.
        """
        if self.current_epoch >= self.curriculum_epochs:
            return self.fgsm_epsilon_range[1]
        
        progress = self.current_epoch / self.curriculum_epochs
        epsilon = self.fgsm_epsilon_range[0] + progress * (self.fgsm_epsilon_range[1] - self.fgsm_epsilon_range[0])
        return epsilon
    
    def _generate_fgsm_adversarial_examples(self, data: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Generate adversarial examples using FGSM.
        
        Args:
            data: Input images [B, C, H, W, D]
            target: Target labels
            
        Returns:
            Adversarial examples
        """
        epsilon = self._get_current_epsilon()
        
        # Clone data to avoid modifying original
        data_adv = data.clone().detach().requires_grad_(True)
        
        for step in range(self.fgsm_steps):
            # Forward pass
            with torch.enable_grad():
                output = self.network(data_adv)
                loss = self.loss(output, target)
            
            # Compute gradients
            grad = torch.autograd.grad(loss, data_adv, retain_graph=False)[0]
            
            # FGSM update
            data_adv = data_adv + self.fgsm_alpha * torch.sign(grad)
            
            # Project to epsilon ball
            delta = data_adv - data
            delta = torch.clamp(delta, -epsilon, epsilon)
            data_adv = torch.clamp(data + delta, 0, 1)
            
            # Detach for next iteration
            data_adv = data_adv.detach().requires_grad_(True)
        
        return data_adv.detach()
    
    def train_step(self, batch: dict) -> dict:
        """
        Modified train step that includes adversarial training.
        """
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        
        # Generate adversarial examples
        data_adv = self._generate_fgsm_adversarial_examples(data, target)
        
        # Autocast for mixed precision training
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            # Clean forward pass
            output_clean = self.network(data)
            loss_clean = self.loss(output_clean, target)
            
            # Adversarial forward pass
            output_adv = self.network(data_adv)
            loss_adv = self.loss(output_adv, target)
            
            # Combined loss
            total_loss = (1 - self.adversarial_weight) * loss_clean + self.adversarial_weight * loss_adv

        # Backward pass
        if self.grad_scaler is not None:
            self.grad_scaler.scale(total_loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        
        # Log current epsilon for monitoring
        current_epsilon = self._get_current_epsilon()
        
        return {
            'loss': total_loss.detach().cpu().numpy(),
            'loss_clean': loss_clean.detach().cpu().numpy(),
            'loss_adv': loss_adv.detach().cpu().numpy(),
            'epsilon': current_epsilon
        }
    
    def on_train_epoch_end(self, train_outputs: List[dict]):
        """
        Modified epoch end to log adversarial training metrics.
        """
        outputs = collate_outputs(train_outputs)

        if self.is_ddp:
            losses_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(losses_tr, outputs['loss'])
            loss_here = np.vstack(losses_tr).mean()
            
            losses_clean = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(losses_clean, outputs['loss_clean'])
            loss_clean_here = np.vstack(losses_clean).mean()
            
            losses_adv = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(losses_adv, outputs['loss_adv'])
            loss_adv_here = np.vstack(losses_adv).mean()
            
            epsilons = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(epsilons, outputs['epsilon'])
            epsilon_here = np.vstack(epsilons).mean()
        else:
            loss_here = np.mean(outputs['loss'])
            loss_clean_here = np.mean(outputs['loss_clean'])
            loss_adv_here = np.mean(outputs['loss_adv'])
            epsilon_here = np.mean(outputs['epsilon'])

        self.logger.log('train_losses', loss_here, self.current_epoch)
        self.logger.log('train_losses_clean', loss_clean_here, self.current_epoch)
        self.logger.log('train_losses_adv', loss_adv_here, self.current_epoch)
        self.logger.log('fgsm_epsilon', epsilon_here, self.current_epoch)
    
    def on_epoch_end(self):
        """
        Modified epoch end to print adversarial training metrics.
        """
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

        self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('train_loss_clean', np.round(self.logger.my_fantastic_logging['train_losses_clean'][-1], decimals=4))
        self.print_to_log_file('train_loss_adv', np.round(self.logger.my_fantastic_logging['train_losses_adv'][-1], decimals=4))
        self.print_to_log_file('fgsm_epsilon', np.round(self.logger.my_fantastic_logging['fgsm_epsilon'][-1], decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
        self.print_to_log_file('Pseudo dice', [np.round(i, decimals=4) for i in
                                               self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]])
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

        # handling periodic checkpointing
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        # handle 'best' checkpointing. ema_fg_dice is computed by the logger and can be accessed like this
        if self._best_ema is None or self.logger.my_fantastic_logging['ema_fg_dice'][-1] > self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
            self.print_to_log_file(f"Yayy! New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1


class nnUNetTrainerFGSM_Conservative(nnUNetTrainerFGSM):
    """
    Conservative FGSM trainer with smaller epsilon range for medical images.
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(
            plans, configuration, fold, dataset_json, device,
            fgsm_epsilon_range=(0.005, 0.05),  # Smaller range for medical images
            fgsm_alpha=0.01,
            fgsm_steps=1,
            adversarial_weight=0.3,  # Lower weight to preserve clean performance
            curriculum_epochs=100
        )


class nnUNetTrainerFGSM_Aggressive(nnUNetTrainerFGSM):
    """
    Aggressive FGSM trainer with larger epsilon range for maximum robustness.
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(
            plans, configuration, fold, dataset_json, device,
            fgsm_epsilon_range=(0.02, 0.15),  # Larger range for maximum robustness
            fgsm_alpha=0.05,
            fgsm_steps=2,  # Multiple steps for stronger attacks
            adversarial_weight=0.7,  # Higher weight for adversarial training
            curriculum_epochs=30
        ) 