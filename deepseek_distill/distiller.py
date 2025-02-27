"""
Core knowledge distillation implementation for DeepSeek R1 models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoTokenizer


class DistillationLoss(nn.Module):
    """
    Implements various distillation losses for language models.
    """
    
    def __init__(
        self,
        temperature: float = 2.0,
        alpha: float = 0.5,
        beta: float = 0.0,
        hidden_loss_weight: float = 0.0,
        attention_loss_weight: float = 0.0,
        relation_loss_weight: float = 0.0,
        distill_method: str = "kd",
    ):
        """
        Initialize the distillation loss module.
        
        Args:
            temperature: Temperature for softening probability distributions
            alpha: Weight for distillation loss vs task loss (KL divergence)
            beta: Weight for hard-label loss
            hidden_loss_weight: Weight for hidden state mimicking loss
            attention_loss_weight: Weight for attention map mimicking loss
            relation_loss_weight: Weight for relation-based distillation loss
            distill_method: Distillation technique to use (kd, pkd, dkd, skd, mtd)
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.hidden_loss_weight = hidden_loss_weight
        self.attention_loss_weight = attention_loss_weight
        self.relation_loss_weight = relation_loss_weight
        self.distill_method = distill_method
        
    def kl_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute KL divergence loss between student and teacher logits.
        
        Args:
            student_logits: Student model logits
            teacher_logits: Teacher model logits
            mask: Optional mask to apply (1 for tokens to compute loss on, 0 elsewhere)
            
        Returns:
            KL divergence loss
        """
        student_logits = student_logits / self.temperature
        teacher_logits = teacher_logits / self.temperature
        
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        
        kl_div = F.kl_div(student_log_probs, teacher_probs, reduction="none")
        kl_div = kl_div.sum(-1)  # Sum over vocabulary dimension
        
        if mask is not None:
            kl_div = kl_div * mask
            return (kl_div.sum() / mask.sum()) * (self.temperature ** 2)
        
        return kl_div.mean() * (self.temperature ** 2)
    
    def hidden_mse_loss(self, student_hidden: torch.Tensor, teacher_hidden: torch.Tensor,
                       mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute MSE loss between student and teacher hidden states.
        
        Args:
            student_hidden: Student model hidden states
            teacher_hidden: Teacher model hidden states
            mask: Optional mask to apply
            
        Returns:
            MSE loss between hidden states
        """
        if student_hidden.shape != teacher_hidden.shape:
            # If dimensions don't match, project student hidden states
            proj = nn.Linear(student_hidden.size(-1), teacher_hidden.size(-1), bias=False).to(student_hidden.device)
            student_hidden = proj(student_hidden)
        
        mse = F.mse_loss(student_hidden, teacher_hidden, reduction="none").mean(-1)
        
        if mask is not None:
            mse = mse * mask
            return mse.sum() / mask.sum()
        
        return mse.mean()
    
    def attention_loss(self, student_attentions: List[torch.Tensor], 
                      teacher_attentions: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute loss between student and teacher attention maps.
        
        Args:
            student_attentions: List of student attention maps
            teacher_attentions: List of teacher attention maps
            
        Returns:
            Loss between attention maps
        """
        total_loss = 0
        count = 0
        
        # Use a subset of layers if student and teacher have different number of layers
        for student_attn, teacher_attn in zip(student_attentions, teacher_attentions):
            # Handle different attention head dimensions
            if student_attn.size(1) != teacher_attn.size(1):
                # Average teacher heads if teacher has more heads
                if teacher_attn.size(1) > student_attn.size(1):
                    # Reshape and average groups of teacher heads to match student heads
                    head_ratio = teacher_attn.size(1) // student_attn.size(1)
                    teacher_attn = teacher_attn.view(
                        teacher_attn.size(0),
                        student_attn.size(1),
                        head_ratio,
                        teacher_attn.size(2),
                        teacher_attn.size(3)
                    ).mean(2)
                else:
                    # For simplicity, just use a subset of student heads if student has more heads
                    student_attn = student_attn[:, :teacher_attn.size(1), :, :]
            
            # Compute loss between attention maps
            loss = F.mse_loss(student_attn, teacher_attn, reduction="mean")
            total_loss += loss
            count += 1
            
        return total_loss / max(count, 1)
    
    def relation_distillation_loss(self, student_hidden: torch.Tensor, 
                                  teacher_hidden: torch.Tensor) -> torch.Tensor:
        """
        Compute relation-based distillation loss.
        
        Args:
            student_hidden: Student model hidden states
            teacher_hidden: Teacher model hidden states
            
        Returns:
            Relation-based distillation loss
        """
        # Compute cosine similarity matrices
        student_norm = F.normalize(student_hidden, p=2, dim=-1)
        teacher_norm = F.normalize(teacher_hidden, p=2, dim=-1)
        
        student_similarity = torch.matmul(student_norm, student_norm.transpose(-1, -2))
        teacher_similarity = torch.matmul(teacher_norm, teacher_norm.transpose(-1, -2))
        
        # Compute MSE between similarity matrices
        loss = F.mse_loss(student_similarity, teacher_similarity, reduction="mean")
        return loss
    
    def forward(
        self,
        student_outputs: Dict[str, torch.Tensor],
        teacher_outputs: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the distillation loss.
        
        Args:
            student_outputs: Outputs from the student model
            teacher_outputs: Outputs from the teacher model
            labels: Optional labels for supervised loss
            attention_mask: Optional attention mask
            
        Returns:
            Total loss and a dictionary of component losses
        """
        losses = {}
        
        # Get logits from both models
        student_logits = student_outputs.get("logits")
        teacher_logits = teacher_outputs.get("logits")
        
        # Create a mask for computing loss (exclude padding tokens)
        if attention_mask is not None:
            mask = attention_mask.view(-1).float()
        else:
            mask = None
        
        # Calculate KL divergence loss
        if student_logits is not None and teacher_logits is not None:
            kd_loss = self.kl_loss(student_logits, teacher_logits, mask)
            losses["kd_loss"] = kd_loss
        else:
            kd_loss = 0
            losses["kd_loss"] = torch.tensor(0.0, device=student_outputs["hidden_states"][-1].device)
        
        # Calculate supervised loss if labels are provided
        if labels is not None and student_logits is not None:
            # Shift logits and labels for next token prediction
            shift_logits = student_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Create loss mask based on attention and valid labels
            loss_mask = attention_mask[..., 1:].contiguous() if attention_mask is not None else None
            if loss_mask is not None:
                valid_label_mask = (shift_labels != -100).float()
                loss_mask = loss_mask * valid_label_mask
            
            # Compute cross-entropy loss
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none"
            )
            
            if loss_mask is not None:
                ce_loss = ce_loss * loss_mask.view(-1)
                ce_loss = ce_loss.sum() / loss_mask.sum()
            else:
                ce_loss = ce_loss.mean()
                
            losses["ce_loss"] = ce_loss
        else:
            ce_loss = 0
            losses["ce_loss"] = torch.tensor(0.0, device=student_outputs["hidden_states"][-1].device)
        
        # Calculate hidden state loss if requested
        if self.hidden_loss_weight > 0:
            student_hidden = student_outputs["hidden_states"][-1]
            teacher_hidden = teacher_outputs["hidden_states"][-1]
            hidden_loss = self.hidden_mse_loss(student_hidden, teacher_hidden, mask)
            losses["hidden_loss"] = hidden_loss
        else:
            hidden_loss = 0
            losses["hidden_loss"] = torch.tensor(0.0, device=student_outputs["hidden_states"][-1].device)
        
        # Calculate attention map loss if requested
        if self.attention_loss_weight > 0 and "attentions" in student_outputs and "attentions" in teacher_outputs:
            student_attentions = student_outputs["attentions"]
            teacher_attentions = teacher_outputs["attentions"]
            attn_loss = self.attention_loss(student_attentions, teacher_attentions)
            losses["attention_loss"] = attn_loss
        else:
            attn_loss = 0
            losses["attention_loss"] = torch.tensor(0.0, device=student_outputs["hidden_states"][-1].device)
        
        # Calculate relation-based distillation loss if requested
        if self.relation_loss_weight > 0:
            student_hidden = student_outputs["hidden_states"][-1]
            teacher_hidden = teacher_outputs["hidden_states"][-1]
            relation_loss = self.relation_distillation_loss(student_hidden, teacher_hidden)
            losses["relation_loss"] = relation_loss
        else:
            relation_loss = 0
            losses["relation_loss"] = torch.tensor(0.0, device=student_outputs["hidden_states"][-1].device)
        
        # Combine all losses
        total_loss = (
            (1 - self.alpha) * ce_loss +
            self.alpha * kd_loss +
            self.hidden_loss_weight * hidden_loss +
            self.attention_loss_weight * attn_loss +
            self.relation_loss_weight * relation_loss
        )
        
        losses["total_loss"] = total_loss
        
        return total_loss, losses


class DeepSeekDistiller:
    """
    Main class for knowledge distillation of DeepSeek R1 models.
    """
    
    def __init__(
        self,
        teacher_model: Union[str, PreTrainedModel],
        student_model: Union[str, PreTrainedModel],
        tokenizer: Optional[Any] = None,
        temperature: float = 2.0,
        alpha: float = 0.5,
        hidden_loss_weight: float = 0.0,
        attention_loss_weight: float = 0.0,
        relation_loss_weight: float = 0.0,
        distill_method: str = "kd",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        teacher_dtype: str = "float16",
        student_dtype: str = "float16",
    ):
        """
        Initialize the distiller.
        
        Args:
            teacher_model: Teacher model or path to teacher model
            student_model: Student model or path to student model
            tokenizer: Tokenizer for both models
            temperature: Temperature for distillation
            alpha: Weight for distillation loss vs task loss
            hidden_loss_weight: Weight for hidden state mimicking
            attention_loss_weight: Weight for attention map mimicking
            relation_loss_weight: Weight for relation-based distillation
            distill_method: Distillation technique to use
            device: Device to run models on
            teacher_dtype: Data type for teacher model
            student_dtype: Data type for student model
        """
        self.device = device
        self.teacher_dtype = self._get_dtype(teacher_dtype)
        self.student_dtype = self._get_dtype(student_dtype)
        
        # Load teacher model
        if isinstance(teacher_model, str):
            self.teacher = AutoModelForCausalLM.from_pretrained(
                teacher_model,
                torch_dtype=self.teacher_dtype,
                device_map="auto" if self.device == "cuda" else None,
                output_hidden_states=True,
                output_attentions=attention_loss_weight > 0,
            )
        else:
            self.teacher = teacher_model
            self.teacher.config.output_hidden_states = True
            self.teacher.config.output_attentions = attention_loss_weight > 0
            
        # Ensure teacher is in eval mode and optionally freeze
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        # Load student model
        if isinstance(student_model, str):
            self.student = AutoModelForCausalLM.from_pretrained(
                student_model,
                torch_dtype=self.student_dtype,
                device_map="auto" if self.device == "cuda" else None,
                output_hidden_states=True,
                output_attentions=attention_loss_weight > 0,
            )
        else:
            self.student = student_model
            self.student.config.output_hidden_states = True
            self.student.config.output_attentions = attention_loss_weight > 0
            
        # Load tokenizer if not provided
        if tokenizer is None:
            if isinstance(teacher_model, str):
                self.tokenizer = AutoTokenizer.from_pretrained(teacher_model)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(student_model if isinstance(student_model, str) else "deepseek-ai/deepseek-r1-model-330m")
        else:
            self.tokenizer = tokenizer
            
        # Initialize distillation loss
        self.distillation_loss = DistillationLoss(
            temperature=temperature,
            alpha=alpha,
            hidden_loss_weight=hidden_loss_weight,
            attention_loss_weight=attention_loss_weight,
            relation_loss_weight=relation_loss_weight,
            distill_method=distill_method,
        )
        
    def _get_dtype(self, dtype_str: str) -> torch.dtype:
        """
        Convert string dtype to torch dtype.
        
        Args:
            dtype_str: String representation of dtype
            
        Returns:
            Corresponding torch dtype
        """
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(dtype_str, torch.float32)
    
    def train_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Perform a single training step.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Optional labels for supervised loss
            
        Returns:
            Total loss and dictionary of component losses
        """
        # Get teacher outputs (no gradients needed)
        with torch.no_grad():
            teacher_outputs = self.teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            
        # Get student outputs
        student_outputs = self.student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,  # This will compute loss internally if provided
            return_dict=True,
        )
        
        # Calculate distillation loss
        total_loss, losses = self.distillation_loss(
            student_outputs=student_outputs,
            teacher_outputs=teacher_outputs,
            labels=labels,
            attention_mask=attention_mask,
        )
        
        return total_loss, losses
    
    def save_student(self, output_dir: str):
        """
        Save the student model and tokenizer.
        
        Args:
            output_dir: Directory to save model to
        """
        self.student.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
