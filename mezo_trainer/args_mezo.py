from dataclasses import dataclass, field
from transformers import TrainingArguments

@dataclass
class TrainingArguments_Mezo(TrainingArguments):
    trainer: str = field(
        default=None,
        metadata={"help": "Mezo Trainer"}
    )
    
    zo_eps: float = field(
        default=1e-3,
        metadata={"help": "zo_eps"}
    )

    non_diff: bool = field(
        default=False,
        metadata={"help": "Forward Auto-Differentiation"}
    )
