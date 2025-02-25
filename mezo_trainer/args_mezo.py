from dataclasses import dataclass, field
from transformers import TrainingArguments

@dataclass
class TrainingArguments_Mezo(TrainingArguments):
    # 添加第一个自定义参数
    trainer: str = field(
        default=None,
        metadata={"help": "Mezo Trainer"}
    )
    
    # 添加第二个自定义参数
    zo_eps: float = field(
        default=1e-3,
        metadata={"help": "zo_eps"}
    )

    # 添加第3个自定义参数
    non_diff: bool = field(
        default=False,
        metadata={"help": "Forward Auto-Differentiation"}
    )