import torch
import xformers
from unsloth import FastLanguageModel
from datasets import load_dataset
import torch
from trl import SFTTrainer
from transformers import TrainingArguments

assert xformers.__version__ == "0.0.23"
assert torch.version.cuda=="12.1"

import huggingface_hub

hf_token = '허깅페이스 토큰'

huggingface_hub.login(hf_token)

max_seq_length = 2048
dtype = None
load_in_4bit = True

fourbit_model = "unsloth/gemma-7b-bnb-4bit"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = fourbit_model,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 2024,
    use_rslora = False,
    loftq_config = None
)

data_prompt = """아래는 헤드헌터와 기업 채용담당자 간 대화의 컨텍스트와, 기업에 대한 정보를 얻기 위한 헤드헌터의 다음 질문, 그리고 외부 평가자가 컨텍스트를 고려하여 질문의 품질을 평가한 점수입니다.

### 1-Context:
{}

### 1-Question:
{}

### 1-Score:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    contexts = examples["context"]
    questions = examples["question"]
    scores      = examples["score"]
    texts = []
    for context, question, score in zip(contexts, questions, scores):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = data_prompt.format(context, question, score) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

qa_dataset = "kevinrain/alpaca-hr-cleaned"

qa_dataset = load_dataset(qa_dataset, split="train")

dataset = qa_dataset.map(formatting_prompts_func, batched = True)

training_params = TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=3,
        max_steps=100,
        logging_steps=20,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=123,
        output_dir="outputs",
    )

tokenizer.padding_side = "right"

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=training_params
)

trainer_stats = trainer.train()

model.save_pretrained('gemma-finetuned')
tokenizer.save_pretrained('gemma-finetuned')

gguf_model_nm = 'gemmahr-unsloth-gguf'

quantization_method = "q8_0"

model.push_to_hub_gguf(
    gguf_model_nm,
    tokenizer,
    quantization_method=quantization_method,
    token=hf_token,
)