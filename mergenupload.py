from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = '모델 이름'
lora_path = '학습된 모델 위치'
# save_path = f'{lora_path}/merged'
upload_name = '업로드할 이름'
hf_path = '업로드될 허깅 페이스'
# 원본 모델 로드
base_model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(lora_path, local_files_only=True,)
base_model.resize_token_embeddings(len(tokenizer))

# LoRA 모델 로드
lora_model = PeftModel.from_pretrained(base_model, lora_path)

# LoRA 가중치를 원본 모델에 병합
merged_model = lora_model.merge_and_unload()

# merged_model.save_pretrained(save_path)
# tokenizer.save_pretrained(save_path)

merged_model.push_to_hub(hf_path, token = '허깅페이스 토큰')
tokenizer.push_to_hub(hf_path, token = '허깅페이스 토큰')