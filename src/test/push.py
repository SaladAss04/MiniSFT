from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer
# 使用 Huggingface API 创建仓库
model = AutoModelForCausalLM.from_pretrained('outputs/model/zero3/checkpoint-1352')
tokenizer = AutoTokenizer.from_pretrained('outputs/model/zero3/checkpoint-1352')
repo_name = "minisft"  # 模型库的名字
username = "cosmicthrillseeking"  # 你的 Huggingface 用户名
name = username + '/' + repo_name


# 推送模型到 Huggingface
model.push_to_hub(name)

# 推送分词器到 Huggingface
tokenizer.push_to_hub(name)