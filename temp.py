import torch
from transformers import AdamW, BertModel

model_name_or_path = r"E:\BaiduNetdiskDownload\chinese_L-12_H-768_A-12"
output_path = r"E:\BaiduNetdiskDownload\bert_onnx"

model = BertModel.from_pretrained(model_name_or_path)
example_input_array = {"input_ids": torch.randint(4, (1, 32), dtype=torch.long),
                       "attention_mask": torch.randint(4, (1, 32), dtype=torch.long)}
torch.onnx.export(model, example_input_array, output_path, input_names=["input_ids", "attention_mask"], output_names=['output'])