# Adapter folders go here

Copy these two folders from Google Drive (`MyDrive/mental_health_review_outputs/`):

```
models/
  qwen_qlora_sft_100/     adapter_config.json + adapter_model.safetensors
  qwen_orpo_adapter/      adapter_config.json + adapter_model.safetensors
```

Only those two files per folder are needed. Do NOT copy `checkpoint-*` folders, `optimizer.pt`,
or tokenizer files. Each adapter is ~37 MB, which is within GitHub's 100 MB per-file limit.
