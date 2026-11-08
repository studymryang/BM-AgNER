# ChatGLM2 训练错误修复记录

## 环境信息
- 模型：THUDM/chatglm2-6b
- 路径：/root/autodl-tmp/chatglm2-6b
- transformers 版本：5.8.0
- 配置文件：/root/llama/LlamaFactory/config/train_val_split2.yaml

---

## 问题 1：模型路径中缺少 modeling_chatglm.py

### 错误信息
OSError: /root/autodl-tmp/chatglm2-6b does not appear to have a file named modeling_chatglm.py.

### 原因
模型文件下载不完整，缺少 modeling_chatglm.py。

### 修复
从 HuggingFace 官方仓库 THUDM/chatglm2-6b 下载缺失的 modeling_chatglm.py 文件到模型目录。

---

## 问题 2：'ChatGLMConfig' object has no attribute 'max_length'

### 错误信息
AttributeError: 'ChatGLMConfig' object has no attribute 'max_length'. Did you mean: 'seq_length'?

### 原因
新版 transformers (5.8.0) 中 ChatGLMConfig 没有 max_length 属性，只有 seq_length。

### 修复
修改 /root/autodl-tmp/chatglm2-6b/modeling_chatglm.py 第 855 行：
- 修改前：self.max_sequence_length = config.max_length
- 修改后：self.max_sequence_length = getattr(config, "max_length", config.seq_length)

---

## 问题 3：object has no attribute 'all_tied_weights_keys'

### 错误信息
AttributeError: 'ChatGLMForConditionalGeneration' object has no attribute 'all_tied_weights_keys'.

### 原因
transformers 5.8.0 中 all_tied_weights_keys 等属性在 post_init() 中设置，旧版 chatglm2 代码未调用。

### 修复
在 modeling_chatglm.py 的三个模型类 __init__ 末尾添加 self.post_init()：
1. ChatGLMModel.__init__ — 在 self.dropout = torch.nn.Dropout(0.1) 之后
2. ChatGLMForConditionalGeneration.__init__ — 在 self.quantized = False 之后
3. ChatGLMForSequenceClassification.__init__ — 在 self.config = config 之后

---

## 问题 4：LoRA target modules 未找到

### 错误信息
ValueError: Target modules {'down_proj', 'up_proj', 'gate_proj', 'k_proj', 'o_proj', 'v_proj', 'q_proj'} not found.

### 原因
配置使用了 LLaMA 风格模块名，但 chatglm2 命名不同。

### ChatGLM2 与 LLaMA 模块命名对比
| 功能 | LLaMA 命名 | ChatGLM2 命名 |
|------|-----------|---------------|
| Q/K/V 投影 | q_proj, k_proj, v_proj | query_key_value |
| 注意力输出投影 | o_proj | dense |
| MLP 上投影 | up_proj | dense_h_to_4h |
| MLP 下投影 | down_proj | dense_4h_to_h |
| MLP 门控投影 | gate_proj | (无) |

### 修复
修改 /root/llama/LlamaFactory/config/train_val_split2.yaml 第 12 行：
- 修改前：lora_target: "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
- 修改后：lora_target: "query_key_value,dense,dense_h_to_4h,dense_4h_to_h"

---

## 修改的文件清单
| 文件路径 | 修改内容 |
|---------|---------|
| /root/autodl-tmp/chatglm2-6b/modeling_chatglm.py | 1. max_length 修复 2. 三个模型类添加 self.post_init() |
| /root/llama/LlamaFactory/config/train_val_split2.yaml | lora_target 改为 chatglm2 模块名 |

## 库版本变动
无变动。transformers 5.8.0 符合 LlamaFactory 要求。

---

## 一键修复命令
# 1. 修复 max_length
sed -i 's/self.max_sequence_length = config.max_length/self.max_sequence_length = getattr(config, "max_length", config.seq_length)/' /root/autodl-tmp/chatglm2-6b/modeling_chatglm.py

# 2. 添加 post_init()
python3 -c "
with open('/root/autodl-tmp/chatglm2-6b/modeling_chatglm.py', 'r') as f:
    content = f.read()
content = content.replace(
    '            self.dropout = torch.nn.Dropout(0.1)\n\n    def get_input_embeddings(self):',
    '            self.dropout = torch.nn.Dropout(0.1)\n\n        self.post_init()\n\n    def get_input_embeddings(self):')
content = content.replace(
    '        self.quantized = False\n\n        if self.config.quantization_bit:',
    '        self.quantized = False\n\n        self.post_init()\n\n        if self.config.quantization_bit:')
content = content.replace(
    '        self.config = config\n\n        if self.config.quantization_bit:\n            self.quantize(self.config.quantization_bit, empty_init=True)\n\n    def forward(',
    '        self.config = config\n\n        self.post_init()\n\n        if self.config.quantization_bit:\n            self.quantize(self.config.quantization_bit, empty_init=True)\n\n    def forward(')
with open('/root/autodl-tmp/chatglm2-6b/modeling_chatglm.py', 'w') as f:
    f.write(content)
print('Done')
"

# 验证
grep -n "self.post_init()" /root/autodl-tmp/chatglm2-6b/modeling_chatglm.py
