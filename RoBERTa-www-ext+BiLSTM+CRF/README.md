# BERT-BILSTM-CRF

使用RoBERTa-wwm-ext-BILSTM-CRF进行中文命名实体识别。突出一个词：**方便**。

## 问题勘验

- TypeError: init() got an unexpected keyword argument 'batch_first'： `pip install pytorch-crf==0.7.2`

# 依赖

```python
scikit-learn==1.1.3
scipy==1.10.1
seqeval==1.2.2
transformers==4.31.0
pytorch-crf==0.7.2
```

# 目录结构

```python
--checkpoint：模型和配置保存位置
--model_hub：预训练模型
----chinese-bert-wwm-ext:
--------vocab.txt
--------pytorch_model.bin
--------config.json
--data：存放数据
----dgre
--------ori_data：原始的数据
--------ner_data：处理之后的数据
------------labels.txt：标签
------------train.txt：训练数据
------------dev.txt：测试数据
--config.py：配置
--model.py：模型
--process.py：处理ori数据得到ner数据
--predict.py：加载训练好的模型进行预测
--main.py：训练和测试
```

# 说明

数据格式示例，data文件里有。
