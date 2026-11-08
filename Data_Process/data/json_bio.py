import json

def convert_to_bio_format(input_file, output_file):
    """
    将Agricultural_NER_Dataset.txt中的数据转换为BIO格式的命名实体识别数据集
    
    参数:
        input_file: 输入文件路径 (Agricultural_NER_Dataset.txt)
        output_file: 输出文件路径 (BIO格式)
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    bio_results = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        try:
            data = json.loads(line)
            
            # 提取input中的文本内容（去掉"文本："前缀）
            input_text = data['input']
            if input_text.startswith('文本：'):
                text = input_text[3:]
            else:
                text = input_text
            
            # 解析output中的实体和类型
            output_text = data['output']
            entities = []
            
            if output_text != '没有':
                entity_lines = output_text.split('\n')
                for entity_line in entity_lines:
                    entity_line = entity_line.strip()
                    if '_' in entity_line:
                        entity_name, entity_type = entity_line.rsplit('_', 1)
                        
                        # 统一实体类型名称（中文转英文缩写）
                        entity_type_mapping = {
                            '农作物': 'CROP',
                            '病害': 'DIS',
                            '虫害': 'PEST',
                            '症状': 'SYM',
                            '为害部位': 'PART',
                            '病原': 'PATH',
                            '防治方法': 'CTRL',
                            '药剂': 'MED',
                            '发病条件': 'COND',
                            '病害别名': 'DIS_ALIAS',
                            '虫害别名': 'PEST_ALIAS',
                            '害虫天敌': 'NAT_ENEMY'
                        }
                        
                        if entity_type in entity_type_mapping:
                            entity_type = entity_type_mapping[entity_type]
                        else:
                            continue  # 跳过未知类型
                        
                        entities.append((entity_name, entity_type))
            
            # 生成BIO标注
            bio_tags = ['O'] * len(text)
            
            # 按实体长度降序排列，优先匹配长实体
            entities.sort(key=lambda x: len(x[0]), reverse=True)
            
            for entity_name, entity_type in entities:
                entity_len = len(entity_name)
                
                # 查找实体在文本中的所有位置
                start_idx = 0
                while start_idx < len(text):
                    start_idx = text.find(entity_name, start_idx)
                    if start_idx == -1:
                        break
                    
                    # 检查该位置是否已被标注，如果已标注则跳过（解决实体嵌套问题）
                    can_mark = True
                    for i in range(entity_len):
                        if start_idx + i < len(bio_tags) and bio_tags[start_idx + i] != 'O':
                            can_mark = False
                            break
                    
                    if can_mark:
                        # 标记B-XXX
                        bio_tags[start_idx] = f'B-{entity_type}'
                        # 标记I-XXX
                        for i in range(1, entity_len):
                            if start_idx + i < len(bio_tags):
                                bio_tags[start_idx + i] = f'I-{entity_type}'
                    
                    # 移动到下一个位置继续查找
                    start_idx += entity_len
            
            # 生成BIO格式的行
            bio_lines = []
            for char, tag in zip(text, bio_tags):
                bio_lines.append(f'{char} {tag}')
            
            # 添加空行分隔不同句子
            bio_results.append('\n'.join(bio_lines))
        
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            continue
    
    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(bio_results))
    
    print(f"转换完成！已生成 {len(bio_results)} 条BIO格式数据。")

# 使用示例
if __name__ == '__main__':
    input_path = r'./data/Agricultural_val_dataset.txt'
    output_path = r'./data/Agricultural_val_dataset_BIO.txt'
    
    convert_to_bio_format(input_path, output_path)