
# f:\农业病虫害多模态知识图谱\数据验证与处理\extract_to_txt.py
import json
import os

def extract_json_to_txt(json_file_path, output_dir):
    """
    将JSON文件中的每条数据提取并保存为单独的txt文件
    """

    # 读取JSON文件
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ 错误: 文件 {json_file_path} 未找到")
        return
    except json.JSONDecodeError as e:
        print(f"❌ JSON解析错误: {e}")
        return
    
    total_count = len(data)
    success_count = 0
    
    print(f"\n📊 开始处理，共 {total_count} 条数据...\n")
    
    for i in data:
        try:  
           with open(output_dir, 'a', encoding='utf-8') as f:
                f.write(json.dumps(i, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"❌ 出错: {e}")
            continue
    
    print(f"\n✅ 处理完成！")
    print(f"   成功导出: {total_count} 条")
    print(f"   输出目录: {os.path.abspath(output_dir)}")
    
if __name__ == "__main__":
    # 配置路径
    json_file = r'./data/Agricultural_val_dataset.json'
    output_directory = './data/Agricultural_val_dataset.txt'
    
    extract_json_to_txt(json_file, output_directory)
