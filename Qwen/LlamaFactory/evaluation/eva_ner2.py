# f:\农业病虫害多模态知识图谱\数据验证与处理\1.py
import json
from collections import defaultdict

def parse_entities(entity_str):
    """
    解析实体字符串，返回实体列表
    """
    if not entity_str or not isinstance(entity_str, str):
        return []
    
    entities = [item.strip() for item in entity_str.split('\n') if item.strip()]
    return entities

def compute_prf(tp_count, fp_count, fn_count):
    """
    根据计数计算 Precision, Recall, F1
    """
    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
    recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def main(file_path):
    # 初始化全局计数器
    global_tp = 0
    global_fp = 0
    global_fn = 0

    # 初始化各类别计数器 (农业病虫害领域实体类型)
    category_counts = {
        '农作物': {'tp': 0, 'fp': 0, 'fn': 0},
        '病害': {'tp': 0, 'fp': 0, 'fn': 0},
        '虫害': {'tp': 0, 'fp': 0, 'fn': 0},
        '症状': {'tp': 0, 'fp': 0, 'fn': 0},
        '为害部位': {'tp': 0, 'fp': 0, 'fn': 0},
        '病原': {'tp': 0, 'fp': 0, 'fn': 0},
        '防治方法': {'tp': 0, 'fp': 0, 'fn': 0},
        '药剂': {'tp': 0, 'fp': 0, 'fn': 0},
        '发病条件': {'tp': 0, 'fp': 0, 'fn': 0},
        '病害别名': {'tp': 0, 'fp': 0, 'fn': 0},
        '虫害别名': {'tp': 0, 'fp': 0, 'fn': 0},
        '害虫天敌': {'tp': 0, 'fp': 0, 'fn': 0}
    }

    total_samples = 0

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                predict_str = data.get('predict', '')
                label_str = data.get('label', '')
                
                # 解析预测和标签实体
                pred_entities = parse_entities(predict_str)
                label_entities = parse_entities(label_str)
                
                # 转换为集合进行比对
                pred_set = set(pred_entities)
                label_set = set(label_entities)
                
                # TP: 预测中且标签中也有的
                tp_set = pred_set.intersection(label_set)
                # FP: 预测中有但标签中没有的
                fp_set = pred_set - label_set
                # FN: 标签中有但预测中没有的
                fn_set = label_set - pred_set

                # 全局计数
                global_tp += len(tp_set)
                global_fp += len(fp_set)
                global_fn += len(fn_set)

                # 分类计数 - 处理 TP
                for item in tp_set:
                    if '_' in item:
                        _, cat = item.rsplit('_', 1)
                        if cat in category_counts:
                            category_counts[cat]['tp'] += 1

                # 分类计数 - 处理 FP
                for item in fp_set:
                    if '_' in item:
                        _, cat = item.rsplit('_', 1)
                        if cat in category_counts:
                            category_counts[cat]['fp'] += 1

                # 分类计数 - 处理 FN
                for item in fn_set:
                    if '_' in item:
                        _, cat = item.rsplit('_', 1)
                        if cat in category_counts:
                            category_counts[cat]['fn'] += 1
                
                total_samples += 1

    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 未找到。")
        return

    # 打印结果表头
    print("=" * 70)
    print(f"{'类别':<12} | {'精确率(P)':<12} | {'召回率(R)':<12} | {'F1值':<12} | {'支持数':<8}")
    print("=" * 70)

    # 计算并打印每个类别的指标
    for cat in ['农作物', '病害', '虫害', '症状', '为害部位', '病原', '防治方法', '药剂', '发病条件', '病害别名', '虫害别名', '害虫天敌']:
        counts = category_counts[cat]
        p, r, f1 = compute_prf(counts['tp'], counts['fp'], counts['fn'])
        support = counts['tp'] + counts['fn']
        print(f"{cat:<12} | {p:<12.4f} | {r:<12.4f} | {f1:<12.4f} | {support:<8}")

    print("=" * 70)
    
    # 计算并打印总体指标 (Micro Average)
    overall_p, overall_r, overall_f1 = compute_prf(global_tp, global_fp, global_fn)
    print(f"{'总体(Micro)':<12} | {overall_p:<12.4f} | {overall_r:<12.4f} | {overall_f1:<12.4f} | {global_tp+global_fn:<8}")
    print("=" * 70)
    
    # 打印统计信息
    print(f"\n📊 统计信息：")
    print(f"   总样本数: {total_samples}")
    print(f"   总真实实体数: {global_tp + global_fn}")
    print(f"   总预测实体数: {global_tp + global_fp}")

if __name__ == "__main__":
    file_path = r'./saves/Qwen2.5-1.5B/eval/2026-09-07-15-43/generated_predictions.jsonl'
    main(file_path)