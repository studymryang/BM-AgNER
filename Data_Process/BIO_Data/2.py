# -*- coding: utf-8 -*-
import os
import re
import json
import codecs
import random
import codecs
from tqdm import tqdm
from collections import defaultdict
 
 
class ProcessDgreData:
    def __init__(self):
        self.data_path = "./BIO_Data/"
        self.train_file = self.data_path + "Agricultural_train_dataset_BIO1.txt"
        self.test_file = self.data_path + "Agricultural_val_dataset_BIO1.txt"
 
    def get_ner_data(self):
        with codecs.open(self.test_file, 'r', encoding="utf-8", errors="replace") as fp:
            data = fp.readlines()            
        
        # 文件路径
        filename_train = "./BIO_Data/Agricultural_train_dataset.json"
        filename_test = "./BIO_Data/Agricultural_val_dataset.json"
        
        # 处理数据的列表
        text_list = []
        label_list = []
        final_text_list = []
        final_label_list = []
        tmp_list = []
        i = 0
        
        for did, line in enumerate(data):
            line = line.strip()  # 去除行首尾的空白字符
            i = i + 1
            print(line)
            print(i)
            if not line:  # 如果行是空行
                final_text_list.append(text_list)
                final_label_list.append(label_list)
                text_list = []
                label_list = []
                continue  # 跳出循环，停止运行后续的代码  
            else:
                # text,label = line.split('\t')
                text,label = line.split()
                text_list.append(text)
                label_list.append(label)
        
        count = 0     
        for i,j in zip(final_text_list,final_label_list):
            tmp = {}
            count = count + 1 
            tmp['id'] = 'BIO' + str(count)
            tmp['text'] = i
            tmp["labels"] = j  
            tmp_list.append(tmp)
        
        # 打开文件，准备写入。'w'模式表示写入，如果文件已存在则会被覆盖  
        # 如果你希望追加数据到已有文件，可以使用'a'模式  
        with open(filename_test, 'w', encoding='utf-8') as file:  
            # 遍历列表中的每一项  
            for item in tmp_list:  
                # 将每一项写入文件。默认会添加换行符，如果你不希望换行可以省略或替换为其他分隔符  
                json.dump(item, file,ensure_ascii=False)  
                file.write("\n")  
  
        # 文件操作完成后，文件会自动关闭（因为使用了with语句）  
        print(f"数据已成功写入到文件 {filename_test}")
        
 
if __name__ == "__main__":
    processDgreData = ProcessDgreData()
    processDgreData.get_ner_data()