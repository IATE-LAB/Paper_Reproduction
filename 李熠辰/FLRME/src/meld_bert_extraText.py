# #coding:utf-8
# import os.path as osp
# import pandas as pd
# import json
# from transformers import RobertaTokenizer  
# from transformers import BertTokenizer 
# from collections import defaultdict

# Max_seq_length = 512

# def make_text_dia(csv_path):

#     df = pd.read_csv(csv_path, encoding='utf8')
#     dia_utt_list = defaultdict(list)
#     for _, row in df.iterrows():
#         dia_num = int(row['Dialogue_ID'])
#         utt_num = int(row['Utterance_ID'])
#         dia_utt_list[str(dia_num)].append(f'dia{dia_num}_utt{utt_num}')
#     return dia_utt_list


# def _truncate_seq_pair(tokens, max_length):
#     """Truncates a sequence pair in place to the maximum length."""

#     # This is a simple heuristic which will always truncate the longer sequence
#     # one token at a time. This makes more sense than truncating an equal percent
#     # of tokens from each, since if one sequence is very short then each token
#     # that's truncated likely contains more information than a longer sequence.
#     while True:
#         tokens_len = []
#         for i,utt in enumerate(tokens):
#             tokens_len.append((i, len(utt)))
#         # print("*"*10, tokens_len)
#         sumlen = sum([i[1] for i in tokens_len])
#         # print(sumlen)
#         '''
#         MELD中只有test集合中的dialogue17的长度会很大, 剩下的dialogue都没有超过512
#         '''
#         if sumlen <= max_length:   
#             break
#         else:
#             index = sorted(tokens_len, key=lambda x:x[1], reverse=True)[0][0]
#             # print(index)
#             tokens[index].pop()
#             # print(tokens)
#     return tokens

# class InputFeatures(object):
#     """A single set of features of data."""

#     def __init__(self, input_ids, input_mask, sep_mask):
#         self.input_ids = input_ids
#         self.input_mask = input_mask
#         self.sep_mask = sep_mask

# class MELD():
#     def __init__(self, load_anno_csv,pretrainedBertPath, meld_text_path, set_name):

#         # data path  
#         self.load_anno_csv = load_anno_csv    #
#         self.pretrainedBertPath = pretrainedBertPath
#         self.meld_text_path = meld_text_path
#         self.set_name = set_name

#     def preprocess_data(self):

#         if self.pretrainedBertPath.split('/')[-1] == 'roberta-large':
#             print('  - Loading RoBERTa...')
#             tokenizer = RobertaTokenizer.from_pretrained(self.pretrainedBertPath)
#         elif self.pretrainedBertPath.split('/')[-1] == 'bert-large':
#             print('  - Loading Bert...')
#             tokenizer = BertTokenizer.from_pretrained(self.pretrainedBertPath)

#         features = []
#         csv_path = osp.join(self.load_anno_csv,self.set_name+'_sent_emo.csv')
#         int2name = make_text_dia(csv_path) #

#         text_root = osp.join(self.meld_text_path, self.set_name +'_text.json')
#         with open(text_root, 'r') as load_f:
#             load_dict = json.load(load_f)

#         for dia_id in list(int2name.keys()):   #demo [:4]

#             temp_utt = []
#             sep_mask = []
#             tokens = []

#             for utt_id in int2name[dia_id]:
#                 utterance = load_dict[utt_id]['txt'][0] 
#                 temp_utt.append(tokenizer.tokenize(utterance))
            
#             if self.pretrainedBertPath.split('/')[-1] == 'roberta-large':
#                 tokens_temp = _truncate_seq_pair(temp_utt, Max_seq_length-34*2)  #这是计算单独的时候, 之后每个dialogue下的utterance concat一起的时候, 每个utterance会加上</s>A</s>
#             elif self.pretrainedBertPath.split('/')[-1] == 'bert-large':
#                 tokens_temp = _truncate_seq_pair(temp_utt, Max_seq_length-34)  #这是计算单独的时候, 之后每个dialogue下的utterance concat一起的时候, 每个utterance后面要加上[sep]

#             for num,tokens_utt in enumerate(tokens_temp):
#                 if num == 0:
#                     if self.pretrainedBertPath.split('/')[-1] == 'roberta-large':
#                         tokens = ["<s>"] + tokens_utt + ["</s>"]
#                     elif self.pretrainedBertPath.split('/')[-1] == 'bert-large':
#                         tokens = ["[CLS]"] + tokens_utt + ["[SEP]"]
#                     sep_mask = [0] * (len(tokens)-1) + [1]  
#                 else:
#                     if self.pretrainedBertPath.split('/')[-1] == 'roberta-large':
#                         # <s> A </s></s> B </s>
#                         tokens += ["</s>"] + tokens_utt + ["</s>"]
#                         sep_mask += [0] * (len(tokens_utt)+1) + [1]  
#                     elif self.pretrainedBertPath.split('/')[-1] == 'bert-large':
#                         #[CLS] A [SEP] B [SEP]
#                         tokens += tokens_utt + ["[SEP]"]
#                         sep_mask += [0] * len(tokens_utt) + [1]  

#             input_ids = tokenizer.convert_tokens_to_ids(tokens)

#             # The mask has 1 for real tokens and 0 for padding tokens. Only real
#             # tokens are attended to.
#             input_mask = [1] * len(input_ids)  

#             # Zero-pad up to the sequence length. 对不够512长度的填充0
#             padding = [0] * (Max_seq_length - len(input_ids))  
#             input_ids += padding
#             input_mask += padding
#             sep_mask += padding

#             features.append(
#             InputFeatures(input_ids=input_ids,
#                             input_mask=input_mask,
#                             sep_mask=sep_mask))
#         return features


#coding:utf-8
import os.path as osp
import pandas as pd
import json
from transformers import DebertaV2TokenizerFast  # 导入 DeBERTa-v3 的 tokenizer
from transformers import RobertaTokenizer
from transformers import BertTokenizer
from collections import defaultdict

Max_seq_length = 512

# 对MELD函数进行预处理
# 包括：1、将对话和语句的文本数据转化为模型可接受的输入格式
#      2、保证不超过最大长度、填充成一致的长度、添加特殊标记等
#      3、使得它能够被用于Transformer模型
def make_text_dia(csv_path):
    df = pd.read_csv(csv_path, encoding='utf8')
    dia_utt_list = defaultdict(list)
    for _, row in df.iterrows():
        dia_num = int(row['Dialogue_ID'])
        utt_num = int(row['Utterance_ID'])
        dia_utt_list[str(dia_num)].append(f'dia{dia_num}_utt{utt_num}')
    return dia_utt_list

# 用于截断对话中的语句序列，使其总长度不超过max_length
def _truncate_seq_pair(tokens, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        tokens_len = []
        for i, utt in enumerate(tokens):
            tokens_len.append((i, len(utt)))
        sumlen = sum([i[1] for i in tokens_len])

        if sumlen <= max_length:   
            break
        else:
            index = sorted(tokens_len, key=lambda x:x[1], reverse=True)[0][0]
            tokens[index].pop()

    return tokens

# 定义一个数据结构，用于存储处理后的特征
class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, sep_mask):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.sep_mask = sep_mask

# 主要负责根据指定的预训练文本模型（如 DeBERTa、RoBERTa 或 BERT），对输入数据进行分词、编码、掩码处理，并返回处理后的特征列表
class MELD():
    def __init__(self, load_anno_csv, pretrainedBertPath, meld_text_path,set_name):
        # data path  
        self.load_anno_csv = load_anno_csv
        self.pretrainedBertPath = pretrainedBertPath
        self.meld_text_path = meld_text_path
        self.set_name=set_name

    def preprocess_data(self):
        print(f"Pretrained model path: {self.pretrainedBertPath}")  # 打印路径
        # 根据路径名称判断使用的模型类型
        if 'deberta' in self.pretrainedBertPath.split('/')[-1]:
            print('  - Loading DeBERTa...')
            tokenizer = DebertaV2TokenizerFast.from_pretrained(self.pretrainedBertPath)
        elif 'roberta' in self.pretrainedBertPath.split('/')[-1]:
            print('  - Loading RoBERTa...')
            tokenizer = RobertaTokenizer.from_pretrained(self.pretrainedBertPath)
        elif 'bert' in self.pretrainedBertPath.split('/')[-1]:
            print('  - Loading Bert...')
            tokenizer = BertTokenizer.from_pretrained(self.pretrainedBertPath)
        else:
            raise ValueError(f"Unsupported model type: {self.pretrainedBertPath.split('/')[-1]}. Please use 'deberta-large', 'roberta-large' or 'bert-large'.")

        features = []
        csv_path = osp.join(self.load_anno_csv,self.set_name+'_sent_emo.csv')
        int2name = make_text_dia(csv_path)

        text_root = osp.join(self.meld_text_path, self.set_name +'_text.json')
        with open(text_root, 'r') as load_f:
            load_dict = json.load(load_f)

        # 遍历所有对话
        for dia_id in list(int2name.keys()):

            temp_utt = []  # 临时存储语句
            sep_mask = []  # 存储分隔符掩码
            tokens = []  # 存储分词结果

            # 遍历当前对话的所有语句
            for utt_id in int2name[dia_id]:
                utterance = load_dict[utt_id]['txt'][0] 
                temp_utt.append(tokenizer.tokenize(utterance))

            # 根据模型类型设置最大长度限制
            if 'deberta' in self.pretrainedBertPath.split('/')[-1]:
                tokens_temp = _truncate_seq_pair(temp_utt, Max_seq_length - 34 * 2)  # DeBERTa 使用</s>标记
            elif 'roberta' in self.pretrainedBertPath.split('/')[-1]:
                tokens_temp = _truncate_seq_pair(temp_utt, Max_seq_length - 34 * 2)  # RoBERTa的特殊标记</s>
            elif 'bert' in self.pretrainedBertPath.split('/')[-1]:
                tokens_temp = _truncate_seq_pair(temp_utt, Max_seq_length - 34)  # BERT的特殊标记[SEP]

            # 遍历截断后的语句，处理特殊标记
            for num, tokens_utt in enumerate(tokens_temp):
                if num == 0:
                    # 如果是第一个utterance,则需要添加特殊标记
                    if 'deberta' in self.pretrainedBertPath.split('/')[-1]:
                        tokens = ["<s>"] + tokens_utt + ["</s>"]  # DeBERTa的标记
                    elif 'roberta' in self.pretrainedBertPath.split('/')[-1]:
                        tokens = ["<s>"] + tokens_utt + ["</s>"]
                    elif 'bert' in self.pretrainedBertPath.split('/')[-1]:
                        tokens = ["[CLS]"] + tokens_utt + ["[SEP]"]
                    sep_mask = [0] * (len(tokens) - 1) + [1]  
                else:
                    if 'deberta' in self.pretrainedBertPath.split('/')[-1]:
                        tokens += ["</s>"] + tokens_utt + ["</s>"]  # DeBERTa 使用</s>
                        sep_mask += [0] * (len(tokens_utt) + 1) + [1]  
                    elif 'roberta' in self.pretrainedBertPath.split('/')[-1]:
                        tokens += ["</s>"] + tokens_utt + ["</s>"]
                        sep_mask += [0] * (len(tokens_utt) + 1) + [1]  
                    elif 'bert' in self.pretrainedBertPath.split('/')[-1]:
                        tokens += tokens_utt + ["[SEP]"]
                        sep_mask += [0] * len(tokens_utt) + [1]  
            
            # 将tokens转换为模型输入所需的ID
            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # 创建输入掩码
            input_mask = [1] * len(input_ids)  

            # 填充序列
            padding = [0] * (Max_seq_length - len(input_ids))  
            input_ids += padding
            input_mask += padding
            sep_mask += padding

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              sep_mask=sep_mask))
        return features








