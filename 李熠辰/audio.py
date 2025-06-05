import os
import os.path as osp
import numpy as np
from tqdm import tqdm
import torch
import json
import pickle
import pandas as pd
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor,HubertModel
import librosa
from transformers import AutoProcessor, AutoModelForCTC
import noisereduce as nr
ALL_EMOTION_LABELS = {'joy':0, 'surprise':1, 'sadness':2, 'disgust':3, 'anger':4, 'fear':5, 'neutral':6}  # 数据集中的7个情绪类别

import torch
import torch.nn as nn


# class Wav2VecExtractor(object):
#     ''' 抽取wav2vec特征, 输入音频路径, 输出npy数组, 每帧768d '''
#     def __init__(self, pretrainedAudiopath, denoise=True):
#         self.downsample = 4
#         self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
#         print('[INFO] use asr based model')
#         self.processor = Wav2Vec2Processor.from_pretrained(pretrainedAudiopath)
#         self.model = Wav2Vec2ForCTC.from_pretrained(pretrainedAudiopath).to(self.device)
#         self.linear = torch.nn.Linear(1024, 768).to(self.device)
#         self.denoise = denoise  # 是否启用去噪

#     @staticmethod
#     def read_audio(wav_path):
#         '''从音频文件读取数据'''
#         speech_array, sr = librosa.load(wav_path, sr=16_000)  # 从soundfile中读取音频，返回numpy.ndarray类型
#         return speech_array, sr

#     def denoise_audio(self, audio, sr):
#         """ 对音频进行时域和频域去噪 """
#         # 时域去噪
#         audio_denoised_time = nr.reduce_noise(y=audio, sr=sr, stationary=False)

#         # 频域去噪
#         stft = librosa.stft(audio_denoised_time)
#         magnitude, phase = librosa.magphase(stft)
#         magnitude_denoised = librosa.decompose.nn_filter(magnitude, aggregate=np.median, metric='cosine')
#         stft_denoised = magnitude_denoised * phase
#         audio_denoised_freq = librosa.istft(stft_denoised)

#         return audio_denoised_freq

#     def __call__(self, wav):
#         '''提取音频特征'''
#         input_values, sr = Wav2VecExtractor.read_audio(wav)  # 读取音频

#         # 如果启用了去噪，则对音频进行去噪处理
#         if self.denoise:
#             input_values = self.denoise_audio(input_values, sr)

#         # 处理音频
#         input_values = self.processor(input_values, sampling_rate=sr, return_tensors="pt").input_values.to(self.device)

#         with torch.no_grad():
#             outputs = self.model(input_values, output_hidden_states=True)
#             hidden_states = outputs.hidden_states  # 获取所有层的隐层状态

#         # 取最后一层隐藏状态（1024维特征）
#         last_hidden_state = hidden_states[-1]  # 形状: (batch_size, seq_len, hidden_size)

#         # 对特征进行下采样
#         if self.downsample > 0:
#             last_hidden_state = torch.cat([
#                 torch.mean(last_hidden_state[:, i:i+self.downsample], dim=1) 
#                 for i in range(0, last_hidden_state.shape[1], self.downsample)
#             ], dim=0)

#         # 确保 last_hidden_state 在相同的设备上
#         last_hidden_state = last_hidden_state.to(self.device)

#         # 线性变换将1024维特征转为768维
#         last_hidden_state = self.linear(last_hidden_state)

#         # 清除缓存，释放内存
#         torch.cuda.empty_cache()

#         # 返回最终的音频特征（转换为numpy格式）
#         return last_hidden_state.detach().cpu().numpy()
    

    
def padding_forUtt(feature, MAX_LEN):
    
    length = feature.shape[0]
    if length >= MAX_LEN:
        return feature[:MAX_LEN, :], np.ones((MAX_LEN))

    pad = np.zeros([MAX_LEN - length, feature.shape[-1]])
    
    utt_pad = np.ones((length),dtype=int)

    utt_mask = np.zeros((MAX_LEN-length),dtype=int)

    feature = np.concatenate((feature, pad), axis=0)
    utt_fina_mask = np.concatenate((utt_pad, utt_mask),axis=0)
    return feature, utt_fina_mask



# 处理多个
def paddingSequence_forUtt(sequences):
    feature_dim = sequences[0].shape[-1]   # 特征维度
    lens = [s.shape[0] for s in sequences] # 用列表装所有utterance下的帧级别或者词级别的长度
    # confirm length using (mean + 3 * std)
    final_length = int(np.mean(lens) + 3 * np.std(lens))  # 最终确立的最长长度
    
    MAX_UTT_LEN = 127

    final_sequence = np.zeros([len(sequences), MAX_UTT_LEN, feature_dim])  

    final_utt_mask = np.zeros([len(sequences), MAX_UTT_LEN])
    for i, s in enumerate(sequences):
        final_sequence[i], final_utt_mask[i] = padding_forUtt(s, MAX_UTT_LEN)

    return final_sequence, final_utt_mask


# data = {}
# data['train'] = {}
# data['test'] = {}
# data['valid'] = {}

# load_project_path = "data"
# oriAnnot_path = "data/1.csv"
# curr_SetData = pd.read_csv(oriAnnot_path)
# utt_list = curr_SetData['FileName'].tolist()
# emotion_labels = curr_SetData['Emotion'].tolist()

# print('开始执行语音模态的初始化特征提取！')

# curr_SetData = pd.read_csv(oriAnnot_path)
# # 获取文件名
# utt_list = curr_SetData['FileName'].tolist()
# # 获取情感标签
# emotion_labels = curr_SetData['Emotion'].tolist()

# # 设置本地模型路径
# local_model_path = "data/wav"
# # 加载模型
# extract_audio_feat = Wav2VecExtractor(local_model_path)

# total_emotion_list, total_intent_list = [], []
# total_modalityFeat_list = []
# i = 0
# for idx, utt_name in enumerate(tqdm(utt_list)):
#     utt_id = utt_name.split('.')[0]
#     audio_path = osp.join(load_project_path, 'save_test_audio', utt_id + '.wav')

#     try:
#         audio_embedding = extract_audio_feat(audio_path)
#         print(f"Audio embedding for {audio_path} has shape: {audio_embedding.shape}")
# #         print(f"Audio embedding preview: {audio_embedding[:5]}")  # 打印前5个元素，避免过长
#         total_modalityFeat_list.append(audio_embedding)
        
#     except FileNotFoundError as e:
#         print(f"Warning: {e} - Skipping file {audio_path}")
#         continue  # 跳过当前缺失文件
#     total_emotion_list.append(ALL_EMOTION_LABELS[emotion_labels[idx]])
#     i += 1
    
#     feature_fina, utt_mask = paddingSequence_forUtt(total_modalityFeat_list)
    
# # data['test']['emotion_labels'] = np.array(total_emotion_list)
# data['val']['audio'] = feature_fina
# data['val']['audio_utt_mask'] = utt_mask

# # 打印 data['test']['audio'] 的形状
# # print(f"Shape of data['test']['audio']: {data['test']['audio'].shape}")

# save_path_tmp = os.path.join(load_project_path, 'final_feat', 'A')
# os.makedirs(save_path_tmp, exist_ok=True)

# if not os.path.exists(save_path_tmp):
#     os.makedirs(save_path_tmp)

# track_name = 'test'
# modality = 'audiolarge-denoise'
# save_path = os.path.join(save_path_tmp, f'{track_name}_{modality}.pkl')
# print(f"Saving to {save_path}")

# try:
#     with open(save_path, 'wb') as f:
#          pickle.dump(data, f, protocol=4)
# except Exception as e:
#     print(f"Error saving the data: {e}")




class HubertExtractor(object):
    ''' 抽取HuBERT特征, 输入音频路径, 输出npy数组, 每帧768d '''
    def __init__(self, pretrainedAudiopath):
        self.downsample = 4
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        print('[INFO] Using HuBERT model')
        self.processor = Wav2Vec2Processor.from_pretrained(pretrainedAudiopath)
        self.model = HubertModel.from_pretrained(pretrainedAudiopath).to(self.device)
        
        # 定义线性变换层，将1024维映射到768维
        self.linear = torch.nn.Linear(1024, 768).to(self.device)
    
    def denoise_audio(self, audio, sr):
        """ 对音频进行时域和频域去噪 """
        # 时域去噪
        audio_denoised_time = nr.reduce_noise(y=audio, sr=sr, stationary=False)

        # 频域去噪
        stft = librosa.stft(audio_denoised_time)
        magnitude, phase = librosa.magphase(stft)
        magnitude_denoised = librosa.decompose.nn_filter(magnitude, aggregate=np.median, metric='cosine')
        stft_denoised = magnitude_denoised * phase
        audio_denoised_freq = librosa.istft(stft_denoised)

        return audio_denoised_freq
    @staticmethod
    def read_audio(wav_path):
        '''从音频文件读取数据'''
        speech_array, sr = librosa.load(wav_path, sr=16_000)  # 从soundfile中读取音频，返回numpy.ndarray类型
        return speech_array, sr

    def __call__(self, wav):
        '''提取音频特征'''
        input_values, sr = HubertExtractor.read_audio(wav)  # 修复为使用 HubertExtractor 的方法
        input_values = self.processor(input_values, sampling_rate=sr, return_tensors="pt").input_values.to(self.device)  # 处理音频

        with torch.no_grad():
            outputs = self.model(input_values, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # 获取所有层的隐层状态

        # 取最后一层隐藏状态（1024维特征）
        last_hidden_state = hidden_states[-1]  # 形状: (batch_size, seq_len, hidden_size)

        # 对特征进行下采样
        if self.downsample > 0:
            last_hidden_state = torch.cat([
                torch.mean(last_hidden_state[:, i:i+self.downsample], dim=1) 
                for i in range(0, last_hidden_state.shape[1], self.downsample)
            ], dim=0)

        # 确保 last_hidden_state 在相同的设备上
        last_hidden_state = last_hidden_state.to(self.device)

        # 线性变换将1024维特征转为768维
        last_hidden_state = self.linear(last_hidden_state)

        # 清除缓存，释放内存
        torch.cuda.empty_cache()

        # 返回最终的音频特征（转换为numpy格式）
        return last_hidden_state.detach().cpu().numpy()



data = {}
data['train'] = {}
data['test'] = {}
data['valid'] = {}

load_project_path = "data"
oriAnnot_path = "data/1.csv"
curr_SetData = pd.read_csv(oriAnnot_path)
utt_list = curr_SetData['FileName'].tolist()
emotion_labels = curr_SetData['Emotion'].tolist()

print('开始执行语音模态的初始化特征提取！')


curr_SetData = pd.read_csv(oriAnnot_path)
# 获取文件名
utt_list = curr_SetData['FileName'].tolist()
# 获取情感标签
emotion_labels = curr_SetData['Emotion'].tolist()
# 初始化语音特征提取模型

# 设置本地模型路径
local_model_path = "data/Hubert"
# 加载模型
extract_audio_feat = HubertExtractor(local_model_path)

total_emotion_list, total_intent_list = [], []
total_modalityFeat_list = []
i = 0
for idx, utt_name in enumerate(tqdm(utt_list[:1])):
    utt_id = utt_name.split('.')[0]
    audio_path = osp.join(load_project_path, 'save_test_audio', utt_id + '.wav')

    try:
        audio_embedding = extract_audio_feat(audio_path)
        total_modalityFeat_list.append(audio_embedding)
        total_emotion_list.append(ALL_EMOTION_LABELS[emotion_labels[idx]])
    except FileNotFoundError as e:
        print(f"Warning: {e} - Skipping file {audio_path}")
        continue  # 跳过当前缺失文件
    i += 1
    
    feature_fina, utt_mask = paddingSequence_forUtt(total_modalityFeat_list)
    
# data['test']['emotion_labels'] = np.array(total_emotion_list)
data['test']['audio'] = feature_fina
data['test']['audio_utt_mask'] = utt_mask

# 打印 data['test']['audio'] 的形状
print(f"Shape of data['test']['audio']: {data['test']['audio'].shape}")

save_path_tmp = os.path.join(load_project_path, 'final_feat', 'A')
os.makedirs(save_path_tmp, exist_ok=True)

if not os.path.exists(save_path_tmp):
    os.makedirs(save_path_tmp)

track_name = 'test'
modality = 'newhubertaudio'
save_path = os.path.join(save_path_tmp, f'{track_name}_{modality}.pkl')
print(f"Saving to {save_path}")

try:
    with open(save_path, 'wb') as f:
         pickle.dump(data, f, protocol=4)
except Exception as e:
    print(f"Error saving the data: {e}")