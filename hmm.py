import numpy as np
from utils import *


class Hmm(object):
    def __init__(self, char2idx_path, tag2idx_path):
        self.char2idx = load_dict(char2idx_path)
        self.tag2idx = load_dict(tag2idx_path)
        self.idx2tag = {v: k for k, v in self.tag2idx.items()}

        self.tag_size = len(self.tag2idx)
        self.vocab_size = max([v for _, v in self.char2idx.items()]) + 1
        self.emission = np.zeros([self.tag_size, self.vocab_size])
        self.transition = np.zeros([self.tag_size, self.tag_size])
        self.pi = np.zeros(self.tag_size)
        self.epsilon = 1e-8

    def fit(self, train_dic_path):
        print("initialize training")
        train_dic = load_data(train_dic_path)
        self.estimate_emission_probs(train_dic)  # 估计发射概率矩阵
        self.estimate_transition_and_initial_probs(train_dic)
        self.pi = np.log(self.pi)
        self.transition = np.log(self.transition)
        self.emission = np.log(self.emission)

    def estimate_emission_probs(self, train_dic):
        print("estimating emission probabilities")
        for dic in train_dic:
            for char, tag in zip(dic["text"], dic["label"]):
                self.emission[self.tag2idx[tag], self.char2idx[char]] += 1
        self.emission[self.emission == 0] = self.epsilon
        self.emission /= np.sum(self.emission, axis=1, keepdims=True)

    def estimate_transition_and_initial_probs(self, train_dict):
        print("estimation transition and initial probabilities")
        for dic in train_dict:
            for i, tag in enumerate(dic['label'][:-1]):
                if i == 0:
                    self.pi[self.tag2idx[tag]] += 1
                curr_tag = self.tag2idx[tag]
                next_tag = self.tag2idx[dic['label'][i+1]]
                self.transition[curr_tag, next_tag] += 1
        self.transition[self.transition == 0] = self.epsilon
        self.transition /= np.sum(self.transition, axis=-1, keepdims=True)
        self.pi[self.pi == 0] = self.epsilon
        self.epsilon /= np.sum(self.epsilon)

    def get_p_Obs_State(self, char):
        # 计算p( observation | state)
        # 如果当前字属于未知, 则讲p( observation | state)设为均匀分布
        char_token = self.char2idx.get(char, 0)
        if char_token == 0:
            return np.log(np.ones(self.tag_size)/self.tag_size)
        return np.ravel(self.emission[:, char_token])

    def predict(self, text):
        # 预测并打印出预测结果
        # 维特比算法解码
        if len(text) == 0:
            raise NotImplementedError("输入文本为空!")
        best_tag_id = self.viterbi_decode(text)
        self.print_func(text, best_tag_id)

    def print_func(self, text, best_tags_id):
        # 用来打印预测结果
        for char, tag_id in zip(text, best_tags_id):
            print(char+"_"+self.idx2tag[tag_id]+"|", end="")

    def viterbi_decode(self, text):
        """
        维特比解码, 详见视频教程或文字版教程
        :param text: 一段文本string
        :return: 最可能的隐状态路径
        """
        # 得到序列长度
        seq_len = len(text)
        # 初始化T1和T2表格
        T1_table = np.zeros([seq_len, self.tag_size])
        T2_table = np.zeros([seq_len, self.tag_size])
        # 得到第1时刻的发射概率
        start_p_Obs_State = self.get_p_Obs_State(text[0])

        # 计算第一步初始概率, 填入表中
        T1_table[0, :] = self.pi + start_p_Obs_State
        T2_table[0, :] = np.nan

        for i in range(1, seq_len):
            # 维特比算法在每一时刻计算落到每一个隐状态的最大概率和路径
            # 并把他们暂存起来
            # 这里用到了矩阵化计算方法, 详见视频教程
            p_Obs_State = self.get_p_Obs_State(text[i])
            p_Obs_State = np.expand_dims(p_Obs_State, axis=0)
            prev_score = np.expand_dims(T1_table[i-1, :], axis=-1)
            # 广播算法, 发射概率和转移概率广播 + 转移概率
            curr_score = prev_score + self.transition + p_Obs_State
            # 存入T1 T2中
            T1_table[i, :] = np.max(curr_score, axis=0)
            T2_table[i, :] = np.argmax(curr_score, axis=0)
        # 回溯
        best_tag_id = int(np.argmax(T1_table[-1, :]))
        best_tags = [best_tag_id, ]
        for i in range(seq_len-1, 0, -1):
            best_tag_id = int(T2_table[i, best_tag_id])
            best_tags.append(best_tag_id)
        return list(reversed(best_tags))

    def usage(self, text):  # 标记句子中的实体
        best_bag_id = self.viterbi_decode(text)
        token_tag_tup = list(zip(text, best_bag_id))
        cur = 0
        nxt = 1
        res = ""
        flag = 0  # 用于标记第一个实体$
        head = 1

        while nxt < len(token_tag_tup):
            if head == 1:  # 如果是开头的话
                if token_tag_tup[cur][1] == 1:  # 开头为B-ORG
                    res += "$"
                    res += token_tag_tup[cur][0]
                    flag += 1
                else:
                    res += token_tag_tup[cur][0]
                head = head - 1
            else:  # 不是开头
                res += token_tag_tup[cur][0]
                if token_tag_tup[cur][1] == 0 and token_tag_tup[nxt][1] == 1:
                    if flag < 2:
                        res += "$"
                        flag += 1
                    else:
                        res += "#"
                if token_tag_tup[cur][1] == 3 and token_tag_tup[nxt][1] == 0:
                    if flag < 2:
                        res += "$"
                        flag += 1
                    else:
                        res += "#"
            cur += 1
            nxt += 1
        return res

    def yq_usage(self, text):
        best_bag_id = self.viterbi_decode(text)
        token_tag_tup = list(zip(text, best_bag_id))
        ents = []
        tmp = ''
        for tup in token_tag_tup:
            if tup[1] == 1:
                tmp += tup[0]
            if tup[1] == 2:
                tmp += tup[0]
            if tup[1] == 3:
                tmp += tup[0]
                ents.append(tmp)
                tmp = ''
        # for ent in ents:
        #     print(ent)
        return ents


if __name__ == '__main__':
    hmm = Hmm(char2idx_path='./dicts/char2idx.json', tag2idx_path='./dicts/tag2idx.json')
    hmm.fit('./corpus/train_data.txt')
    hmm.predict("两架美国B-2隐身轰炸机和两架英国空军F-35隐身战斗机在英格兰东南部进行协调飞行练习以应对越来越复杂的安全环境，五角大楼表示B-2将会长期部署在欧洲从而更好威慑对手如俄罗斯和伊朗等国。")
