import itertools
import math
import random

import numpy as np
import pandas as pd
import pm4py
from pm4py.objects.conversion.log import converter as log_converter
import Levenshtein
import sys
from openpyxl import load_workbook
#计算最大公共字符序列
def longest_common_subsequence(X, Y):
    m, n = len(X), len(Y)
    L = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])

    # 从L[m][n]开始构建LCS
    index = L[m][n]
    lcs = [""] * (index + 1)
    lcs[index] = ""

    i, j = m, n
    while i > 0 and j > 0:
        if X[i - 1] == Y[j - 1]:
            lcs[index - 1] = X[i - 1]
            i -= 1
            j -= 1
            index -= 1
        elif L[i - 1][j] > L[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return "".join(lcs)
class OnlinePrefixAlignment:
    def __init__(self, max_cases):
        self.max_cases = max_cases
        self.dc = {}  # Data structure for storing prefix alignments
        self.rc = {}  # Repository for summary states
        self.trace = {} #trace
        #以下用来记录表格的变量
        self.data = pd.DataFrame()

    #在线流程
    def process_event(self, case_id, event, subset, alignments_dicf, model_fitness, min_model,alignments_dic):
        # Check if the case is already in memory
        # print(case_id)
        # print(event)
        print(event)
        #匹配计算轨迹所在位置

        if case_id in self.dc:
            # Update the existing prefix alignment
            self.update_prefix_alignment(case_id, event, subset, alignments_dicf, model_fitness, min_model)


        else:
            # If the case limit is reached, forget the least important case
            if len(self.dc) >= self.max_cases:
                self.forget_case()
            # Start a new prefix alignment for the new case
            self.create_new_prefix_alignment(case_id, event , subset, alignments_dicf, model_fitness, min_model)

        print(case_id)
        result = ''.join(self.trace[case_id])
        print(result)
        fitness = self.dc[case_id][0]
        #置信度由上界和下界宽度决定
        confidence = (self.dc[case_id][3] * (1 - ( self.dc[case_id][2] - self.dc[case_id][1])))
        completeness = self.dc[case_id][4]

        print(fitness)
        self.data = self.data.append({
            "案例id": case_id,
            "轨迹": result,
            "Conformance": fitness,
            "置信度": confidence,
            "完整度": completeness,
        }, ignore_index=True)
        #记录
        # Excel 文件路径



    #轨迹开始阶段
    def update_prefix_alignment(self, case_id, event, subset, alignments_dicf, model_fitness, min_model):
        # This method should be implemented based on the specific details of the algorithm
        self.trace[case_id].append(event)
        result = ''.join(self.trace[case_id])
        # print(result)
        #计算拟合度
        # self.dc[case_id] = calculate_alignment_cost(self.trace[case_id], process_mode, im, fm)
        self.dc[case_id] = calculate_appr_online_alignment_cost(result, subset, alignments_dicf, model_fitness, min_model)
        print(self.dc[case_id])
        # self.dc[case_id] = pm4py.fitness_alignments(self.trace[case_id],process_mode,im,event)
        # print(self.trace[case_id])
        pass


    def create_new_prefix_alignment(self, case_id, event, subset, alignments_dicf, model_fitness, min_model):
        # This method should be implemented based on the specific details of the algorithm
        self.trace[case_id] = []
        self.trace[case_id].append(event)
        result = ''.join(self.trace[case_id])
        self.dc[case_id] = calculate_appr_online_alignment_cost(result, subset, alignments_dicf, model_fitness, min_model)
        print(self.dc[case_id])
        print("创建记录")
        # self.dc[case_id] = pm4py.fitness_alignments(self.trace[case_id], process_mode, im, event)
        pass

    #遗忘的检测信息
    def forget_case(self):
        # Determine the case to forget based on the forgetting criteria
        case_to_forget = self.select_case_to_forget()
        # Store the summary state in the repository
        self.rc[case_to_forget] = self.dc[case_to_forget].get_summary_state()
        # Remove the case from the active memory
        del self.dc[case_to_forget]

    def select_case_to_forget(self):
        # This method should implement the forgetting criteria logic
        # For now, we'll just return a random case_id
        import random
        return random.choice(list(self.dc.keys()))


    #返回当前的检测的全部数据
    def check_evaluation(self):

        return self.dc

#将事件日志轨迹顺序打乱
def stream_mix(data):
    # 按数字元素分组
    groups = [list(group) for _, group in itertools.groupby(data, lambda x: x[0])]

    # 打乱分组
    random.shuffle(groups)
    # 重新分配数字
    new_data = []
    for new_num, group in enumerate(groups):
        new_data.extend([[new_num, item[1]] for item in group])
    return new_data
#打乱事件日志中的轨迹
def streamtostream(tracks):
    # 将事件日志分组
    grouped_arrays = {}
    for item in tracks:
        key = item[0]
        if key not in grouped_arrays:
            grouped_arrays[key] = []
        grouped_arrays[key].append(item)

    # 打印分组后的字典
    # print(grouped_arrays)

    # 将字典转换为列表
    grouped_list = list(grouped_arrays.values())
    # print(grouped_list)

    # 创建一个新的空数组用于合并
    combined_array = []

    # 自动创建索引列表
    indexes = [0] * len(grouped_list)

    # 自动计算数组长度
    array_lengths = [len(sub_array) for sub_array in grouped_list]

    # 循环直到所有原数组的元素都被添加到新数组中
    while sum(indexes) < sum(array_lengths):
        # 随机选择一个还有剩余元素的原数组
        array_choice = random.choice([i for i in range(len(grouped_list)) if indexes[i] < array_lengths[i]])
        # 从选中的原数组中取出当前索引位置的元素，添加到新数组中
        combined_array.append(grouped_list[array_choice][indexes[array_choice]])
        # 更新选中原数组的索引位置
        indexes[array_choice] += 1
    #返回模拟事件流
    return combined_array

#只考虑插入和删除操作的编辑距离
def edit_distance(x,y):
    #return math.fabs(1-tfidf_similarity(x,y)*(len(x) + len(y)) - (len(x) + len(y)))
    return math.fabs(Levenshtein.ratio(x, y) * (len(x) + len(y)) - (len(x) + len(y)))
#计算近似对齐的拟合度subset, modelsubset,list(set(variants_list) - set(subset)), variants_dic, min_model, modelsubset_fitness



from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments_algorithm
from pm4py.util import constants
from pm4py.objects.petri_net.obj import PetriNet, Marking
# 近似对齐
#frequency_list1,frequency_list,variants_list,variants_dic,min_model,fitness,alignments_dicf
def calculate_appr_alignment_cost(trace,log_sub, alignments_dicf, model_fitness, min_model):
    fitness_dic1 = {}  # 对应对齐下界
    fitness_dic2 = {}  # 对齐上界
    fitness_list = {}  # 近似拟合度
    sum1_1 = 0;
    sum1_2 = 0;
    sum2 = 0;
    sum = 0
    min = sys.maxsize
    t1 = ""
    for y in log_sub:
        # print("x=",x," y=",y)
        # print('edit_distance(x,y)=',edit_distance(x,y))
        t = edit_distance(trace, y)
        if min > t:
            min = t
            t1 = y
    # print("x=",x,"y=",t1,"min=",min,"alignments_dic[x]=",alignments_dic[x])
    print("=====")
    print(t1)
    if min == 0:
        fitness_dic1[trace] = alignments_dicf[trace]
        return alignments_dicf[trace], alignments_dicf[trace], alignments_dicf[trace]
    else:
        fitness_dic1[trace] = 1 - min / (min_model + len(trace))

    if fitness_dic1[trace] < model_fitness:
        fitness_list[trace] = model_fitness
    else:
        fitness_list[trace] = fitness_dic1[trace]

    if len(trace) < min_model:
        fitness_dic2[trace] = 1 - (min_model - len(trace)) / (min_model + len(trace))
    else:
        fitness_dic2[trace] = 1
    #计算轨迹的近似拟合度
    sum = fitness_list[trace]
    sum1_1 = fitness_dic1[trace]
    sum1_2 =  fitness_dic2[trace]
    # alignments_dicf[x]
    sum2 = 1
    fitness_value = sum / sum2
    # print(sum2) L U
    return fitness_value, sum1_1 / sum2, sum1_2 / sum2



from difflib import SequenceMatcher
def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

import heapq
def edit_sort(trace,log_sub):
    # 使用堆结构来存储相似度得分和单词
    heap = []
    # 计算数组中每个单词与目标字符串的相似度，并将其添加到堆中
    for word in log_sub:
        score = similarity(trace, word)
        # 使用负数，因为heapq是最小堆，这样可以得到最大的相似度得分
        heapq.heappush(heap, (-score, word))

    # 获取相似度得分最高的前三个轨迹
    top_3_words = [heapq.heappop(heap) for _ in range(6)]
    # print("比较编辑距离")
    # print(trace)
    # # 输出结果
    # for score, word in top_3_words:
    #     print(f"'{word}' - 编辑距离为 {-score}")
    return top_3_words


def calculate_appr_online_alignment_cost(trace,log_sub, alignments_dicf, model_fitness, min_model):
    fitness_dic1 = {}  # 对应对齐下界
    fitness_dic2 = {}  # 对齐上界
    fitness_list = {}  # 近似拟合度
    sum1_1 = 0;
    sum1_2 = 0;
    sum2 = 0;
    sum = 0
    min = sys.maxsize
    t1 = ""; t2 = ""; t3 = ""
    # print(trace)
    # top_n_trace = edit_sort(trace,log_sub)
    # print(top_n_trace[1][1])
    # min = edit_distance(top_n_trace[1][1],trace)
    # print(min)
    # t1 = top_n_trace[1][1]
    for y in log_sub:
        # print("x=",x," y=",y)
        # print('edit_distance(x,y)=',edit_distance(x,y))
        #
        t = edit_distance(trace, y)
        if min > t:
            min = t
            t1 = y
    # print("x=",x,"y=",t1,"min=",min,"alignments_dic[x]=",alignments_dic[x])
    print("=====11=====")
    #比较轨迹长度判断是事件流中的成分比
    stage = len(trace)/len(t1)
    completeness = len(set(trace)) / len(set(t1))
    # print(confidence)
    if stage > 1:
        stage = 1
    if stage <= 0.4:
        print("轨迹预计处于开始阶段")
        for log in log_sub:
            if edit_distance(trace,log[:len(trace)]) == 0:
                t1 = log
                return alignments_dicf[t1], alignments_dicf[t1], alignments_dicf[t1], stage, completeness

        #简单测试轨迹是否相符
    elif stage <= 0.9:
        print("轨迹预计处于中间")
        #提取多个相似完整轨迹作为参考，计算拟合度
        for log in log_sub:
            if edit_distance(trace,log[:len(trace)]) == 0:
                t1 = log
                return alignments_dicf[t1], alignments_dicf[t1], alignments_dicf[t1],stage, completeness
        #如果不在则计算多条轨迹，获得其近似拟合度
        else:
            top_n_trace = edit_sort(trace, log_sub)
            number = []
            completeness = 0
            for score, word in top_n_trace[:math.ceil(3/stage)]:
                print(f"'{word}' - 编辑距离为 {-score}")
                print(alignments_dicf[word])
                number.append(alignments_dicf[word])
                sum = alignments_dicf[word] + sum
                completeness = completeness + len(set(word))
            print(number)
            max_fitness = max(number)
            min_fitness =  np.amin(np.array(number))
            fitness = sum/len(number)
            completeness = len(set(trace)) / (completeness / len(number))
            if completeness > 1:
                completeness = 1
            return fitness, min_fitness, max_fitness, stage, completeness
            print(top_n_trace)

    else:
        print("轨迹处于结束阶段")
        # 提取多个相似完整轨迹作为参考，计算拟合度
        for log in log_sub:
            if edit_distance(trace, log[:len(trace)]) == 0:
                t1 = log
                return alignments_dicf[t1], alignments_dicf[t1], alignments_dicf[t1], stage, completeness
        # 如果不在则计算多条轨迹，获得其近似拟合度
        else:
            top_n_trace = edit_sort(trace, log_sub)
            number = []
            completeness = 0
            for score, word in top_n_trace[:math.ceil(2 / stage)]:
                print(f"'{word}' - 编辑距离为 {-score}")
                print(alignments_dicf[word])
                number.append(alignments_dicf[word])
                sum = alignments_dicf[word] + sum
                completeness = completeness + len(set(word))
            print(number)
            max_fitness = max(number)
            min_fitness = np.amin(np.array(number))
            fitness = sum / len(number)
            completeness= len(set(trace)) / (completeness / len(number))
            if completeness > 1:
                completeness = 1
            return fitness, min_fitness, max_fitness, stage, completeness
        #与近似一致性检测相似，给予上下界，该阶段的轨迹可以随时看作即将结束

    top_n_trace = edit_sort(trace, log_sub)

    if min == 0:
        fitness_dic1[trace] = alignments_dicf[t1]
        return alignments_dicf[t1], alignments_dicf[t1], alignments_dicf[t1], stage, completeness
    else:
        # fitness_dic1[trace] = 1 - min / (min_model + len(trace))
        fitness_dic1[trace] = 1 - min / (min_model + len(t1))

    if fitness_dic1[trace] < model_fitness:
        fitness_list[trace] = model_fitness
    else:
        fitness_list[trace] = fitness_dic1[trace]

    if len(trace) < min_model:
        fitness_dic2[trace] = 1 - (min_model - len(trace)) / (min_model + len(trace))
    else:
        fitness_dic2[trace] = 1
    #计算轨迹的近似拟合度
    sum = fitness_list[trace]
    sum1_1 = fitness_dic1[trace]
    sum1_2 =  fitness_dic2[trace]
    # alignments_dicf[x]
    sum2 = 1
    fitness_value = sum / sum2
    # print(sum2) L U
    return fitness_value, sum1_1 / sum2, sum1_2 / sum2, stage, completeness


