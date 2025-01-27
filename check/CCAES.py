import itertools
import profile
import random
import TracePrefixAlignment
import pm4py
from collections import defaultdict
from DataPrep import DataPreparation
from datasets import datasets
import time
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from pm4py.objects.log.importer.xes import importer as xes_importer
import pandas as pd
from cminSampler import CminSampler
import Levenshtein
import math
import sys

#把轨迹转换为子字符串
def trace_str(variants ,activity_dic):
    variants_list = []
    stream_list = []
    num = 0
    for i in range(len(variants)):
        for j in range(len(variants[i])):
            if variants[i][j] in activity_dic:
                pass
            else:
                print("not exit",variants[i][j])
            variants[i][j] = activity_dic[variants[i][j]]
        variants_list.append("".join(variants[i]))
    return variants_list

#将事件日志轨迹顺序打乱
def stream_mix(data):
    # 按数字元素分组
    groups = [list(group) for _, group in itertools.groupby(data, lambda x: x[0])]

    # 设置随机种子 BPIC2020_1、BPIC2013_2
    # random.seed(89)
    # random.seed(13)
    # 打乱分组
    random.shuffle(groups)
    # 重新分配数字
    new_data = []
    for new_num, group in enumerate(groups):
        new_data.extend([[new_num, item[1]] for item in group])
    return new_data


#控制最大同时轨迹,在这里进行在线监测
def simulation_streamlog(data,k):
    # 初始化分组列表和当前分组
    grouped_data = []
    current_group = []
    # 上一个数字元素
    previous_num = None
    # 遍历数组并分组
    for item in data:
        current_num = item[0]
        # 如果当前数字元素与上一个数字元素相同，或者当前分组为空（即开始新的分组）
        if current_num == previous_num or not current_group:
            current_group.append(item)
        else:
            # 如果当前数字元素不同，且当前分组已有五个不同的数字元素，则保存当前分组并开始新的分组
            if len(set([x[0] for x in current_group])) == k:
                grouped_data.append(current_group)
                current_group = [item]
            else:
                # 如果当前分组中不同数字元素少于五个，继续添加到当前分组
                current_group.append(item)
        previous_num = current_num
    # 添加最后一个分组（如果有）
    if current_group:
        grouped_data.append(current_group)
    return grouped_data


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

from pm4py.objects.petri_net.exporter import exporter as pnml_exporter
from pm4py.objects.log.exporter.xes import exporter as xes_exporter

#发现过程模型
def discover_petri_net(trace_data):
    # 将轨迹数据转换为pandas DataFrame
    df = pd.DataFrame(trace_data, columns=["case:concept:name", "concept:name"])
    print(df)
    # 将DataFrame转换为pm4py可以理解的事件日志
    log = log_converter.apply(df,variant=log_converter.Variants.TO_EVENT_LOG)

    # 使用Alpha Miner算法发现过程模型

    print("发现过程模型")
    net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(log, noise_threshold=0.7)
    t = t = time.time_ns()
    aligned_traces = alignments.apply_log(log, net, initial_marking, final_marking)
    t = (time.time_ns() - t) / 10 ** 9
    # 使用原事件日志发现过程模型与使用事件流生成的过程模型在只考虑轨迹的情况下没有区别
    pnml_exporter.apply(net, initial_marking, './Spearman_xlsx/test/petri_model.pnml')
    xes_exporter.apply(log, './Spearman_xlsx/test/log.xes')
    # print("正在发现过程模型")
    # log1 = xes_importer.apply('./data/RequestForPayment.xes')  # 新变量采样最好
    # net1, initial_marking1, final_marking1 = pm4py.discover_petri_net_inductive(log1, noise_threshold=0.7)
    # fitness2 = pm4py.conformance.fitness_alignments(log1,net1, initial_marking1, final_marking1)
    # 可视化过程模型
    from pm4py.visualization.petri_net import visualizer as pn_visualizer
    gviz = pn_visualizer.apply(net, initial_marking, final_marking)
    pn_visualizer.view(gviz)
    # gviz1 = pn_visualizer.apply(net1, initial_marking1, final_marking1)
    # pn_visualizer.view(gviz1)
    return net, initial_marking, final_marking, log, aligned_traces, t

# 增量前缀对齐算法
def incremental_prefix_alignment(event_stream, petri_net, initial_marking, final_marking):
    case_alignments = {}  # 存储案例的前缀对齐结果
    for event in event_stream:
        case_id, activity = event
        if case_id not in case_alignments:
            case_alignments[case_id] = [initial_marking]
        current_marking = case_alignments[case_id][-1]
        # 查找与活动对应的转换
        transition = next((t for t, act in petri_net.transitions.items() if act == activity), None)
        if transition and petri_net.is_enabled(transition, current_marking):
            # 如果转换可触发，则进行同步移动
            new_marking = petri_net.fire(transition, current_marking)
            case_alignments[case_id].append(new_marking)
        else:
            # 如果转换不可触发，记录活动移动
            case_alignments[case_id].append(activity)
    return case_alignments

#计算近似对齐的拟合度subset, modelsubset,list(set(variants_list) - set(subset)), variants_dic, min_model, modelsubset_fitness
def collect_fitness(frequency_list1,frequency_list,variants_list,variants_dic,min_model,fitness,alignments_dicf,alignments_dic):
    fitness_dic1={}#对应对齐下界
    fitness_dic2 = {}#对齐上界
    fitness_list={}#近似拟合度
    sum1_1 = 0;sum1_2 = 0;sum2 = 0;sum=0
    for x in variants_list:
        #取一个最大整数
        min = sys.maxsize
        t1=""
        #计算矩阵
        for y in frequency_list:
            # print("x=",x," y=",y)
            # print('edit_distance(x,y)=',edit_distance(x,y))
            t=edit_distance(x,y)
            if min>t:
                min=t
                t1=y
        # print("x=",x,"y=",t1,"min=",min,"alignments_dic[x]=",alignments_dic[x])
        fitness_dic1[x]=1-min/(min_model+len(x))
        #实际拟合度与fitness_dic的差距
        # if alignments_dicf[x]<fitness_dic1[x]-0.0000001:
        #             print("fitness_dic1[x]=", fitness_dic1[x])
        #             print("alignments_dicf[x]=", alignments_dicf[x])
        #             if min==0:
        #                 print("alignments_dic[t1]=",alignments_dic[t1])
        #print(alignments_dicf[x]>=fitness_dic1[x])
        #近式fitness
        if fitness_dic1[x]<fitness:
            fitness_list[x]=fitness
        else:
            fitness_list[x]=fitness_dic1[x]

        if len(x)<min_model:
            fitness_dic2[x]=1-(min_model-len(x))/(min_model+len(x))
        else:
            fitness_dic2[x] = 1
    #需要优化 sum是近似值，sum1_1是下界，sum2_1下界
    for x in variants_list:
        sum = sum + fitness_list[x] * variants_dic[x]
        sum1_1=sum1_1+fitness_dic1[x] * variants_dic[x]
        sum1_2= sum1_2 + fitness_dic2[x] * variants_dic[x]
        # alignments_dicf[x]
        sum2= sum2 + variants_dic[x]
        # print(variants_dic[x])
        # print(fitness_list[x])

    #其中frequency_list有问题
    for x in frequency_list1:
        sum = sum + alignments_dicf[x] * variants_dic[x]
        sum1_1 = sum1_1 + alignments_dicf[x] * variants_dic[x]
        sum1_2 = sum1_2 + alignments_dicf[x] * variants_dic[x]
        sum2 = sum2 + variants_dic[x]
    # print("sum=",sum,"sum1_1=",sum1_1,"sum1_2=",sum1_2,"sum2=",sum2)
    fitness_value = sum / sum2
    # print(sum2)
    # print(sum)
    return fitness_value,sum1_1/sum2,sum1_2/sum2


#只考虑插入和删除操作的编辑距离
def edit_distance(x,y):
    #return math.fabs(1-tfidf_similarity(x,y)*(len(x) + len(y)) - (len(x) + len(y)))
    return math.fabs(Levenshtein.ratio(x, y) * (len(x) + len(y)) - (len(x) + len(y)))

#只考虑插入和删除操作的编辑距离，用于采样的归一化处理
def edit_distance_sample(x,y):
    #return math.fabs(1-tfidf_similarity(x,y)*(len(x) + len(y)) - (len(x) + len(y)))
    return (math.fabs(Levenshtein.ratio(x, y) * (len(x) + len(y)) - (len(x) + len(y))))/max(len(x),len(y))

def alignments_dict(aligned_traces):
    # 日志对应的移动序列
    l1 = []
    fitness_list = []
    for i in range(len(aligned_traces)):
        t = []
        for x in aligned_traces[i]['alignment']:
            if x[0] != ">>" and x[0] != None:
                t.append(x[0])
        l1.append(t)
        fitness_list.append(aligned_traces[i]['fitness'])
    print(aligned_traces[3]['fitness'])
    # print(l1[1])
    # print(l1[2])
    l1 = [''.join(sublist) for sublist in l1]
    l2 = []
    for i in range(len(aligned_traces)):
        t = []
        for x in aligned_traces[i]['alignment']:
            if x[1] != ">>" and x[1] != None:
                t.append(x[1])
        l2.append(t)
    l2 = [''.join(sublist) for sublist in l2]
    alignments_dic = {}
    alignments_dicf = {}
    for i in range(len(l1)):
        alignments_dic[l1[i]] = l2[i]
        alignments_dicf[l1[i]] = fitness_list[i]
    return alignments_dic, alignments_dicf



import tracemalloc

from collections import Counter

def main():
    for ds in datasets:
        data = DataPreparation(forceReload=True, **ds)
        activitys = data.activitys
        unique_activity = set()
        for x in activitys:
            unique_activity.add(x)
        print(unique_activity)
        # print(len(unique_activity))
        # 活动的字典表
        activity_dic = {}
        t = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ叧叨叭叱叴叵叺叻叼叽叾卟叿吀吁吂吅吆吇吋吒吔吖吘吙吚吜吡吢吣吤吥' \
            '吧吩吪吭吮吰吱吲呐吷吺吽呁呃呄呅呇呉呋呋呌呍呎呏呐呒呓呔呕呗呙呚呛呜呝呞呟呠呡呢呣呤呥呦呧周呩呪呫呬呭呮呯呰呱呲呴呶呵呷呸' \
            '呹呺呻呾呿咀咁咂咃咄咅咇咈咉咊咋咍咎咐咑咓咔咕咖咗咘咙咚咛咜咝咞咟咠咡咢咣咤咥咦咧咨咩咪咫咬咭咮咯咰咲咳咴咵咶啕咹咺咻呙咽' \
            '咾咿哂哃哅哆哇哈哊哋哌哎哏哐哑哒哓哔哕哖哗哘哙哚哛哜哝哞哟哠咔哣哤哦哧哩哪哫哬哯哰唝哵哶哷哸哹哻哼哽哾哿唀唁唂唃呗唅唆唈唉' \
            '唊唋唌唍唎唏唑唒唓唔唣唖唗唘唙吣唛唜唝唞唟唠唡唢唣唤唥唦唧唨唩唪唫唬唭唯唰唲唳唴唵唶唷念唹唺唻唼唽唾唿啀啁啃啄啅啇啈啉啋啌' \
            '啍啎问啐啑啒启啔啕啖啖啘啙啚啛啜啝哑启啠啡唡衔啥啦啧啨啩啪啫啬啭啮啯啰啱啲啳啴啵啶啷啹啺啻啼啽啾啿喀喁喂喃善喅喆喇喈喉喊喋' \
            '喌喍喎喏喐喑喒喓喔喕喖喗喙喛喞喟喠喡喢喣喤喥喦喨喩喯喭喯喰喱哟喳喴喵営喷喸喹喺喼喽喾喿嗀嗁嗂嗃嗄嗅呛啬嗈嗉唝嗋嗌嗍吗嗏嗐嗑嗒' \
            '嗓嗕嗖嗗嗘嗙呜嗛嗜嗝嗞嗟嗠嗡嗢嗧嗨唢嗪嗫嗬嗭嗮嗰嗱嗲嗳嗴嗵哔嗷嗸嗹嗺嗻嗼嗽嗾嗿嘀嘁嘂嘃嘄嘅嘅嘇嘈嘉嘊嘋嘌喽嘎嘏嘐嘑嘒嘓呕嘕啧嘘' \
            '嘙嘚嘛唛嘝嘞嘞嘟嘠嘡嘢嘣嘤嘥嘦嘧嘨哗嘪嘫嘬嘭唠啸囍嘴哓嘶嘷呒嘹嘺嘻嘼啴嘾嘿噀噂噃噄咴噆噇噈噉噊噋噌噍噎噏噐噑噒嘘噔噕噖噗噘噙噚噛' \
            '噜咝噞噟哒噡噢噣噤哝哕噧噩噪噫噬噭噮嗳噰噱哙噳喷噵噶噷吨噺噻噼噽噾噿咛嚁嚂嚃嚄嚅嚆吓嚈嚉嚊嚋哜嚍嚎嚏尝嚑嚒嚓嚔噜嚖嚗嚘啮嚚嚛嚜嚝' \
            '嚞嚟嚠嚡嚢嚣嚤呖嚧咙嚩咙嚧嚪嚫嚬嚭嚯嚰嚱亸喾嚵嘤嚷嚸嚹嚺嚻嚼嚽嚾嚿啭嗫嚣囃囄冁囆囇呓囊囋囍囎囏囐嘱囒啮囔囕囖'
        print("字母表：", t, len(t))
        i = 0
        for x in unique_activity:
            activity_dic[x] = t[i]
            i = i + 1
        print(activity_dic)
        variants = data.variants['seq'].tolist()
        variants_data = data.variants['count'].tolist()
        # print(variants)
        # print(variants_data)

        #把trace转换成字符串
        variants_list = trace_str(variants,activity_dic)
        # print("variants:", variants)
        print("variants_list:", variants_list)
        # 创建变量字典
        variants_dic = {}
        stream = []
        stream_event = []
        stream_log = []
        logcaseid = []
        i = 0
        caseid = 0

        # #采样
        # trace = []
        # for x in variants_list:
        #     for y in range(variants_dic[x]):
        #         trace.append(x)
        # sampler = CminSampler(trace, number, variants_list)
        # print("矩阵采样")
        # print(sampler)
        for x in variants_list:
            variants_dic[x] = variants_data[i]
            print(x)
            dataint = int(variants_data[i])
            for num in range (variants_data[i]):
                stream.append(x)
                num = num + 1
                logcaseid.append(caseid)
                caseid = caseid + 1
            # print(stream)
            i = i + 1

        for num in range (caseid):
            for char in stream[num]:
                x1 = []
                x1.append(num)
                x1.append(char)
                stream_event.append(x1)
        # print(stream_event)
        #根据第一个元素选取分组

        # 根据第一个元素分组
        #测试数据集
        #输入k为事件流最大同时出现的轨迹数量
        k = 10000
        # sk为事件流设置的事件日志倍数
        sk = 1
        stream_log_sk = stream_event*sk
        stream_log_sk = stream_mix(stream_log_sk)
        stream_log = stream_event
        stream_grouped_data_sample = simulation_streamlog(stream_log, k)
        stream_grouped_data = simulation_streamlog(stream_log_sk,k)
        # #压力测试
        # stream_grouped_data1 = simulation_streamlog(stream_log,k)
        # stream_grouped_data2 = simulation_streamlog(stream_log, k)
        # stream_grouped_data3 = simulation_streamlog(stream_log, k)
        # print(streamtostream(tracks))
        print("variants_dic:", variants_dic)
        net, im, fm, log, aligned_traces ,a_time= discover_petri_net(stream_event)
        # 计算标准的最优对齐
        # aligned_traces = alignments.apply_log(log,net,im,fm)
        #获取过程模型..........
        alignments_dic, alignments_dicf = alignments_dict(aligned_traces)
        print("对齐字典的长度：", len(alignments_dic))
        print("对齐字典的长度：", len(alignments_dicf))
        print(alignments_dic)
        print(alignments_dicf)
        sum1 = 0;
        sum2 = 0
        sum_sample = 0
        for x in variants_list:
            if len(x) < 5000:
                sum1 = sum1 + alignments_dicf[x] * variants_dic[x]
                sum2 = sum2 + variants_dic[x]
            else:
                sum1 = sum1
                sum2 = sum2 + variants_dic[x]
        # print("===========alignments===========")
        # print(alignments_dicf[x])

        print("sum1=", sum1, "sum2=", sum2)
        print("标准方法的fitness为", sum1 / sum2)
        subset = [" "]
        min_model = 0
        modelsubset = ['']
        model_dic = {}
        testtrace = []
        sample_index = 1
        t_sa = 0
        t = time.time_ns()
        trace = ""
        # 模拟事件流 并进行采样
        classified_tuples = {}
        for index, group in enumerate(stream_grouped_data_sample):
            # print(f"Group {index + 1}: {group}")
            #获得事件流
            eventstream = streamtostream(group)
            # # print(eventstream)
            opa = OnlinePrefixAlignment.OnlinePrefixAlignment(max_cases=10000)
            # # print(eventstream)
            #用于在线一致性检测
            # for item in eventstream:
            #     print(item)
            #     # 事件流检测
            #     # opa.process_event(item[0], item[1], net, im, fm)
            #
            # 在此可以进行事件流检测
            for tup in streamtostream(group):
                if tup[0] not in classified_tuples:
                    classified_tuples[tup[0]] = tup[1]
                else:
                    classified_tuples[tup[0]] += tup[1]
            # 遍历字典，构建每个数字对应的字符串，当多次无增加轨迹时跳跃采样
            modelsubset_fitness = 0
            # t = time.time_ns()
            # 创建一个空字典来记录每个 combined_string 的出现次数
            string_counts = {}
            for number, combined_string in classified_tuples.items():
                # 计算轨迹与子集中每个字符串的编辑距离，并找出最小值
                min_distance = min(edit_distance_sample(combined_string, s) for s in subset)
                trace = combined_string
                print(subset)
                ##
                if min_distance > 0.3:
                    subset.append(combined_string)
                    model_dic[combined_string] = alignments_dic[combined_string]
                    modelsubset.append(alignments_dic[combined_string])
                    modelsubset_fitness = modelsubset_fitness + (alignments_dicf[combined_string] * variants_dic[combined_string])
                    sum_sample = sum_sample + variants_dic[combined_string]
                # print(f"最小编辑距离是: {min_distance}")
                # print(f"案例 {number} 对应的轨迹: {combined_string}")
                # 如果轨迹的信息与事件日志中差太多
                if combined_string not in string_counts:
                    string_counts[combined_string] = 1
                    # 如果已经存在，增加其计数
                else:
                    string_counts[combined_string] += 1
        # 使用 sorted() 函数和 lambda 表达式按值从大到小排序
        sorted_data = sorted(string_counts.items(), key=lambda item: item[1], reverse=True)
        for item in sorted_data[:math.ceil(len(sorted_data) * 0.05)]:
            if item[0] not in subset:
                subset.append(item[0])
                model_dic[item[0]] = alignments_dic[item[0]]
                modelsubset.append(alignments_dic[item[0]])
                modelsubset_fitness = modelsubset_fitness + (alignments_dicf[item[0]] * variants_dic[item[0]])
                # print(alignments_dicf[item[0]])
                sum_sample = sum_sample + variants_dic[item[0]]
            print(item[0])
        print(modelsubset_fitness)
        print(sum_sample)
        modelsubset_fitness = modelsubset_fitness/sum_sample
        t = (time.time_ns() - t) / 10 ** 9
        t_sa = t_sa + t;
        subset.remove(' ')
        print(subset)
        modelsubset = list(set(modelsubset))
        print(modelsubset)
        print("------------------------")
        print(model_dic)
        print(modelsubset_fitness)
        # 将列表保存到文件
        # with open('./savestream/modelsubset.txt', 'w', encoding='utf-8') as file:
        #     for item in modelsubset:
        #         file.write("%s\n" % item)
        # with open('./savestream/variants_list.txt', 'w', encoding='utf-8') as file:
        #     for item in variants_list:
        #         file.write("%s\n" % item)
        print(f"采样轨迹的数量: {len(subset)-1}")
        print(len(modelsubset))
        print(f"轨迹采样时间：{t}")
        print(f"对齐时间：{a_time}")
        print(t)
        #利用统计方法获得的近似拟合度
        t = time.time_ns()
        fitness_value, fitness_L, fitness_U = collect_fitness(subset, modelsubset,
                                                                 list(set(variants_list) - set(subset)),
                                                                 variants_dic, min_model, modelsubset_fitness,
                                                                 alignments_dicf, alignments_dic)
        t_f = (time.time_ns() - t) / 10 ** 9
        # print("每条轨迹对齐的值:",fitness_dic)
        # print(len(fitness_dic))
        print("近似对齐总的拟合度：", fitness_value)
        print("对齐的下界为：", fitness_L)
        print("对齐的上界为：", fitness_U)
        # print(time.time())
        print("近似对齐计算时间s：", t_f)
        print("统计采样方法的时间为s：", t_f, a_time / (t_f + a_time * len(subset) / len(variants)))
        print(subset)
        print("对齐轨迹")
        print(modelsubset)
        print(list(set(variants_list) - set(subset)))
        print(variants_dic)
        print(modelsubset_fitness)
        print(alignments_dicf)
        #trace,log_sub, model_sub, alignments_dicf, model_fitness, min_model
        # trace = 'fgygfBCuovnhsBzfAkcjptilbarxwma'
        fitness, L, U = OnlinePrefixAlignment.calculate_appr_alignment_cost(trace, subset, alignments_dicf,
                                                                                   modelsubset_fitness, 24)


        print(trace)
        print(fitness)
        print(L)
        print(U)
        # 利用统计方法获得的近似拟合度
        t = time.time_ns()
        print("单独轨迹计算==================")
        # for number, combined_string in classified_tuples.items():
        #     for i in range(len(combined_string)):
        #         fitness, L, U = OnlinePrefixAlignment.calculate_appr_online_alignment_cost(combined_string[:i], subset,
        #                                                                                    alignments_dicf,
        #                                                                                    modelsubset_fitness, 34)

        fitness, L, U, com,conf = OnlinePrefixAlignment.calculate_appr_online_alignment_cost(trace[:8], subset, alignments_dicf,
                                                                                   modelsubset_fitness, 3)

        t_online = (time.time_ns() - t) / 10 ** 9
        print("在线对齐计算时间s：",t_online)
        fitness1, L1, U1, com1, conf1 = OnlinePrefixAlignment.calculate_appr_online_alignment_cost(trace[:32], subset,
                                                                                                   alignments_dicf,
                                                                                                   modelsubset_fitness,
                                                                                                   3)
        # fitness1, L1, U1, com1, conf1 = OnlineAppro_Alignment.calculate_appr_online_alignment_cost(trace[:32], subset,
        #                                                                                            alignments_dicf,
        #                                                                                            modelsubset_fitness,
        #                                                                                            3, model_dic)
        print(fitness1)
        print(L1)
        print(U1)
        # 模拟事件流 并进行检测
        t = time.time_ns()
        fitness_test = {}
        memory_data = pd.DataFrame()
        memorycout = 0
        data = pd.DataFrame()
        for index, group in enumerate(stream_grouped_data):
            # print(f"Group {index + 1}: {group}")
            # 获得事件流
            eventstream = streamtostream(group)
            # # print(eventstream)
            opa = OnlinePrefixAlignment.OnlinePrefixAlignment(max_cases=100000)
            #在线检测
            # opa = OnlineAppro_Alignment.OnlinePrefixAlignment(max_cases=10000)
            # # print(eventstream)
            # 用于在线一致性检测
            tracemalloc.start()  # 开始跟踪内存
            for item in eventstream:
                # 事件流检测
                opa.process_event(item[0], item[1], subset, alignments_dicf, modelsubset_fitness, 3, model_dic)
                memorycout = memorycout + 1;
                if memorycout % 100 == 0:
                    snapshot = tracemalloc.take_snapshot()  # 获取快照
                    top_stats = snapshot.statistics('lineno')  # 按行统计
                    total_memory = sum(stat.size for stat in top_stats)
                    memory_data = memory_data.append({'cout': memorycout,'Menmory':total_memory / 1024}, ignore_index=True )
            # 合并数据
            snapshot = tracemalloc.take_snapshot()  # 获取快照
            top_stats = snapshot.statistics('lineno')  # 按行统计
            print("[ Top 10 memory usage ]")
            for stat in top_stats[:10]:
                print(stat)
            total_memory = sum(stat.size for stat in top_stats)
            print(f"内存使用的总和: {total_memory / 1024:.2f} KiB")
            memory_data = memory_data.append({'Menmory': total_memory / 1024}, ignore_index=True)
            fitness_test.update(opa.dc)
            data = data.append(opa.data)
        t_online = (time.time_ns() - t) / 10 ** 9
        print(data)
        print(memory_data)
        # data.to_excel('./Online_evaluation/MB.xlsx')
        # data.to_excel('./Online_evaluation/PrepaidTravelCost_1.xlsx')
        # data.to_excel('./Online_evaluation/RequestForPayment_1.xlsx')
        # memory_data.to_excel('./Spearman_xlsx/test/20_2memory_data（200w）.xlsx')
        data.to_excel('./Spearman_xlsx/test/MB.xlsx')
        # data.to_excel('./Online_evaluation/BPIC2015_2_10000.xlsx')
        # data.to_excel('./Online_evaluation/BPIC2013_2_10000.xlsx')
        # data.to_excel('./Online_evaluation/Hospital_log.xlsx')


        # print(fitness_test)
        # 计算平均值
        # 初始化总和变量
        sums = [0.0, 0.0, 0.0]
        count = 0
        for key, values in fitness_test.items():
            if values is not None:
                sums[0] += values[0]
                sums[1] += values[1]
                sums[2] += values[2]
                count += 1
        # 计算平均值
        averages = [sums[0] / count, sums[1] / count, sums[2] / count]
        print("平均值：", averages)
        print("在线对齐计算时间s：", t_online + ((len(subset)/len(alignments_dicf)) * a_time))
        print("热启动在线对齐计算时间s：", t_online)
        print(fitness)
        print(L)
        print(U)
        print("近似对齐总的拟合度：", fitness_value)
        print("对齐的下界为：", fitness_L)
        print("对齐的上界为：", fitness_U)
        print("标准方法的fitness为", sum1 / sum2)
        print("标准对齐时间", a_time)
        print("sort方法的误差",abs(averages[0] - sum1 / sum2))
        print("top方法的误差", abs(fitness_value - sum1 / sum2))
        print(sum1)
        print(sum2)

if __name__ == '__main__':
    main()
    # profile.run('main()')