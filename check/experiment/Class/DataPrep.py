import pandas as pd
import os
import numpy as np
import editdistance
import time
from itertools import combinations
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD

class DataPreparation():
    ca = 'case:concept:name'
    ac = 'concept:name'
    ts = 'time:timestamp'
    pickles_folder = 'D:/WorkSpace/sampling-master/experiment/pickles/'.format(os.path.abspath(os.curdir))
    #判读是否是文件目录
    if not os.path.isdir(pickles_folder):
        #创建目录
        os.mkdir(pickles_folder)

    def __init__(self, name, path, ca=ca, ac=ac, ts=ts, forceReload=True):
        '''
        Prepare the data (load csv, build distance matrices, build signature)
        and export pickles (so we don't have to load them again if forceReload is False)
        :param name: name of the dataset
        :param path: path of the csv
        :param ca: column of the csv corresponding to case identifier
        :param ac: column of the csv corresponding to activity
        :param ts: column of the csv corresponding to timestamp
        :param forceReload: If True, will create the pickles. If False it will load existing pickles
        '''
        self.name = name
        #print(name)
        self.path = path
        #print(path)
        self.ca = ca
        #print(ca)
        self.ac = ac
       # print(ac)
        self.ts = ts
        self.activitys=[]
       # print(ts)
        self.distanceMatrix, self.signature, self.variants, self.facts = [None]*4
        if forceReload:
            self.activitys, self.distanceMatrix, self.signature, self.variants, self.facts = self.load_from_csv()
            np.save('{}{}_dm.pickle.npy'.format(self.pickles_folder, self.name), self.distanceMatrix)
            np.save('{}{}_sig.pickle.npy'.format(self.pickles_folder, self.name), self.signature)
            self.variants.to_pickle('{}{}_var.pickle'.format(self.pickles_folder, self.name))
            self.facts.to_pickle('{}{}_facts.pickle'.format(self.pickles_folder, self.name))
        else:
            self.distanceMatrix = np.load('{}{}_dm.pickle.npy'.format(self.pickles_folder, self.name), allow_pickle=True)
            self.signature = np.load('{}{}_sig.pickle.npy'.format(self.pickles_folder, self.name), allow_pickle=True)
            self.variants = pd.read_pickle('{}{}_var.pickle'.format(self.pickles_folder, self.name))
            self.facts = pd.read_pickle('{}{}_facts.pickle'.format(self.pickles_folder, self.name))

    def load_from_csv(self):
        facts = {}

        # Load the CSV
        s = time.time()
        df = self.load_csv()
        facts['time_load_csv'] = time.time()-s

        # Extract variants
        s = time.time()
        variants = self.extract_variants(df)
        facts['time_extract_variants'] = time.time()-s

        # 活动
        activitys = self.extract_activitys(df)
        # Build the distance matrix between variants
        s = time.time()
        #表示不同的轨迹['Request For Payment SUBMITTED by EMPLOYEE', 'Request For Payment APPROVED by ADMINISTRATION',
        # 'Request For Payment FINAL_APPROVED by SUPERVISOR', 'Request Payment', 'Payment Handled']
        distanceMatrix = self.buildDistanceMatrix(variants['seq'].tolist())
        #print(variants['seq'].tolist())
        facts['time_build_distance_matrix'] = time.time()-s

        s = time.time()
        signature = self.buildSignature(variants['seq'].tolist())
        facts['time_to_build_signature'] = time.time()-s

        # Extract more facts about the dataset
        # (for descriptive statistics purpose)
        facts['dataset'] = self.name
        facts['ds_n_variants'] = variants.shape[0]
        print('ds_n_variants:',facts['ds_n_variants'])
        #df.shape[0]放回行数
        facts['ds_n_events'] = df.shape[0]
        print('ds_n_events:',facts['ds_n_events'])
        #有多少不同的活动
        facts['ds_n_unique_activity'] = df[self.ac].nunique()
        print("ds_n_unique_activityfacts:",facts['ds_n_unique_activity'])
        facts['ds_n_unique_trace'] = df[self.ca].nunique()
        print('ds_n_unique_trace',facts['ds_n_unique_trace'])
        facts['cov_top5_vars'] = variants.head(5)['count'].sum()/variants['count'].sum()
        facts['cov_top10_vars'] = variants.head(10)['count'].sum()/variants['count'].sum()
        facts['cov_top20_vars'] = variants.head(20)['count'].sum()/variants['count'].sum()
        facts['cov_top50_vars'] = variants.head(50)['count'].sum()/variants['count'].sum()
        facts['average_levenshtein'] = distanceMatrix.mean()
        facts = pd.Series(facts)

        return activitys, distanceMatrix, signature, variants, facts


    def load_csv(self):
        #nrows 读取的行数,low_memory而一旦设置low_memory=False，那么pandas在读取
        # csv的时候就不分块读了，而是直接将文件全部读取到内存里面，
        df = pd.read_csv(self.path,encoding='ISO-8859-1', nrows=None, low_memory=False)
        #time:timestamp
        #print(df[self.ts]);
        df[self.ts] = pd.to_datetime(df[self.ts])
        #print(df[self.ts])
        #进行排序，Id和完成时间
        df.sort_values([self.ca, self.ts], inplace=True)
        df = df[[self.ca,self.ac]]
        #对行和列进行选择，df.notna().all(axis=1)全为1才为ture
        df = df.loc[df.notna().all(axis=1),:]
        #print(df) # ID request for payment 147529      ac  Request For Payment SUBMITTED by EMPLOYEE
        return df

    def extract_variants(self, df):
        variants = df.groupby(self.ca)[self.ac].agg(list)\
            .value_counts().reset_index().rename({'index':'seq', self.ac:'count'}, axis=1)
        variants['length'] = variants['seq'].str.len()
        #print(variants)
        return variants
    #定义活动的不同种类
    def extract_activitys(self, df):
        activitys=df[self.ac].tolist()
        # print(activitys)
        # print(len(activitys))
        return activitys

    def distance_function(self, x1, x2):
        return editdistance.eval(x1, x2) / max([len(x1), len(x2)])

    def buildDistanceMatrix(self, seq):
        #初始矩阵，里面的默认值为0.0,为numpy.float64
        m = np.zeros([len(seq), len(seq)])
        #组合，combinations(iterable, r)方法可以创建一个迭代器，
        # 返回iterable中所有长度为r的子序列，返回的子序列中的项按输入iterable中的顺序排序
        #例如 list1 = [1, 3, 4, 5],list2 = list(itertools.combinations(list1, 2)),[(1, 3), (1, 4), (1, 5), (3, 4), (3, 5), (4, 5)]
        #range()：当传入两个参数时，则将第一个参数做为起始位，第二个参数为结束位 range(0,5) [0, 1, 2, 3, 4]
        for x, y in combinations(range(0,len(seq)), 2):
            d = self.distance_function(seq[x], seq[y])
            m[x,y] = d
            m[y,x] = d
        for x in range(len(seq)):
            m[x,x] = 0
        #数据类型的转换，转换为float64类型
        return m.astype(np.float64)

    def buildSignature(self, seq):
        cv = CountVectorizer(ngram_range=(1,2), tokenizer=lambda doc: doc, lowercase=False, max_features=1024)
        data = cv.fit_transform([['$$START$$']+x+['$$END$$'] for x in seq])
        data = TruncatedSVD(min(64, int(data.shape[1]/2)+1)).fit_transform(data).astype(np.float32)
        return data

    def randomlyReOrder(self, seed):

        # The order of the distance matrix
        # or signature will influence the results
        # For reproducibility purpose, we manage the orderbuildSignature
        #
        np.random.seed(seed)
        self.variants['random'] = np.random.random(self.variants.shape[0])
        new_order = self.variants.sort_values(['count', 'random'], ascending=False).index
        self.variants = self.variants.loc[new_order,:].reset_index()
        if self.distanceMatrix is not None:
            self.distanceMatrix = self.distanceMatrix[new_order,:][:,new_order]
            self.distanceMatrix = np.ascontiguousarray(self.distanceMatrix, dtype=np.float64)
        if self.signature is not None:
            self.signature = self.signature[new_order,:]
            self.signature = np.ascontiguousarray(self.signature[new_order,:])





