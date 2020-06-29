#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import pickle
import numpy as np

from collections import defaultdict
import matplotlib.pyplot as plt
import tensorflow_hub as hub


# In[27]:


class FeatureExtractor(): 
    
    def __init__(self): 
        self.embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    def get_running_count_binary(self, data, threadheader): 
        runcountpos = []
        runcountneg = []
        distlabel0 = []
        distlabel1 = []
        c0 = 0
        c1 = 0
        d1 = 999
        d2 = 999
        prev_thread = 0
        for a,b in zip(data[threadheader], data['label']): 
            if a == prev_thread: ##still same question

                distlabel0.append(d0)
                distlabel1.append(d1)

                if b == 0: 
                    c0 += 1
                    d0 = 0
                    if d1 == 999: 
                        d1 = 999
                    else: 
                        d1 +=1
                elif b ==1: 
                    c1 += 1
                    d1 = 0
                    if d0 == 999: 
                        d0 = 999
                    else: 
                        d0 += 1
                else: 
                    print("Labels are different")
                runcountneg.append(c0) 
                runcountpos.append(c1)


            else: 
                prev_thread = a
                c0 =0
                c1= 0 
                d1 = 999
                d0 = 999
                runcountneg.append(c0)
                runcountpos.append(c1)
                distlabel0.append(d0)
                distlabel1.append(d1)

                if b == 0: 
                    d0 = 0 
                    d1= 999
                if b == 1: 
                    d1 = 1
                    d0 = 999

        return runcountneg, runcountpos, distlabel0, distlabel1

    def set_seed(self,num):
    #     random.seed(num)
        np.random.seed(num)
    #     torch.manual_seed(num)
    #     if n_gpu > 0:
    #             torch.cuda.manual_seed_all(num)

    def cosine_similarity(self,v1, v2):
        mag1 = np.linalg.norm(v1)
        mag2 = np.linalg.norm(v2)
        if (not mag1) or (not mag2):
            return 0
        return np.dot(v1, v2) / (mag1 * mag2)

    def test_similarity(self,text1, text2):
        vecs = self.embed([text1, text2])
        v1 = vecs[0]
        v2 = vecs[1]
        return self.cosine_similarity(v1, v2)

    def sent_vec(self,text1):
        vec = self.embed(text1) ['outputs']
        return vec
    
    def calculate_thread_sim(self, data, threadheader): ##compared to the other messages in the thread
        thread_sim = []
        for a,b,c in zip(data.comment_text, data[threadheader], data.post_id): 
            df= data[data[threadheader] == b]
            df2 = df[df.post_id != c]
            threadtxt = " ".join(df2.comment_text)

            t = self.test_similarity(a, threadtxt)

            thread_sim.append(t)
        return thread_sim
    
    def get_rel_running_count(self,data): 
        all_rel_c = []
        for a,b in zip(data.RuncountPos, data.distance_score): 
            if a == 999: 
                rel_c = 0
            else: 
                rel_c = (a/b)*100
            all_rel_c.append(rel_c)
        return all_rel_c
    
    def prev_similarity (self, row): 
        t1 = row.comment_text
        t2 = row.prev_post_text
        if t2 != None: 
            return self.test_similarity(t1, t2)
        else: 
            return None
        
        
    def distance_score (self, data): 
        lst1 = list(data.thread_id) 
        lst2 = []

        c= 1 
        current_q = 0
        # print(current_q)

        for i in lst1: 
            if i != current_q: 
                c =1 
                current_q = i
                lst2.append(c)
            else: 
                c = c+1 
                lst2.append(c)
        return lst2        
        
    def get_prev_post_id (self, lst1, lst3): 
        #end posts
        lst2= []
        for num,i in enumerate(lst1): 
            if num == 0: 
                lst2.append(None)
    #             pass
            elif lst3[num] == lst3[num-1]: ##same question 
                lst2.append(lst1[num-1])
            else: ##different question
                lst2.append(None)
        return lst2
    
    def get_prev_post_text (self, row): 
        ix = row.prev_post_id
        if ix != None: 
            txt = self.d[ix]

            return txt
        else: 
            return None
        
    def create_dict_text(self,data):
        self.d= {}
        for a,b in zip(data.comment_text, data.post_id): 
            self.d[b] = a
            
    def create_dict_labels(self,data):
        self.d2= {}
        for a,b in zip(data.label, data.post_id): 
            self.d2[b] = a
        
            
    def get_label_prev (self,row):
        ix = row.prev_post_id
        if ix != None:
            ac = self.d2[ix]
            return ac
        else: 
            return None
        

    
    def main (self,datapath, outpath): 
        self.set_seed(1)
        
        data = pd.read_csv(datapath, sep= '\t')
        self.create_dict_text(data)
        self.create_dict_labels(data)
        
        ##get the ids of hte previous posts
        lst2 = self.get_prev_post_id(data.post_id, data.thread_id)
        
        data2 = pd.concat([data, pd.Series(lst2, name = "prev_post_id")], axis=1)
        
        ##get text and label of previous post

        data2['prev_post_label'] = data2.apply(lambda x: self.get_label_prev(x), axis=1)    
        
        data2['prev_post_text'] = data2.apply(lambda x: self.get_prev_post_text (x), axis=1)
        
        ##calculate similarity to previous post and thread 
        data2['prev_sim']= data2.apply(lambda x: self.prev_similarity(x), axis =1)
        
        data2 = data2.reset_index(drop= True)
        
        thread_sim = self.calculate_thread_sim(data2, threadheader = 'thread_id')
        
        ##calculate distance score 
        dist = self.distance_score(data)
        
        ##calculate label distribution features
                
        runcountneg, runcountpos, distlabel0, distlabel1 = self.get_running_count_binary(data2, 'thread_id')
        
        data3 = pd.concat([data2, pd.Series(dist, name= 'distance_score'), pd.Series(thread_sim, name = 'thread_sim'), pd.Series(runcountneg, name ='RuncountNeg'), pd.Series(runcountpos, name ='RuncountPos'), pd.Series(distlabel0, name ='DistanceLbl0'), pd.Series(runcountneg, name ='DistanceLbl1')], axis=1)
        
        rel_c = self.get_rel_running_count(data3)
        
        feat_df = pd.concat([data3, pd.Series(rel_c, name = 'RelRuncount')], axis=1)
        
        feat_df.to_csv(outpath, sep = '\t', index = False)
        return feat_df


# In[29]:


datapath = 'C:/Users/dirksonar/Documents/Data/Project8_Discourse/Misinfo/ExampleData.tsv'
outpath = 'C:/Users/dirksonar/Documents/Data/Project8_Discourse/Misinfo/ExampleDatawithFeat.tsv'
out= FeatureExtractor().main(datapath, outpath)


# In[ ]:




