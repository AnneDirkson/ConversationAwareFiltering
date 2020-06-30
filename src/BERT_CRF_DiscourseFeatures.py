#!/usr/bin/env python
# coding: utf-8

# In[6]:


from sklearn.model_selection import KFold
import pandas as pd 
import numpy as np
import pickle
from collections import defaultdict
import ktrain 
from ktrain import text
import torch 
import tensorflow as tf
import nltk
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer, f1_score, recall_score, precision_score
from sklearn.model_selection import cross_val_score, RandomizedSearchCV

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
##set seed
import random 

import os


# In[71]:


class ConvAwareModel(): 
    
    def __init__ (self): 
        self.set_seed(1,1)
        
    def set_seed(self,num, n_gpu):
        random.seed(num)
        np.random.seed(num)
        tf.random.set_seed(num)
        try: 
            torch.manual_seed(num)
        except NameError: 
            pass
        if n_gpu > 0:
                torch.cuda.manual_seed_all(num)
                
                
    def make_dir(self, outpath):
        path = outpath + '/Folds'
        path2 = outpath + '/BERTprobs'
        path3 = outpath + '/FeatureSelect'
        path4 = outpath + '/BERTpredictors'

        for i in [path, path2, path3, path4]:
            try:
                os.mkdir(i)
            except OSError:
                print ("Creation of the directory %s failed" % i)
            else:
                print ("Successfully created the directory %s " % i)
        
            
        

    def save_obj(self, obj, name):
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
            
    def create_one_fold(self, d, dev, test, outpath, suffix):  
        
        dt = defaultdict(list)
        train= []
        dev2 = []
        test2 = []
        
        for num, i in enumerate(self.all_sets): 
#             print(len(i))
            if num != dev and num!= test: 
                for conv in i: 
#                     print(conv)
                    [train.append(j) for j in d[conv]]
                    

        for conv in self.all_sets[dev]: 
            [dev2.append(j) for j in d[conv]]

        for conv in self.all_sets[test]: 
            [test2.append(j) for j in d[conv]]
        
        
        dt['train'] = train
        dt['dev'] = dev2
        dt['test'] = test2

#         print(dt['test'])

        outpath2 = outpath + 'Folds/' + suffix
        self.save_obj(dt, outpath2)
        return dt

    def create_fold_dict(self, data, outpath): 
        ##get all thread ids
        lst = list(set(list(data.thread_id)))
        
        self.all_sets = []
        kf = KFold(n_splits=10, random_state = 10)
        kf.get_n_splits(lst)
        for train_index, test_index in kf.split(lst):
            s = []
            for i in test_index: 
                s.append(lst[i])
            self.all_sets.append(s)
            
        print(len(self.all_sets))

        d = {}
        for i in lst: 
            df = data[data.thread_id == i]
#             print(len(df))
            d[i] = list(set(df.post_id))
            
#         print(d)
        
        self.folddicts= []
        
        for test, dev, suffix in zip([0,1,2,3,4,5,6,7,8,9], [1,2,3,4,5,6,7,8,9,0], ['fold1', 'fold2','fold3', 'fold4', 'fold5', 'fold6', 'fold7','fold8', 'fold9', 'fold10']): 
            x = self.create_one_fold(d, dev, test, outpath, suffix)
            self.folddicts.append(x)
            
    
    def reformat_for_bert (self, data,label, folddict): ##need commenttxt and label 
        x = folddict['dev']
        y = folddict['train']
        z = folddict['test']

        data[label] = pd.Categorical(data[label])
        data[label] = data[label].cat.codes

        dev_df = data[data.post_id.isin(x)]
        train_df = data[data.post_id.isin(y)]
        test_df = data[data.post_id.isin(z)]

        y_dev = dev_df[label]
        y_train = train_df[label]
        y_test = test_df[label]

        X_dev = dev_df.comment_text
        X_train = train_df.comment_text
        X_test = test_df.comment_text

        print('The length of the test data is: ' + str(len(X_test)))
        print('The lenght of the train data is: ' + str(len(X_train)))
        print('The lenght of the dev data is: ' + str(len(X_dev)))

        return X_dev, y_dev, X_train, y_train, X_test, y_test
    
    

    def run_BERT_model(self, data, fold, sufix, label, t, base_out_path, epochs =3): 

        X_dev, y_dev, X_train, y_train, X_test, y_test = self.reformat_for_bert(data, label, fold)
        X_train2 = [str(i) for i in X_train]
        X_test2 = [str(i) for i in X_test]
        X_dev2 = [str(i) for i in X_dev]

        y_test2 = [str(i) for i in y_test]
        y_train2 = [str(i) for i in y_train]
        y_dev2 = [str(i) for i in y_dev]

        trn = t.preprocess_train (X_train2, y_train2)
        val = t.preprocess_test(X_dev2, y_dev2)
        test = t.preprocess_test(X_test2, y_test2)

        model = t.get_classifier()
        learner = ktrain.get_learner(model, train_data = trn, val_data =val, batch_size = 32)
        hist1 = learner.fit_onecycle(5e-5, epochs)

        predictor1 = ktrain.get_predictor(learner.model, preproc=t)

        hist2 = learner.fit_onecycle(5e-5, 1)

        predictor2 = ktrain.get_predictor(learner.model, preproc=t)


        v1 = hist1.history['val_accuracy'] 
        v2 = hist2.history['val_accuracy']

        if v1 >= v2: 
            chosen = 3
            self.save_obj(chosen, (base_out_path + '/BERTpredictors/chosen_epochs_' + sufix))
            predictor1.save(base_out_path + '/BERTpredictors/predictor1_' + sufix )

#             res_test = predictor1.predict(X_test2)
            res = predictor1.predict(X_test2, return_proba = True)
            res_train = predictor1.predict(X_train2, return_proba = True)
            res_dev = predictor1.predict(X_dev2, return_proba = True)
        else: 
            chosen = 4
            self.save_obj(chosen, (base_out_path + '/BERTpredictors/chosen_epochs_' + sufix))
            predictor2.save(base_out_path + '/BERTpredictors/predictor2_' + sufix )
            
#             res_test = predictor2.predict(X_test2)
            res = predictor2.predict(X_test2, return_proba = True)
            res_train = predictor2.predict(X_train2, return_proba = True)
            res_dev = predictor2.predict(X_dev2, return_proba = True)
         
        
#         f1_out = f1_score (y_true = y_test2, y_pred = res_test)
#         recall_out = recall_score(y_true =  y_test2, y_pred = res_test)
#         prec_out = precision_score (y_true =  y_test2, y_pred = res_test)
 
        outpath = base_out_path + 'BERTprobs/probs_' + sufix + '.npy'
        np.save(outpath, res, allow_pickle=False)
        outpath2 = base_out_path + 'BERTprobs/probs_train_' + sufix + '.npy'
        np.save(outpath2, res_train, allow_pickle=False)
        outpath3 = base_out_path + 'BERTprobs/probs_dev_' + sufix + '.npy'
        np.save(outpath3, res_dev, allow_pickle=False)
        
#         return f1_out, recall_out, prec_out
    
    def run_BERT(self, data, outpath): 
        maxlen=128
        MODELNAME='distilbert-base-uncased'
        t = text.Transformer(MODELNAME, maxlen=maxlen, classes=[0,1])
        
        out = defaultdict(dict)
        
        print(len(self.folddicts[0]['test']))
        
        for a,b in zip(self.folddicts, ['fold1', 'fold2','fold3', 'fold4', 'fold5', 'fold6', 'fold7','fold8', 'fold9', 'fold10']):
            self.run_BERT_model (data, a, b, 'label', t, outpath)
#             out['F1'].append(f1_out)
#             out['recall'].append(recall_out)
#             out['precision'].append(prec_out)
  
        print('BERT is done')
        
        ##print average outcome 
#         print('The mean F1 score is: ')
#         print(np.mean(out['F1']))

    def calculate_BERT_metrics(self, data, outpath): 
        out = defaultdict(list)
        
        for a,b in zip(self.folddicts, ['fold1', 'fold2','fold3', 'fold4', 'fold5', 'fold6', 'fold7','fold8', 'fold9', 'fold10']):
            y_true = self.collect_y_true(a, data)
            ##collect probs
            path = outpath + '/BERTprobs/probs_' + b + '.npy'
            p = np.load (path)
            
            y_pred = self.reformat_probas(p)
            
#             print(len(y_true))
#             print(len(y_pred))
            
            f = f1_score(y_pred = y_pred, y_true = y_true)
            p = precision_score(y_pred = y_pred, y_true = y_true)
            r = recall_score(y_pred = y_pred, y_true=y_true)

            out['F1'].append(f)
            out['Recall'].append(r)
            out['Precision'].append(p)
            
        ##print
        print('The F1 score for BERT is: ')
        avg = np.mean(out['F1'])
        print(avg)
        
        ##save 
        self.save_obj(out, outpath + 'BERTresults')
        
    
    def collect_y_true(self, folddict, data): 
        z = folddict['test']
#         print(len(z))
        test_df = data[data.post_id.isin(z)]

        y_test = test_df.label

        return y_test
    
    def reformat_probas(self, probas): 
        x = probas[0]
        df = pd.DataFrame(x)
        y_pred= []
        for a,b in zip(df[0], df[1]): 
            if a > b: 
                y_pred.append(0)
            else: 
                y_pred.append(1)

        return y_pred

        
    def obtain_probas_nw (self, data, folddict, foldname, basepath, rightid = 'post_id'):
        ###load the right probas
        probas_dev = np.load(basepath + 'probs_dev_' + foldname + '.npy')
        probas_test = np.load(basepath + 'probs_' + foldname + '.npy')
        probas_train= np.load(basepath + 'probs_train_' + foldname + '.npy')

        ##reformat
        x = folddict['dev']
        y = folddict['train']
    #     nw_x = list(set(x).union(set(y)))
        z = folddict['test']
#         print(len(z))
        ##first get post ids

        dev_df0 =  data[data[rightid].isin(x)]
        train_df0 = data[data[rightid].isin(y)]
        test_df0 = data[data[rightid].isin(z)]

        train_df_nw = pd.concat([train_df0, dev_df0],axis=0)

        dev_posts =list(dev_df0.post_id)
        train_posts= list(train_df0.post_id)
        test_posts = list(test_df0.post_id)

#         print(len(dev_posts))
#         print(len(probas_dev))

        trainlbls = list(train_df_nw.label)
        testlbls = list(test_df0.label)

        df_test = pd.DataFrame(probas_test[0])
        df_train = pd.DataFrame(probas_train[0])
        df_dev = pd.DataFrame(probas_dev[0])

        df_train_nw= pd.concat([df_train, df_dev], axis =0)

        df_test.columns = ['Predictions0', 'Predictions1']
        df_train_nw.columns = ['Predictions0', 'Predictions1']


        train_q = list(train_df0.thread_id)
        test_q = list(test_df0.thread_id)

        df_train_nw = df_train_nw.reset_index(drop= True)


        return df_train_nw, df_test, trainlbls, testlbls, train_q, test_q
        
    
    def get_thread_startpoints (self, lst1): ##list is a list of the question ids 
        lst2= []
        current_q = 0
        for num,i in enumerate(lst1): 
            if num == 0: 
                lst2.append(0)
                current_q = i
    #             pass
            elif current_q == i: ##same question 
                lst2.append(0)
            else: ##different question
                lst2.append(1)
                current_q = i
        return lst2

    def reformat_for_crf(self, train_df, startp): 
        #first turn it into one long list of dictionaries 
        nwcolnames= [str(i) for i in train_df.columns]
        train_df.columns = nwcolnames
    #     print(train_df.head())

        d= train_df.to_dict('records')

        out = []
        temp= []
        for a,b in zip(d, startp): 
            if b == 1: 
                out.append(temp)
                temp = []
                temp.append(a)
            else: 
                temp.append(a)
        out.append(temp)
        return out

    def transform_sent_vecs_into_df(self, sent_vecs): 
        df = pd.DataFrame(sent_vecs[0])
    #     df2 = df.transpose()
        for i in sent_vecs[1:]: 
            y = pd.Series(i)
    #         y2 = y.transpose()
            df = pd.concat([df, y], axis=1)
        df3 = df.transpose()
        return df3

    def reformat_labels(self, data, startp): 
        out = []
        temp= []
        for a,b in zip(data.label, startp): 
            if b == 1: 
                out.append(temp)
                temp = []
                temp.append(str(a))
            else: 
                temp.append(str(a))
        out.append(temp)
        return out

    def reformat_labels_blend(self, lbls, startp): 
        out = []
        temp= []
        for a,b in zip(lbls, startp): 
            if b == 1: 
                out.append(temp)
                temp = []
                temp.append(str(a))
            else: 
                temp.append(str(a))
        out.append(temp)
        return out
    
    def obtain_other_features(self, data, folddict,rightid = 'post_id'):
    
         ##reformat
        x = folddict['dev']
        y = folddict['train']
        z = folddict['test']

        ##first get post ids

        dev_df0 =  data[data[rightid].isin(x)]
        train_df0 = data[data[rightid].isin(y)]
        test_df0 = data[data[rightid].isin(z)]

        train_df = pd.concat([train_df0, dev_df0], axis =0)
        train_df = train_df.fillna(999)
        test_df = test_df0.fillna(999)

        train_df = train_df.reset_index(drop = True)
        test_df = test_df.reset_index(drop = True)

        return train_df, test_df
    
    def run_BlendedBERTCRF(self, data, basepath_probas): 
        out = defaultdict(list)
        params_space = {
                'c1': scipy.stats.expon(scale=0.5),
                'c2': scipy.stats.expon(scale=0.05),
            }

        f1_scorer = make_scorer(metrics.flat_f1_score,
                            average='weighted',  labels=[0,1])

        for d, w in zip(self.folddicts, ['fold1', 'fold2', 'fold3', 'fold4', 'fold5', 'fold6', 'fold7', 'fold8', 'fold9', 'fold10']):

            crf = sklearn_crfsuite.CRF(
                algorithm='lbfgs',
                max_iterations=100,
                all_possible_transitions=True
            )



            rs = RandomizedSearchCV(crf, params_space,
                                    cv=9,
                                    verbose=1,
                                    n_jobs=-1,
                                    n_iter=50,
                                    scoring=f1_scorer)

#             print(d.keys())
            train_df, test_df, trainlbls, testlbls, train_q, test_q = self.obtain_probas_nw(data, d, w, basepath_probas)
    #         print(train_df.head())

            X_train = train_df
            y_train = trainlbls

            X_test = test_df
            y_test = testlbls

            #get thread start points
    #         lst = train_df.question_post_id
            startp = self.get_thread_startpoints(train_q)
            y_train = self.reformat_labels_blend(trainlbls, startp)

            #reformat X_train
            X_train_nw = self.reformat_for_crf(X_train, startp)

            startp = self.get_thread_startpoints(test_q)
            y_test = self.reformat_labels_blend(testlbls, startp)

            X_test_nw = self.reformat_for_crf(X_test, startp)
    
            rs.fit(X_train_nw, y_train)

#             print("\nBest Score = " + str(rs.best_score_) + ' for the parameters ' + str(rs.best_params_))
    #     
            y_pred = rs.predict (X_test_nw)

            y_test_exp = [int(i) for j in y_test for i in j]
            y_pred_exp = [int(i) for j in y_pred for i in j]
       

            f1_out = f1_score (y_true = y_test_exp, y_pred = y_pred_exp)
            recall_out = recall_score(y_true = y_test_exp, y_pred = y_pred_exp)
            prec_out = precision_score (y_true = y_test_exp, y_pred = y_pred_exp)

            out['F1'].append(f1_out)
            out['recall'].append(recall_out)
            out['precision'].append(prec_out)
 
            out['best_param'].append(rs.best_params_)
  
        print('The mean F1 score for BERT + CRF is: ')
        avg = np.mean(out['F1']) 
        self.currentf1 = avg
        print(str(avg))  
        
        self.crf_params = out['best_param']
        
        return out
       
    
    def run_crf_preset_blended_doublerun(self, data, prevaddlst, addlst, predlbls, crf_params, basepath_probas):  
        out = defaultdict(list)

        f1_scorer = make_scorer(metrics.flat_fbeta_score, beta=1,
                            average='weighted', labels=[0,1])

        for p, d, l in zip(crf_params, self.folddicts, ['fold1', 'fold2', 'fold3', 'fold4', 'fold5', 'fold6', 'fold7', 'fold8', 'fold9', 'fold10']):

            p1 = p['c1']
            p2 = p['c2']

            crf = sklearn_crfsuite.CRF(
                algorithm='lbfgs',
                c1 =p1, 
                c2 = p2,
                max_iterations=100,
                all_possible_transitions=True
            )


            train_df, test_df, trainlbls, testlbls, train_q, test_q = self.obtain_probas_nw(data, d, l, basepath_probas)

            xtra_train_df, xtra_test_df = self.obtain_other_features(data, d)  

            labeldistlst = ['prev_post_label', 'RuncountNeg', 'RuncountPos', 'DistanceLbl0', 'DistanceLbl1', 'RelRuncount']
            prevaddlst2 = [i for i in prevaddlst if i not in labeldistlst]

            if addlst in labeldistlst: 
                addlst2 = []
            else: 
                addlst2 = addlst

            combolst = []
            [combolst.append(i) for i in prevaddlst]
            combolst.append(addlst)


            set1= set(combolst)
            set2 = set(labeldistlst)

            droplst = [i for i in labeldistlst if i not in combolst]

            X_train1 = train_df
            for i in prevaddlst2: 
                X_train1 = pd.concat([X_train1, xtra_train_df[i]], axis=1)

            if addlst2  != []: 
                X_train1 = pd.concat([X_train1, xtra_train_df[addlst]], axis=1)

            ##now add all the label dist 
            for i in labeldistlst: 
                X_train1 = pd.concat([X_train1, xtra_train_df[i]], axis=1)

            ##now drop the unnecessary ones
            X_train1 = X_train1.drop (labels = droplst, axis=1)


            startp = self.get_thread_startpoints(train_q)
            y_train = self.reformat_labels_blend(trainlbls, startp)

            #reformat X_train
            X_train_nw = self.reformat_for_crf(X_train1, startp)    

            crf.fit(X_train_nw, y_train)

            ##testing phase
            X_test2 = test_df

            for i in prevaddlst2: 
                X_test2 = pd.concat([X_test2, xtra_test_df[i]], axis=1)
    #             print(X_test2.columns)

            if addlst2  != []: 
                X_test2 = pd.concat([X_test2, xtra_test_df[addlst]], axis=1)


            startp = self.get_thread_startpoints(test_q)
            y_test = self.reformat_labels_blend(testlbls, startp)


            ##first run 

            X_test_nw = self.reformat_for_crf(X_test2, startp)


            ## make dummy versions of all necessary

            if len(set1.intersection(set2)) > 0 : 
                X_test_nwer = []
                for t in X_test_nw: 
                    temp = [] ##temporary thread
                    for num, p in enumerate(t): ##p is a dictionary for the post
                        nwp = p
                        if 'prev_post_label' in combolst: 
                            nwp['prev_post_label'] = 0 
                        if 'RuncountNeg' in combolst:
                            nwp['RuncountNeg'] = num
                        if 'RuncountPos' in combolst:
                            nwp['RuncountPos'] = 0

                        if 'DistanceLbl0' in combolst:
                            nwp['DistanceLbl0'] = 0
                        if 'DistLbl1' in combolst:
                            nwp['DistLbl1'] = 999 ##same as infinite - there have been none
                        if 'RelRuncount' in combolst:
                            nwp['RelRuncount'] = 0

                        temp.append(nwp)
                    X_test_nwer.append(temp)
            else: 
                X_test_nwer = X_test_nw

            ##first prediction round (or last if no label dist values.)
            y_pred_first = crf.predict(X_test_nwer)

           ##second prediction if necessary 

            if len(set1.intersection(set2)) > 0 : 
                X_test_nwest = [] ## for all threads
                for a,b in zip(X_test_nwer, y_pred_first): ##these are threads
                    ##initialize

                    temp = [] ##for one thread

                    ds = 0
                    c0 = 0
                    c1 = 0
                    d0 = 999
                    d1 = 999
                    rel_c = 0
                    for p, l in zip (a,b): #these are the individual posts - l is predlbl
                        nwp = p
                        ##update dictionary
                        if 'prev_post_label' in combolst: 
                            nwp['prev_post_label'] = l
                        if 'RuncountNeg' in combolst:
                            nwp['RuncountNeg'] = c0
                        if 'RuncountPos' in combolst:
                            nwp['RuncountPos'] = c1

                        if 'DistanceLbl0' in combolst:
                            nwp['DistanceLbl0'] = d0
                        if 'DistLbl1' in combolst:
                            nwp['DistLbl1'] = d1 ##same as infinite - there have been none
                        if 'RelRuncount' in combolst:
                            nwp['RelRuncount'] = rel_c

                        temp.append(nwp)

                        ##update values for next one
                        prevlbl = l
                        if prevlbl == 0: 
                            c0 += 1
                            d0 = 0
                            if d1 == 999: 
                                d1 = 999
                            else: 
                                d1 +=1
                        elif prevlbl ==1: 
                            c1 += 1
                            d1 = 0
                            if d0 == 999: 
                                d0 = 999
                            else: 
                                d0 += 1
                        ds = ds +1

                        if c1 == 0: 
                            rel_c= 0
                        else:
                            rel_c = (c1/ds)
                    ##thread is done - add to new Xtest
                    X_test_nwest.append(temp)



                y_pred_nw = crf.predict(X_test_nwest)
                y_pred_exp = [int(i) for j in y_pred_nw for i in j]

            else: 
                y_pred_exp = [int(i) for j in y_pred_first for i in j]


            y_test_exp= [int(i) for j in y_test for i in j]

            f1_out = f1_score (y_true = y_test_exp, y_pred = y_pred_exp)         
            recall_out = recall_score(y_true = y_test_exp, y_pred = y_pred_exp)
            prec_out = precision_score (y_true = y_test_exp, y_pred = y_pred_exp)
    #         auc_out = roc_auc_score(y_true = y_test, y_pred= y_pred)
            out['F1'].append(f1_out)
            out['recall'].append(recall_out)
            out['precision'].append(prec_out)
        
        avg = np.mean(out['F1'])
        
        return out, avg
    
    def forward_feature_selection(self, data,  basepath_probas, outpath, currentf1, da_acts = False): 
        
        if da_acts == True: 
            origfeaturelst = ['prev_post_label', 'distance_score', 'prev_sim', 'RuncountNeg', 'RuncountPos', 'RelRuncount', 'DistanceLbl0', 'DistanceLbl1', 'thread_sim', 'da_acts', 'da_acts_prev']
        else: 
             origfeaturelst = ['prev_post_label', 'distance_score', 'prev_sim', 'RuncountNeg', 'RuncountPos', 'RelRuncount', 'DistanceLbl0', 'DistanceLbl1', 'thread_sim']

        featurelst = origfeaturelst
        prevaddlst = []
        current_F1 = currentf1
        improv_f1 = []
        improv_f1.append(currentf1)
        quit = 0
        round_num =1

        while quit == 0 : 
            round_dict = {}
            res = []
            for j in featurelst: 
                ##run CRF experiments
#                 print(j)
                if j == 'prev_post_label' or 'prev_post_label' in prevaddlst:
                    p = True
                else: 
                    p = False

                out, avg = self.run_crf_preset_blended_doublerun(data,prevaddlst, addlst=j, predlbls = p, crf_params=self.crf_params, basepath_probas = basepath_probas)            
                round_dict[j] = out

                res.append(avg)
            ##save rounddict
            savepath = outpath + 'round_dict' + str(round_num)
            self.save_obj(round_dict, savepath)

            ##get the best
            bestix = res.index(np.max(res))
            bestval = np.max(res)
            if bestval > current_F1: 
                pass
            else: 
#                 print('Time to quit')
                quit = 1 ##quit if the best feature adds nothing.

#             print('And the best feature is ... ')
            bestfeat = featurelst[bestix]   
#             print(bestfeat)

            round_num = round_num + 1

            ##update feature list
            prevaddlst.append(bestfeat)
            featurelst = [i for i in featurelst if i != bestfeat] ##remove the feature from the list
            current_F1 = bestval
            improv_f1.append(bestval)
            self.save_obj(prevaddlst, outpath + 'features_added')
            self.save_obj(current_F1, outpath + 'best_f1')
            self.save_obj(improv_f1, outpath + 'steps_f1')
        
            ##return some outcome to print
        print('The new best F1 score is: ')
        print(current_F1)
        
        print('The added features to attain this were: ')
        print(prevaddlst)
        
#         return prevaddlst, current_F1
        
    def main(self, data, outpath, da_acts = False): 
        self.make_dir(outpath)
        self.create_fold_dict(data, outpath)
        self.run_BERT(data, outpath)
        self.calculate_BERT_metrics(data, outpath)
    
        basepath_probas = outpath + 'BERTprobs/'
        
        outblend = self.run_BlendedBERTCRF( data, basepath_probas)

        self.save_obj(outblend, outpath + 'BlendedBERTresults')
        
        nwoutpath = outpath + '/FeatureSelect/'
        
        self.forward_feature_selection(data, basepath_probas, nwoutpath, currentf1 = self.currentf1, da_acts = False)
        
#         self.save_obj(outfeat, outpath + 'FeatureSelectionresults')
        
#         return outblend, outfeat
        


# In[72]:


##load data 
data = pd.read_csv('/data/dirksonar/Temp/ExampleDatawithFeat.tsv', sep = '\t')
# print(data.head())

outpath = '/data/dirksonar/Temp/'

ConvAwareModel().main(data, outpath)


# In[ ]:




