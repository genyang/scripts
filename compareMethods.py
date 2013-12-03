'''
Created on 26 nov. 2013

@author: aitech
'''
import classifip
from binarytree import models,clustering,dichotomies
import numpy as np
import math
filename = 'aileronsDelta'

dataArff= classifip.dataset.arff.ArffFile()
dataArff.load('classifip\\dataset\\test\\' + filename + '_dis.arff')

dataLogit = classifip.dataset.arff.ArffFile()
dataLogit.load('classifip\\dataset\\test\\' + filename + '_logit.arff')

def k_fold_cross_validation(data, K, randomise = False, random_seed=None, structured=None):
    """
    Generates K (training, validation) pairs from the items in X.

    Each pair is a partition of X, where validation is an iterable
    of length len(X)/K. So each training iterable is of length (K-1)*len(X)/K.

    If randomise is true, a copy of X is shuffled before partitioning,
    otherwise its order is preserved in training and validation.
    """
    if randomise:
        import random
        if random_seed != None:
            random.seed(random_seed)
        random.shuffle(data.data)
    datatr=data.make_clone()
    datatst=data.make_clone()

    for k in xrange(K):
        datatr.data = []
        index_tr = []
        datatst.data = []
        index_tst = []
        for i, x in enumerate(data.data) :
            if i % K != k :
                datatr.data.append(x)
                index_tr.append(i)
            else :
                datatst.data.append(x)
                index_tst.append(i)
        yield datatr, datatst, index_tr, index_tst
        

result=[]
true=[]
score=0.
imp=0.
total=0.
mean_imp=0.
disc_score=0. 
disc_u65=0.

score_tree=0.
imp_tree=0.
mean_imp_tree=0.
disc_score_tree=0.
disc_u65_tree = 0.

score_ncc=0.
imp_ncc=0.
mean_imp_ncc=0.
disc_score_ncc=0.
disc_u65_ncc = 0.

score_logit = 0.
score_nbc = 0.



print "Tree Building..."

dico = dichotomies.dichotomies(dataArff)
tree = dico.build_ordinal(dataArff)


for training, validation,index_tr,index_val in k_fold_cross_validation(dataArff, K=10,random_seed=31415926,randomise=True):

    l = classifip.models.ncc.NCC()
    model=classifip.models.nccof.NCCOF()
    logit = classifip.models.ordinalLogit.OrdinalLogit()
    
    print 'learning'
    #Logit
    X,y = logit.transform_data(dataLogit, rows=index_tr)  
    w, theta = logit.ordinal_logistic_fit(X, y)
    #NBC
    l.learn(training)
    #NCCOF
    model.learn(training)
    #Nested dichotomies
    tree.learnAll(training)

    
    print 'evaluating'
    #NCCOF
    res=model.evaluate([x[0:len(x)-1] for x in validation.data])
#     result.append(res)
    #Nested dichotomies
    tree.evaluate([x[0:len(x)-1] for x in validation.data],ncc_s_param=[2])    
    res_tree= tree.decision_maximality()
    #Logit
    X_val, y_val = logit.transform_data(dataLogit,rows=index_val)
    res_logit = logit.ordinal_logistic_predict(w, theta, X_val)
    #NBC
    l_resu0 =  l.evaluate(validation.data, ncc_s_param=[0.01])  
    res_nbc = []
    for i in range(0,len(l_resu0)): 
        res_nbc.append(l_resu0[i][0].nc_maximal_decision())
    
    #NCC
    l_resu =  l.evaluate(validation.data, ncc_s_param=[2])  
    res_ncc = []
    for i in range(0,len(l_resu)): 
        res_ncc.append(l_resu[i][0].nc_maximal_decision())
    
    trlab=[x[-1] for x in validation.data]
#     true.append(trlab)

    print len(trlab)
    for run,k in enumerate(trlab):
        # For NBC
        ind=training.attribute_data['class'].index(k)
        if res_nbc[run][ind]==1.:
            score_nbc=score_nbc+1.
        
        # For NCC
        if res_ncc[run][ind]==1.:
            sc_ncc = 1/res_ncc[run].sum()
            score_ncc=score_ncc+1.
            disc_score_ncc = disc_score_ncc + sc_ncc
            disc_u65_ncc = disc_u65_ncc - 1.2* sc_ncc*sc_ncc + 2.2 * sc_ncc
        if res_ncc[run].sum()>1.:
            imp_ncc=imp_ncc+1.
            mean_imp_ncc=mean_imp_ncc+res_ncc[run].sum()

        
        # For Logit
        if y_val[run] == res_logit[run]:
            score_logit = score_logit + 1.
            
        # For NCCOF
        if res[run][0].nc_maximal_decision()[ind]==1.:
            score=score+1.
        if res[run][0].nc_maximal_decision()[ind]==1.:
            sc = 1./(res[run][0].nc_maximal_decision()==1).sum()
            disc_score=disc_score + sc
            disc_u65 = disc_u65 - 1.2* sc*sc + 2.2 * sc
        if res[run][0].nc_maximal_decision().sum()>1.:
            imp=imp+1.
            mean_imp=mean_imp+res[run][0].nc_maximal_decision().sum()
            
        # For dichotomy tree
        ind=tree.node.label.index(k)
        if res_tree[run][ind]==1.:
            score_tree=score_tree+1.
        if res_tree[run][ind]==1.:
            sc = 1./(res_tree[run].sum())
            disc_score_tree=disc_score_tree+ sc
            disc_u65_tree = disc_u65_tree - 1.2* sc*sc + 2.2 * sc
        if res_tree[run].sum()>1.:
            imp_tree=imp_tree+1.
            mean_imp_tree=mean_imp_tree+res_tree[run].sum()
            
        total=total+1.

print '\\multicolumn{6}{|l|}{\\textit{' + filename + '} (total evaluations: '+  str(int(total)) + ') } \\\\'     
print '\\hline'
print 'set accuracy & ' + str(round(score_logit/total*100,2)) + '\% & ' + str(round(score_nbc/total*100,2)) + '\% & ' + str(round(score_ncc/total*100,2)) + '\% & ' + str(round(score/total*100,2)) + '\% & ' + str(round(score_tree/total*100,2)) + '\%\\\\'
print 'disc acc & na & na & ' + str(round(disc_score_ncc/total*100,2)) + '\% & ' + str(round(disc_score/total*100,2)) + '\% & ' + str(round(disc_score_tree/total*100,2)) + '\%\\\\'
print 'disc acc (u65) & na & na & ' + str(round(disc_u65_ncc/total*100,2)) + '\% & ' + str(round(disc_u65/total*100,2)) + '\% & ' + str(round(disc_u65_tree/total*100,2)) + '\%\\\\'
print 'imprecision & na & na & ' + str(int(imp_ncc)) + '( ' + str(round(imp_ncc/total*100,2))+ '\%) & ' + str(int(imp)) + '( ' + str(round(imp/total*100,2))+ '\%) & ' + str(int(imp_tree)) + '( ' + str(round(imp_tree/total*100,2)) + '\%)\\\\'
if imp != 0 : print 'mean imp & na & na & ' + str(round(mean_imp_ncc/imp_ncc,2)) + ' & ' + str(round(mean_imp/imp,2)) + ' & ' + str(round(mean_imp_tree/imp_tree,2))+ '\\\\'
print "\\hline"