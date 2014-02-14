'''
Created on 26 nov. 2013

@author: aitech
'''
import classifip
from binarytree import models,clustering,dichotomies
import numpy as np
import math

filename = ['autoMpg','autoPrice','bank8FM','bank32NH','housingB',
         'housingCal','aileronsDelta','elevators','friedman','house8L','house16H',
         'kinematics','mv','puma8NH','puma32H','stocks','ERA','ESL','LEV']



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
        

for f in filename :
    result=[]
    true=[]
    total=0.
    
    
    dataArff= classifip.dataset.arff.ArffFile()
    dataArff.load('classifip\\dataset\\test\\' + f + '_dis.arff')
    
    dataLogit = classifip.dataset.arff.ArffFile()
    dataLogit.load('classifip\\dataset\\test\\' + f + '_logit.arff')
    
    
    print "Tree Building..."
    
    dico = dichotomies.dichotomies(dataArff)
    tree = dico.build_ordinal(dataArff)
    
    
    for training, validation,index_tr,index_val in k_fold_cross_validation(dataArff, K=10,random_seed=31415926,randomise=True):
        list_score =[]
        list_disc_score=[] 
        list_disc_u65=[]
        
        list_score_tree=[]
        list_disc_score_tree=[]
        list_disc_u65_tree = []
        
        list_score_ncc=[]
        list_disc_score_ncc=[]
        list_disc_u65_ncc = []
        
        list_score_nbc = []
        
        l = classifip.models.ncc.NCC()
        model=classifip.models.nccof.NCCOF()
        
        print 'learning'
        #NBC
        l.learn(training)
        #NCCOF
        model.learn(training)
        #Nested dichotomies
        tree.learnAll(training)
    
        
        print 'evaluating'
        #NCCOF
        res=model.evaluate([x[0:len(x)-1] for x in validation.data],ncc_s_param=[2])
    #     result.append(res)
        #Nested dichotomies
        tree.evaluate([x[0:len(x)-1] for x in validation.data],ncc_s_param=[2])    
        res_tree= tree.decision_maximality()
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
    
        length = len(trlab)
        print length
        for run,k in enumerate(trlab):
            score =0.
            disc_score=0. 
            disc_u65=0.
            
            score_tree=0.
            disc_score_tree=0.
            disc_u65_tree = 0.
            
            score_ncc=0.
            disc_score_ncc=0.
            disc_u65_ncc = 0.
            
            score_logit = 0.
            score_nbc = 0.
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

                
            # For NCCOF
            if res[run][0].nc_maximal_decision()[ind]==1.:
                score=score+1.
            if res[run][0].nc_maximal_decision()[ind]==1.:
                sc = 1./(res[run][0].nc_maximal_decision()==1).sum()
                disc_score=disc_score + sc
                disc_u65 = disc_u65 - 1.2* sc*sc + 2.2 * sc

                
            # For dichotomy tree
            ind=tree.node.label.index(k)
            if res_tree[run][ind]==1.:
                score_tree=score_tree+1.
            if res_tree[run][ind]==1.:
                sc = 1./(res_tree[run].sum())
                disc_score_tree=disc_score_tree+ sc
                disc_u65_tree = disc_u65_tree - 1.2* sc*sc + 2.2 * sc
            
            total = total + 1.
                
            list_score.append(score/length*100)
            list_score_ncc.append(score_ncc/length*100)
            list_score_nbc.append(score_nbc/length*100)
            list_score_tree.append(score_tree/length*100)
            
            list_disc_score.append(disc_score/length*100)
            list_disc_score_ncc.append(disc_score_ncc/length*100)
            list_disc_score_tree.append(disc_score_tree/length*100)
            
            list_disc_u65.append(disc_u65/length*100)
            list_disc_u65_ncc.append(disc_u65_ncc/length*100)
            list_disc_u65_tree.append(disc_u65_tree/length*100)
            
    
    print '\\multicolumn{5}{|l|}{\\textit{' + f + '} (total evaluations: '+  str(int(total)) + ') } \\\\'     
    print '\\hline'
    print 'set accuracy & ' + str(round(np.std(list_score_nbc),2)) + ' & ' + str(round(np.std(list_score_ncc),2)) + ' & ' + str(round(np.std(list_score),2)) + ' & ' + str(round(np.std(list_score_tree),2)) + '\\\\'
    print 'disc acc & na & ' + str(round(np.std(list_disc_score_ncc),2)) + ' & ' + str(round(np.std(list_disc_score),2)) + ' & ' + str(round(np.std(list_disc_score_tree),2)) + '\\\\'
    print 'disc acc (u65) & na & ' +  str(round(np.std(list_disc_u65_ncc),2)) + ' & ' +  str(round(np.std(list_disc_u65),2)) + ' & ' + str(round(np.std(list_disc_u65_tree),2)) + '\\\\'
#     if imp != 0 : print 'mean imp & na & na & ' + str(round(mean_imp_ncc/imp_ncc,2)) + ' & ' + str(round(mean_imp/imp,2)) + ' & ' + str(round(mean_imp_tree/imp_tree,2))+ '\\\\'
    print "\\hline"