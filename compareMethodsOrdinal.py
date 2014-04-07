'''
Created on 26 nov. 2013

@author: aitech
'''
import classifip
from binarytree import dichotomies
import numpy as np
import math
import os.path

# filename = ['autoMpg','autoPrice','bank8FM','bank32NH','boston_housing',
#          'california_housing','delta_ailerons','elevators','friedman','house8L','house16H',
#          'kinematics','mv','puma8NH','puma32H','stocks','ERA','ESL','LEV']

#Continuous data or closely, suitable for Logit model
filename = ['autoPrice','bank8FM','bank32NH','boston_housing','california_housing','cpu_small',
            'delta_ailerons','elevators','delta_elevators','friedman','house_8L','house_16H',
            'kinematics','puma8NH','puma32H','stock','ERA','ESL','LEV']



#data with numerous categorical attributes, non suitable for Logit model
# filename = ['nursery','car','autos','autoMpg','servo','mv']

# Data with upper/lower bounds issue with s = 0.001 or less
# filename = ['ailerons','abalone','cpu_act']


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
        

acc = open("acc.txt",'w+')
prec = open("prec.txt",'w+')


for f in filename :
    result=[]
    true=[]
    total=0.
    
    #NBC 
    score_nbc = 0.
    correct_nbc = 0
    
    #NCC
    score_ncc=0.
    imp_ncc=0.
    mean_imp_ncc=0.
    disc_score_ncc=0.
    disc_u65_ncc = 0.
    correct_ncc = 0
    
    #NCCOF
    score=0.
    imp=0.
    mean_imp=0.
    disc_score=0. 
    disc_u65=0.
    correct_nccof =0
    
    #NBC + ND
    score_nbcof=0.
    correct_nbcof = 0
    
    #NBC + ND
    score_tree_nbc=0.
    correct_tree_nbc = 0
    
    #NCC + ND
    score_tree_ncc=0.
    imp_tree_ncc=0.
    mean_imp_tree_ncc=0.
    disc_score_tree_ncc=0.
    disc_u65_tree_ncc = 0.
    correct_tree_ncc = 0
    
    score_logit = 0.
    

    dataArff= classifip.dataset.arff.ArffFile()
    if os.path.exists('..\\datasets\\' + f + '_dis.arff') :
        dataArff.load('..\\datasets\\' + f + '_dis.arff')
    else:
        dataArff.load('..\\datasets\\' + f + '.arff')
        
    dataLogit = classifip.dataset.arff.ArffFile()
    dataLogit.load('..\\datasets\\' + f + '_logit.arff')
        
    
    
    for training, validation,index_tr,index_val in k_fold_cross_validation(dataArff, K=10,random_seed=31415926,randomise=True):
    
        #print "Tree Building..."
    
        dico = dichotomies.dichotomies(training)
        tree = dico.build_ordinal(training)
        
        
        l = classifip.models.ncc.NCC()
        model=classifip.models.nccof.NCCOF()

        logit = classifip.models.ordinalLogit.OrdinalLogit()
        
        #print 'learning'
        #Logit
        X,y = logit.transform_data(dataLogit, index = index_tr)  
        w, theta = logit.ordinal_logistic_fit(X, y)
#         #NBC
#         l.learn(training)
        #NCCOF
        model.learn(training)
        #Nested dichotomies
        tree.learnAll(training)
    
        
        #print 'evaluating'
        
        #NCCOF
        l_resu=model.evaluate([x[0:len(x)-1] for x in validation.data],ncc_s_param=[2])
        res = []
        for i in range(0,len(l_resu)): 
            res.append(l_resu[i][0].nc_maximal_decision())
    
        #NBCOF
        l_resu=model.evaluate([x[0:len(x)-1] for x in validation.data],ncc_s_param=[0.001])
        res_nbcof = []
        for i in range(0,len(l_resu)): 
            res_nbcof.append(l_resu[i][0].nc_maximal_decision())
        
        #NCC + ND
        tree.evaluate([x[0:len(x)-1] for x in validation.data],ncc_s_param=[2])    
        res_tree_ncc = tree.decision_maximality()
        
        #NBC + ND
        tree.evaluate([x[0:len(x)-1] for x in validation.data],ncc_s_param=[0.001])    
        res_tree_nbc = tree.decision_maximality()
        
        #Logit
        X_val, y_val = logit.transform_data(dataLogit, index = index_val)
        res_logit = logit.ordinal_logistic_predict(w, theta, X_val)

#         #NBC
#         l_resu0 =  l.evaluate(validation.data, ncc_s_param=[0.001])  
#         res_nbc = []
#         for i in range(0,len(l_resu0)): 
#             res_nbc.append(l_resu0[i][0].nc_maximal_decision())
#         
#         #NCC
#         l_resu =  l.evaluate(validation.data, ncc_s_param=[2])  
#         res_ncc = []
#         for i in range(0,len(l_resu)): 
#             res_ncc.append(l_resu[i][0].nc_maximal_decision())
        
        trlab=[x[-1] for x in validation.data]
    #     true.append(trlab)
    
        #print len(trlab)
        for run,k in enumerate(trlab):
#             # For NBC
            ind=training.attribute_data['class'].index(k)
#             if res_nbc[run][ind]==1.:
#                 score_nbc=score_nbc+1.
#             
#             # For NCC
#             if res_ncc[run][ind]==1.:
#                 sc_ncc = 1/res_ncc[run].sum()
#                 score_ncc=score_ncc+1.
#                 disc_score_ncc = disc_score_ncc + sc_ncc
#                 disc_u65_ncc = disc_u65_ncc - 1.2* sc_ncc*sc_ncc + 2.2 * sc_ncc
#             if res_ncc[run].sum()>1.:
#                 imp_ncc=imp_ncc+1.
#                 if res_ncc[run][ind]==1.:
#                     correct_ncc += 1
#                 if res_nbc[run][ind]==1.:
#                     correct_nbc += 1
#             mean_imp_ncc+=res_ncc[run].sum()
    
            
            # For Logit
            if y_val[run] == res_logit[run]:
                score_logit = score_logit + 1.
                
            # For NBCOF
            if res_nbcof[run][ind]==1.:
                score_nbcof +=+1.
            
            # For NCCOF
            if res[run][ind]==1.:
                score=score+1.
            if res[run][ind]==1.:
                sc = 1./(res[run].sum())
                disc_score=disc_score + sc
                disc_u65 = disc_u65 - 1.2* sc*sc + 2.2 * sc
            if res[run].sum()>1.:
                imp=imp+1.
                if res[run][ind]==1.:
                    correct_nccof += 1
                if res_nbcof[run][ind]==1.:
                    correct_nbcof += 1
            mean_imp=mean_imp+res[run].sum()
            
            # For ND + NBC
            ind=tree.node.label.index(k)
            if res_tree_nbc[run][ind]==1.:
                score_tree_nbc=score_tree_nbc+1.
                
            # For ND + NCC
            ind=tree.node.label.index(k)
            if res_tree_ncc[run][ind]==1.:
                score_tree_ncc=score_tree_ncc+1.
            if res_tree_ncc[run][ind]==1.:
                sc = 1./(res_tree_ncc[run].sum())
                disc_score_tree_ncc=disc_score_tree_ncc+ sc
                disc_u65_tree_ncc = disc_u65_tree_ncc - 1.2* sc*sc + 2.2 * sc
            if res_tree_ncc[run].sum()>1.:
                imp_tree_ncc=imp_tree_ncc+1.
                if res_tree_ncc[run][ind]==1.:
                    correct_tree_ncc += 1
                if res_tree_nbc[run][ind]==1.:
                    correct_tree_nbc += 1 
            mean_imp_tree_ncc= mean_imp_tree_ncc + res_tree_ncc[run].sum()
                
            total=total+1.
    
#     print '\\multicolumn{8}{|l|}{\\textit{' + f + '} (total evaluations: '+  str(int(total)) + ') } \\\\'     
#     print '\\hline'
#     print 'set accuracy & ' + str(round(score_logit/total*100,2)) + '\% & ' + str(round(score_nbc/total*100,2)) + '\% & ' + str(round(score_ncc/total*100,2)) + '\% & ' + str(round(score_nbcof/total*100,2)) + '\% & ' + str(round(score/total*100,2)) + '\% & ' + str(round(score_tree_nbc/total*100,2)) + '\% & ' + str(round(score_tree_ncc/total*100,2)) + '\%\\\\'
#     print 'disc acc & ' + 'idem & idem & ' + str(round(disc_score_ncc/total*100,2)) + '\% & ' + 'idem & ' + str(round(disc_score/total*100,2)) + '\% & ' + 'idem & ' + str(round(disc_score_tree_ncc/total*100,2)) + '\%\\\\'
#     print 'disc acc (u65) & ' + 'idem & idem & '  + str(round(disc_u65_ncc/total*100,2)) + '\% & ' + 'idem & ' + str(round(disc_u65/total*100,2)) + '\% & ' + 'idem & ' + str(round(disc_u65_tree_ncc/total*100,2)) + '\%\\\\'
#     print 'imprecision & ' + '0 & 0 & ' + str(round(imp_ncc/total*100,2))+ '\% & ' + '0 & ' + str(round(imp/total*100,2))+ '\% & ' + '0 & ' + str(round(imp_tree_ncc/total*100,2)) + '\% \\\\'
#     print 'acc. when imp & na & ' + str(round(correct_nbc/imp_ncc*100,2))+ '\% & ' + str(round(correct_ncc/imp_ncc*100,2))+ '\% & ' + str(round(correct_nbcof/imp*100,2))+ '\% & ' + str(round(correct_nccof/imp*100,2))+ '\% & ' + str(round(correct_tree_nbc/imp_tree_ncc*100,2))+ '\% & ' + str(round(correct_tree_ncc/imp_tree_ncc*100,2))+ '\%\\\\ ' 
#     print 'mean imp & na & na & ' + str(round(mean_imp_ncc/total,2)) + ' & ' + 'na & ' + str(round(mean_imp/total,2)) + ' & ' + 'na & ' + str(round(mean_imp_tree_ncc/total,2))+ '\\\\'
#     print "\\hline"
    
    print '\\multicolumn{7}{|l|}{\\textit{' + f + '} (total evaluations: '+  str(int(total)) + ') } \\\\'     
#     print '\\hline'
#     print 'set accuracy & ' + str(round(score_nbc/total*100,2)) + '\% & ' + str(round(score_ncc/total*100,2)) + '\% & ' + str(round(score_nbcof/total*100,2)) + '\% & ' + str(round(score/total*100,2)) + '\% & ' + str(round(score_tree_nbc/total*100,2)) + '\% & ' + str(round(score_tree_ncc/total*100,2)) + '\%\\\\'
#     print 'disc acc & ' + 'idem & ' + str(round(disc_score_ncc/total*100,2)) + '\% & ' + 'idem & ' + str(round(disc_score/total*100,2)) + '\% & ' + 'idem & ' + str(round(disc_score_tree_ncc/total*100,2)) + '\%\\\\'
#     print 'disc acc (u65) & ' + 'idem & '  + str(round(disc_u65_ncc/total*100,2)) + '\% & ' + 'idem & ' + str(round(disc_u65/total*100,2)) + '\% & ' + 'idem & ' + str(round(disc_u65_tree_ncc/total*100,2)) + '\%\\\\'
#     print 'imprecision & ' + '0 & ' + str(round(imp_ncc/total*100,2))+ '\% & ' + '0 & ' + str(round(imp/total*100,2))+ '\% & ' + '0 & ' + str(round(imp_tree_ncc/total*100,2)) + '\% \\\\'
#     print 'acc. when imp & ' + str(round(correct_nbc/imp_ncc*100,2))+ '\% & ' + str(round(correct_ncc/imp_ncc*100,2))+ '\% & ' + str(round(correct_nbcof/imp*100,2))+ '\% & ' + str(round(correct_nccof/imp*100,2))+ '\% & ' + str(round(correct_tree_nbc/imp_tree_ncc*100,2))+ '\% & ' + str(round(correct_tree_ncc/imp_tree_ncc*100,2))+ '\%\\\\ ' 
#     print 'mean imp & na & ' + str(round(mean_imp_ncc/total,2)) + ' & ' + 'na & ' + str(round(mean_imp/total,2)) + ' & ' + 'na & ' + str(round(mean_imp_tree_ncc/total,2))+ '\\\\'
#     print "\\hline"

    scores = [score_logit, score_nbcof, disc_u65, score_tree_nbc, disc_u65_tree_ncc]
    rang = np.argsort(scores) + 1
    
    acc.write(f + ' & ' + str(round(score_logit/total*100,2)) + '\% (' + str(rang[0]) + ') & ' + str(round(score_nbcof/total*100,2)) + '\% (' + str(rang[1]) + ') & ' + str(round(disc_u65/total*100,2)) + '\% (' + str(rang[2]) + ') & ' + str(round(score_tree_nbc/total*100,2)) + '\% (' + str(rang[3]) + ') & '  + str(round(disc_u65_tree_ncc/total*100,2)) + '\% (' + str(rang[4]) + ') \\\\ \n' )
    prec.write(f + str(round(correct_nbcof/imp*100,2))+ ' ' + str(round(correct_nccof/imp*100,2))+ ' ' + str(round(correct_tree_nbc/imp_tree_ncc*100,2))+ ' ' + str(round(correct_tree_ncc/imp_tree_ncc*100,2))+ '\n ')
    

acc.close()
prec.close()