'''
Created on 26 nov. 2013

@author: aitech
'''
import classifip
from binarytree import models,clustering,dichotomies
import os.path
import math
import time

# filename = ['autoMpg','autoPrice','bank8FM','bank32NH','housingB',
#          'housingCal','aileronsDelta','elevators','friedman','house8L','house16H',
#          'kinematics','mv','puma8NH','puma32H','stocks','ERA','ESL','LEV']

# filename = ['balance-scale','car','lymph','LEV','nursery','zoo','soybean','grub-damage','page-blocks','glass'
filename = ['glass']



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
    
    score_tree=0.
    correct_tree_nbc = 0
    
    score_tree_ncc=0.
    imp_tree_ncc=0.
    mean_imp_tree_ncc=0.
    disc_score_tree_ncc=0.
    disc_u65_tree_ncc = 0.
    correct_tree_ncc = 0
    
    score_ncc=0.
    imp_ncc=0.
    mean_imp_ncc=0.
    disc_score_ncc=0.
    disc_u65_ncc = 0.
    correct_ncc = 0
    
    score_nbc = 0.
    correct_nbc = 0
    
    dataArff= classifip.dataset.arff.ArffFile()
    if os.path.exists('..\\datasets\\' + f + '_dis.arff') :
        dataArff.load('..\\datasets\\' + f + '_dis.arff')
    else:
        dataArff.load('..\\datasets\\' + f + '.arff')

    
    c = 0

    
    for training, validation,index_tr,index_val in k_fold_cross_validation(dataArff, K=10,random_seed=3141592,randomise=True):
        
        print "Tree Building..."
    
        # Auto-linkage
        dico = dichotomies.dichotomies(training)
        tree = dico.build_hierarchical(training)
    
#         l = classifip.models.ncc.NCC()

        
        #print 'learning'

        #NCC + NBC
#         l.learn(training)
#         #Nested dichotomies
#         tree.learnAll(training)
#     
#         
#         #print 'evaluating'
# 
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
# 
#         #ND + NBC
#         tree.evaluate([x[0:len(x)-1] for x in validation.data],ncc_s_param=[0.001])    
#         res_tree_nbc= tree.decision_maximality()        
        
        #ND + NCC
        tree.evaluate([x[0:len(x)-1] for x in validation.data],ncc_s_param=[2])
   
        a = time.time()
        res_tree_ncc= tree.decision_maximality()
        b = time.time()
        
        c = c + b-a
        
        trlab=[x[-1] for x in validation.data]
    #     true.append(trlab)
    
        #print len(trlab)
        for run,k in enumerate(trlab):
            # For NBC
#             ind=training.attribute_data['class'].index(k)
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
#                 mean_imp_ncc=mean_imp_ncc+res_ncc[run].sum()
#                 if res_ncc[run][ind]==1.:
#                     correct_ncc += 1
#                 if res_nbc[run][ind]==1.:
#                     correct_nbc += 1
#             else:
#                 mean_imp_ncc+=1
#                 
#             # For ND + NBC
#             ind=tree.node.label.index(k)
#             if res_tree_nbc[run][ind]==1.:
#                 score_tree=score_tree+1.


                
            # For ND + NCC
            ind=tree.node.label.index(k)
            if res_tree_ncc[run][ind]==1.:
                score_tree_ncc=score_tree_ncc+1.
                sc_ncc = 1./(res_tree_ncc[run].sum())
                disc_score_tree_ncc = disc_score_tree_ncc + sc_ncc
                disc_u65_tree_ncc = disc_u65_tree_ncc - 1.2* sc_ncc*sc_ncc + 2.2 * sc_ncc
            if res_tree_ncc[run].sum()>1.:
                imp_tree_ncc = imp_tree_ncc +1.
                mean_imp_tree_ncc=mean_imp_tree_ncc+res_tree_ncc[run].sum()
                if res_tree_ncc[run][ind]==1.:
                    correct_tree_ncc += 1
#                 if res_tree_nbc[run][ind]==1.:
#                     correct_tree_nbc += 1 
            else:
                mean_imp_tree_ncc+=1
                
            total=total+1.
    
#     print '\\multicolumn{5}{|l|}{\\textit{' + f + '} (total evaluations: $'+  str(int(total)) + '$, number of classes: $' + str(len(dataArff.attribute_data['class'])) +'$) } \\\\'     
#     print '\\hline'
#     print 'set accuracy & ' + str(round(score_nbc/total*100,2)) + '\% & '  + str(round(score_tree/total*100,2)) + '\% & ' + str(round(score_ncc/total*100,2)) + '\% & ' + str(round(score_tree_ncc/total*100,2)) + '\%\\\\'
#     print 'disc acc & na &' + str(round(score_tree/total*100,2)) + '\% & ' + str(round(disc_score_ncc/total*100,2)) + '\% & ' + str(round(disc_score_tree_ncc/total*100,2)) + '\%\\\\'
#     print 'disc acc (u65) & na & ' + str(round(score_tree/total*100,2)) + '\% & ' + str(round(disc_u65_ncc/total*100,2)) + '\% & ' + str(round(disc_u65_tree_ncc/total*100,2)) + '\%\\\\'
#     print 'imprecision & na & na & ' + str(round(imp_ncc/total*100,2))+ '\% & ' + str(round(imp_tree_ncc/total*100,2)) + '\%\\\\'
#     print 'accuracy when imprecise & ' + str(round(correct_nbc/imp_ncc*100,2))+ '\% & ' + str(round(correct_tree_nbc/imp_tree_ncc*100,2))+ '\% & ' + str(round(correct_ncc/imp_ncc*100,2))+ '\% & ' + str(round(correct_tree_ncc/imp_tree_ncc*100,2))+ '\%\\\\ ' 
#     print 'mean imp & na & na & ' + str(round(mean_imp_ncc/total,2)) + ' & '  + str(round(mean_imp_tree_ncc/total,2))+ '\\\\'
#     print "\\hline"

    print c, round(score_tree_ncc/total*100,2), round(disc_score_tree_ncc/total*100,2), round(disc_u65_tree_ncc/total*100,2), round(imp_tree_ncc/total*100,2), round(correct_tree_ncc/imp_tree_ncc*100,2),round(mean_imp_tree_ncc/total,2)