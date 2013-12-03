from binarytree import models,clustering,dichotomies
import classifip
import time


test = classifip.dataset.arff.ArffFile()
test.load('binarytree\pendigits_dis.arff')
# test.load("soybean.arff")

# Dans le cas continu, on construit l'arbre sans discretization 
data_tree = classifip.dataset.arff.ArffFile() 
data_tree.load("pendigits.arff")

# Pour tester avec 'soybean'
# test.data = [list(row) for row in test.data if (all(row))]
# test.attribute_data['class'] = test.attribute_data['class'][0:15]

# Pour tester avec 'zoo'
# test = test.remove_col('animal')

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
        datatr.data = [x for i, x in enumerate(data.data) if i % K != k]
        datatst.data = [x for i, x in enumerate(data.data) if i % K == k]
        yield datatr, datatst

result=[]
true=[]
score=0.
imp=0.
total=0.
mean_imp=0.
score0=0.
imp0=0.
mean_imp0=0.
score_d=0.
sc = 0.
sc0 = 0.
sc_d = 0.

pre_correct = 0
pre_correct_d = 0
precis_correct = 0
precis_correct_d = 0


# c = test.attribute_data['class']
# tree = models.BinaryTree(label= c)

print "Tree Building..."
# l = models.ncc.NCC()
#           
# l.learn(test)
#   

clusters = clustering.clustering(test)
dico = dichotomies.dichotomies(data_tree)


clusters.distances = dico.dist_centroid(data_tree)
print dico.labels
clusters.labels = dico.labels
# clusters.compute(l)
# print clusters.distances
tree = clusters.clust(method='single')

# dico = dichotomies.dichotomies(data_tree)
# tree = dico.tree_kmeans(data_tree) 
tree.printTree()



start_time = time.time()
    
for training, validation in k_fold_cross_validation(test, K=10,random_seed=31415926,randomise=True):

    print 'learning'
    
    
    tree.learnAll(training)
    
    l = classifip.models.ncc.NCC()
    l.learn(training)

    
    print 'evaluating'
    tree.evaluate([x[0:len(x)-1] for x in validation.data],ncc_s_param=[2])    
    result = tree.decision_maximality()
    
    
    tree.evaluate([x[0:len(x)-1] for x in validation.data],ncc_s_param=[0.00001])    
    res0 = tree.decision_maximality()
    
    l_resu =  l.evaluate(validation.data, ncc_s_param=[2])
    
    res = []

    for i in range(0,len(l_resu)): 
        res.append(l_resu[i][0].nc_maximal_decision())
    

    
    trlab=[x[-1] for x in validation.data]
    true.append(trlab)
    run=0
    print len(trlab)
    
    for k in trlab:
        ind=tree.node.label.index(k)
        if result[run][ind]==1.:
            score=score+1.
            sc = sc + (1/result[run].sum())
        if result[run].sum()>1.:
            imp=imp+1.
            mean_imp=mean_imp+result[run].sum()
            if result[run][ind]==1.:
                pre_correct += 1
            if res0[run][ind]==1.:
                precis_correct += 1

        
        if res0[run][ind]==1.:
            score0=score0+1.

        
        ind = l.feature_values['class'].index(k)
        if res[run][ind]==1.:
            score_d=score_d+1.
            sc_d = sc_d + (1/res[run].sum())
        if res[run].sum()>1.:
            imp0=imp0+1.
            mean_imp0=mean_imp0+res[run].sum()
            if res[run][ind]==1.:
                pre_correct_d += 1
            ind=tree.node.label.index(k)
            if res0[run][ind]==1.:
                precis_correct_d += 1
        
                   
        total=total+1.
        run=run+1

print "===== Results ====="        
print 'total:' + str(total)     
print 'score:' + str(score) + '-' + str(score0) + '-' + str(score_d)
print 'score discounting acc:' + str(sc) + '-' + str(sc_d)
print 'imprecision:' + str(imp) + '-' + str(imp0)
print 'pred correcte quand imprecis: ' + str(pre_correct) + '-' + str(pre_correct_d)
print 'methode precise correcte quand imprecis: ' + str(precis_correct) + '-' + str(precis_correct_d)
if imp != 0 : print 'mean imp:' + str(mean_imp/imp) + '-' + str(mean_imp0/imp0)
    
    
end_time = time.time()

print("Elapsed time was %g seconds" % (end_time - start_time))
    
    
    
    
    
    
    
    
    
    