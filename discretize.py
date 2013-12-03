'''
Created on 26 nov. 2013

@author: aitech
'''

from classifip import dataset


# Parameters :
files = ['abalone','ailerons','autoMpg','autoPrice','bank8FM','bank32NH','housingB',
         'housingCal','cpuAct','cpuSmall','aileronsDelta','elevators','friedman',
         'house8L','house16H','kinematics','mv','puma8NH','puma32H','servo',
         'stocks','ERA','ESL','LEV']


# For nested dichotomies and NCCOF
for f in files :
    print 'Processing file \"' + f + "\" ... "
    filename_in = 'classifip\\dataset\\test\\' + f + '.arff'
    filename_out = 'classifip\\dataset\\test\\' + f + '_dis.arff'
    name_class = 'class'
     
    # Script
    arff = dataset.arff.ArffFile()
     
    # Entry data file
    arff.load(filename_in)
     
    # Discretization of the class attribute, if it is not nominal already
    if arff.attribute_types[name_class] != 'nominal' :
        arff.discretize(discmet='eqfreq', numint=5, selfeat=name_class)
     
    arff.save(filename_out)
     
    # Discretize the rest of continuous variables
    dataset.discretize_ent(filename_out, filename_out)
    print "done"

outprint = []
# For logistic regression
for f in files :
    print 'Processing file \"' + f + "\" ... "
    filename_in = 'classifip\\dataset\\test\\' + f + '.arff'
    filename_out = 'classifip\\dataset\\test\\' + f + '_logit.arff'
    name_class = 'class'
    
    # Script
    arff = dataset.arff.ArffFile()
    
    # Entry data file
    arff.load(filename_in)
    
    # Discretization of the class attribute, if it is not nominal already
    if arff.attribute_types[name_class] != 'nominal' :
        arff.discretize(discmet='eqfreq', numint=5, selfeat=name_class)
    
    if any([arff.attribute_types[attr] == 'nominal' for attr in arff.attributes[:-1]]):
        print "There is a nominal attribute, manual treatment needed"
        outprint.append(f)
    
    arff.save(filename_out)
    
    print "done"
    
print "Files needing manual treatment : " + str(outprint)