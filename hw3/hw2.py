import pandas as pd
import re
import math

df_pos = pd.read_csv("hotelPosT-train.txt",sep='\t',usecols=[1], header=None)
df_neg = pd.read_csv("hotelNegT-train.txt",sep='\t',usecols=[1], header=None)

test =  pd.read_csv("test.txt",sep='\t',usecols=[0,1], header=None)

train_pwords = df_pos.iloc[:,0]
train_nwords = df_neg.iloc[:,0]

test_id = test.iloc[:,0]
test_words = test.iloc[:,1]

pos_words_d = dict()
#for each sameple tokenize the words and check to see if they exist
for i in range(len(train_pwords)):
    tokens = re.compile('\w+').findall(train_pwords[i])
    for j in range (len(tokens)):
        if ( tokens[j].lower() in pos_words_d.keys()):
            pos_words_d[tokens[j].lower()] += 1
        else:
            pos_words_d[tokens[j].lower()] = 1

neg_words_d = dict()
#for each sameple tokenize the words and check to see if they exist
for i in range(len(train_nwords)):
    tokens = re.compile('\w+').findall(train_nwords[i])
    for j in range (len(tokens)):

        if ( tokens[j].lower() in neg_words_d.keys()):
            neg_words_d[tokens[j].lower()] += 1
        else:
            neg_words_d[tokens[j].lower()] = 1


#print (pos_words_d)
#print (neg_words_d)

vocab_count = len(pos_words_d) + len(neg_words_d)
total_pos_words =  sum(pos_words_d.values())
total_neg_words =  sum(neg_words_d.values())
pos_base = vocab_count + total_pos_words
neg_base = vocab_count + total_neg_words

#print (vocab_count)
#print (total_pos_words)
#print (total_neg_words)

#for each sample tokenize the word and check their word counts in the positive and negative dictionaries. Use Bayes formula to compute the probablity

f = open('predictions.txt','w+')
neg_result = 0
pos_result = 0
for i in range(len(test_words)):
    pos_prob = 0
    neg_prob = 0
    tokens = re.compile('\w+').findall(test_words[i])
    for j in range (len(tokens)):
        #print ("checking for: " + str(tokens[j]) + "\n")
        if ( tokens[j] in pos_words_d.keys()):
            pos_prob += math.log10((pos_words_d[tokens[j]] + 1)/pos_base)
            #print (str(tokens[j]) + " found")
        else:
            pos_prob += math.log10(1/pos_base)
            #print (str(tokens[j]) + " not found")
        if (tokens[j] in neg_words_d.keys()):
            neg_prob += math.log10((neg_words_d[tokens[j]] + 1)/neg_base)
        else:
            neg_prob += math.log10(1/neg_base)

  #  pos_prob = math.pow(10, pos_prob)
  #  neg_prob = math.pow(10, neg_prob)

    print ("pos_prob:" + str(pos_prob))
    print ("neg_prob:" + str(neg_prob))    
    
    
    if ( pos_prob < neg_prob):
        result = "NEG"
        pos_result += 1
    else:
        result = "POS"
        neg_result += 1
    print (str(test_id[i]) +"\t"+str(pos_prob) + "\t" + str(neg_prob) + str(result))     
    f.write( str(test_id[i]) + "\t" + str(result) + "\n" )
f.close()
print ("neg results:" + str(neg_result) + "\t" + "pos results:" +  str(pos_result))


    
