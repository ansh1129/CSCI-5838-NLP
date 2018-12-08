import pandas as pd
import math
from collections import Counter

df = pd.read_csv("gene-trainF18.txt",sep='\t',usecols=[1,2], header=None)
test =  pd.read_csv("test.txt",sep='\t',usecols=[1,2], header=None)
#df = pd.read_csv("train.txt",sep='\t',usecols=[1,2], header=None)
#test =  pd.read_csv("test2.txt",sep='\t',usecols=[1,2], header=None)


words = df.iloc[:,0]
tags = df.iloc[:,1]
test_words = test.iloc[:,0]
actual_tags = test.iloc[:,1]

words_d = dict()

#count total words and their respective frequencies
for i,j in zip(words,tags):
    #check to see if the word exists
    if i in words_d.keys():
        #check to see if the tag exists for the word
        if j in words_d[i].keys():
            #update count for existing tag
            words_d[i][j] += 1
        #create neww tag for the given word
        else:
            words_d[i][j] = 1
    #create new entry for word, and create new tag        
    else:
        words_d[i] = dict()
        words_d[i][j] = 1

vocab_count = len(words_d)

#print the most frequent tag
#for word in words_d.keys():
#    for tag in words_d[word].keys():
#        print( word,":", max(words_d[word], key=words_d[word].get))        

#total count for all the tags
tag_counts_d = dict()
for i in tags:
  tag_counts_d[i] = tag_counts_d.get(i, 0) + 1
#print (tag_counts_d)

#next state
nxt_state_d = dict()

#initialize to have all tags set to 0
nxt_state_d['B'] = dict()
nxt_state_d['B']['B'] = 1
nxt_state_d['B']['I'] = 1
nxt_state_d['B']['O'] = 1

nxt_state_d['I'] = dict()
nxt_state_d['I']['B'] = 1
nxt_state_d['I']['I'] = 1
nxt_state_d['I']['O'] = 1

nxt_state_d['O'] = dict()
nxt_state_d['O']['B'] = 1
nxt_state_d['O']['I'] = 1
nxt_state_d['O']['O'] = 1

#for i in nxt_state_d:
#    print ("\n pre tag:",i, nxt_state_d[i])



for i in range(len(tags)-1):
    if (tags[i+1] == '.'):
        continue
    elif tags[i] in nxt_state_d.keys():
        if tags[i+1] in nxt_state_d[tags[i]].keys():
                nxt_state_d[tags[i]][tags[i+1]] += 1
        else:
                nxt_state_d[tags[i]][tags[i+1]] = 1                
    else:
        nxt_state_d[tags[i]] = dict()
        nxt_state_d[tags[i]][tags[i+1]] = 1

        
#for i in nxt_state_d:
#    print ("\n pre tag:",i, nxt_state_d[i])

#add 1 to tags for all the observations,
#for each key in words_d, check to see if a given tag exists,otherwise create a new one.
for word in words_d.keys():
    for tag in nxt_state_d.keys():
        if tag in words_d[word].keys():
            words_d[word][tag] += 1
        else:
            words_d[word][tag] = dict()
            words_d[word][tag] = 1


#for each word calculate observation likelihood
observations_d = dict()
for word in words_d.keys():
    observations_d[word] = dict()
    for tag in words_d[word].keys():
        observations_d[word][tag] = words_d[word][tag]/(tag_counts_d[tag] + vocab_count)

#for i in observations_d:        
#    print("\n word:",i,"->", observations_d[i])
unk_obs_d = dict()
for tag in nxt_state_d.keys():
    unk_obs_d[tag] = 1/(tag_counts_d[tag] + vocab_count)


#add 1 to all the transitions
#get all the possible tags from nxt_state_d
for pre_tag in nxt_state_d.keys():
    for pre_tag_2 in nxt_state_d.keys():
        if pre_tag_2 in nxt_state_d[pre_tag].keys():
            nxt_state_d[pre_tag][pre_tag_2] += 1
        else:
            nxt_state_d[pre_tag][pre_tag_2] = dict()
            nxt_state_d[pre_tag][pre_tag_2] = 1

            
#for i in nxt_state_d:
#    print ("\n",nxt_state_d[i])


        
#convert next state to probabilities with add one smoothing
for key in nxt_state_d:
    for tags_key in nxt_state_d[key]:
        #print 
        nxt_state_d[key][tags_key] = nxt_state_d[key][tags_key]/(tag_counts_d[key] + vocab_count)
        
#for i in nxt_state_d:
#    print ("\n",nxt_state_d[i])        


count = 0
predicted_tags = dict()
unknown_words = []
prev_tag_max_vt = dict()
start_sent = 1

temp_vt_d = dict()
for tag in nxt_state_d.keys():
    temp_vt_d[tag] = 1
    
#viterbi algorithm
for i in test_words:
    col1_d = dict()
    predicted_tags[count] = dict()
#    print("checking for word:",i)
    #find the word with observations
    if i in observations_d.keys():
        if (start_sent == 1 ):
            for obs_key in observations_d[i].keys():
                #look for .->tags transition.
                #for key in nxt_state_d['.']:
                
                #print(". ->",obs_key,":",nxt_state_d['.'][obs_key])
                
                col1_d[obs_key] = nxt_state_d['O'][obs_key] * observations_d[i][obs_key]
                
                #print("possible vt value:",col1_d[obs_key])
                start_sent = 0
            #select the max from all the possible entries in col1
            predicted_tags[count][i] = max(col1_d, key=col1_d.get)
            prev_tag_max_vt = col1_d                
        elif ( i == '.'):
            start_sent = 1
            predicted_tags[count][i] = 'O'
        else:
            for obs_key in observations_d[i].keys():
                #look for previous tag-> new tags transition.
                #for key in nxt_state_d[prv_tag]:
                inter_tag_vt_d = dict()
                for prev_tag in prev_tag_max_vt.keys():
                    #print(prev_tag,"->",obs_key,":",nxt_state_d[prev_tag][obs_key])
                    inter_tag_vt_d[prev_tag] = nxt_state_d[prev_tag][obs_key] * observations_d[i][obs_key] * prev_tag_max_vt[prev_tag]
                    #inter_tag_vt_d[prev_tag] = math.pow(10, (math.log10(nxt_state_d[prev_tag][obs_key]) + math.log10(observations_d[i][obs_key]) + math.log10(prev_tag_max_vt[prev_tag])))
                col1_d[obs_key] = max(inter_tag_vt_d.values())
                #print("possible vt value:",col1_d[obs_key])
            #select the max from all the possible entries in col1                
            predicted_tags[count][i] = max(col1_d, key=col1_d.get)
            prev_tag_max_vt = col1_d                
    else:
       # print("word not found in the observation list")
        if i not in unknown_words:
            unknown_words.append(i)

        if (start_sent == 1):
            for tag in unk_obs_d.keys():
                col1_d[tag] = nxt_state_d['O'][tag] * unk_obs_d[tag]
            predicted_tags[count][i] = max(col1_d, key=col1_d.get)
            prev_tag_max_vt = col1_d
            start_sent = 0                
        else:
            for tag in unk_obs_d.keys():                
                inter_tag_vt_d = dict()
                #print(nxt_state_d)
                for prev_tag in prev_tag_max_vt.keys():
                    #   print(prev_tag,"->",obs_key,":",nxt_state_d[prev_tag][obs_key])            
                    inter_tag_vt_d[prev_tag] = nxt_state_d[prev_tag][tag] * prev_tag_max_vt[prev_tag] * unk_obs_d[tag]
                col1_d[tag] = max(inter_tag_vt_d.values())
                #print("possible vt value:",col1_d[obs_key])                    
                
                #select the max from all the possible entries in col1
                predicted_tags[count][i] = max(col1_d, key=col1_d.get)
                prev_tag_max_vt = col1_d
                #print( "Tag is:", predicted_tags[count][i], "vt value:", prev_tag_max_vt)

    count = count + 1

#for i in predicted_tags:
#    print ("\n",predicted_tags[i])        
#print(actual_tags)

count = 0
unknown_mispred = 0
total_mispred = 0
correct_pred = []
incorrect_pred = []

correct_pred_gene = []
correct_word_gene = []
incorrect_pred_gene = []
incorrect_word_gene = []

#for i,j in zip(predicted_tags[count],actual_tags):
for j,k in zip(actual_tags, test_words):
    for i in predicted_tags[count]:
        if (predicted_tags[count][i] != j):
            print(i,":",predicted_tags[count][i],"vs",j)
            total_mispred +=1
            if (j == 'B' or j == 'I'):
                incorrect_pred_gene.append(j)
                incorrect_word_gene.append(k)
            
            if i not in incorrect_pred and i not in unknown_words:
                incorrect_pred.append(i)
            if i in unknown_words:
                unknown_mispred += 1
        else:
            #if i not in correct_pred and i not in unknown_words:
            correct_pred.append(i)
            if (j == 'B' or j =='I'):
                correct_pred_gene.append(j)
                #print ("adding ",j,"and:",k)
                correct_word_gene.append(k)
                
    count = count + 1

common_words = list(set(correct_pred).intersection(incorrect_pred))

    
print ("total mispred:", total_mispred)
print ("total unknown words:", len(unknown_words),"total unknown mispred:", unknown_mispred)
print ("\nunk words:",unknown_words)
#print ("\n incorrect words:",incorrect_pred, "total:", len(incorrect_pred))
#print ("\n common words:",common_words, "total:", len(common_words))
print("\n incorrect pred gene:", incorrect_pred_gene, "incorrect word gene", incorrect_word_gene, "total:", len(incorrect_word_gene))
print("\n correct pred gene:", correct_pred_gene, "correct word gene", correct_word_gene, "total:", len(correct_word_gene))


f = open('predictions.txt','w+')
count = 1
for i in predicted_tags.keys():
    for word in predicted_tags[i].keys():
        f.write(str(count)+"\t"+str(word)+"\t"+str(predicted_tags[i][word]))
        if (word =='.'):
            count = 0
            f.write("\n")
    f.write("\n")
    count = count+1
 
f.close() 
