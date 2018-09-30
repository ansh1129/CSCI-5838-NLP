import pandas as pd
from collections import Counter

df = pd.read_csv("berp-POS-training.txt",sep='\t',usecols=[1,2], header=None)
#df = pd.read_csv("tmp.txt",sep='\t',usecols=[1,2], header=None)
test =  pd.read_csv("berp-POS-training.txt",sep='\t',usecols=[1,2], header=None)
#test =  pd.read_csv("test.txt",sep='\t',usecols=[1,2], header=None)

words = df.iloc[:,0]
tags = df.iloc[:,1]

test_words = test.iloc[:,0]
actual_tags = test.iloc[:,1]


words_d = dict()

for i,j in zip(words,tags):
    #check to see if the word exists
    if i in words_d.keys():
        #check to see if the tag exists for the word
        if j in words_d[i].keys():
            #update count for existing tag
            words_d[i][j] = words_d[i][j] + 1
        #create neww tag for the given word
        else:
            words_d[i][j] = 1
    #create new entry for word, and create new tag        
    else:
        words_d[i] = dict()
        words_d[i][j] = 1
    
#print(words_d)

#print the most frequent tag
#for word in words_d.keys():
#    for tag in words_d[word].keys():
#        print( word,":", max(words_d[word], key=words_d[word].get))        

#total count for all the tags
tag_counts_d = dict()
for i in tags:
  tag_counts_d[i] = tag_counts_d.get(i, 0) + 1

#print (tag_counts_d)

#for each word calculate observation likelihood
observations_d = dict()
for word in words_d.keys():
    observations_d[word] = dict()
    for tag in words_d[word].keys():
        observations_d[word][tag] = words_d[word][tag]/tag_counts_d[tag]

print(observations_d)

#next state
nxt_state_d = dict()

for i in range(len(tags)-1):
    if tags[i] in nxt_state_d.keys():
        if tags[i+1] in nxt_state_d[tags[i]].keys():
                nxt_state_d[tags[i]][tags[i+1]] = nxt_state_d[tags[i]][tags[i+1]]+1
        else:
                nxt_state_d[tags[i]][tags[i+1]] = 1                
    else:
        nxt_state_d[tags[i]] = dict()
        nxt_state_d[tags[i]][tags[i+1]] = 1

        
for i in nxt_state_d:
    print ("\n pre tag:",i, nxt_state_d[i])

#add 1 to all the transitions
#get all the possible tags from nxt_state_d
for pre_tag in nxt_state_d.keys():
    for pre_tag_2 in nxt_state_d.keys():
        if pre_tag_2 in nxt_state_d[pre_tag].keys():
            nxt_state_d[pre_tag][pre_tag_2] = nxt_state_d[pre_tag][pre_tag_2] + 1
        else:
            nxt_state_d[pre_tag][pre_tag_2] = dict()
            nxt_state_d[pre_tag][pre_tag_2] = 1

            
for i in nxt_state_d:
    print ("\n",nxt_state_d[i])


        
#convert next state to probabilities with add one smoothing
vocab_count = len(observations_d)
for key in nxt_state_d:
    for tags_key in nxt_state_d[key]:
        #print 
        nxt_state_d[key][tags_key] = nxt_state_d[key][tags_key]/(tag_counts_d[key] + vocab_count)
        
for i in nxt_state_d:
    print ("\n",nxt_state_d[i])        


#test_words = ['tell','me','about']
col1_d = dict()
count = 0
predicted_tags = dict()

#viterbi algorithm
for i in test_words:
    col1_d = dict()
    predicted_tags[count] = dict()
    print("checking for word:",i)
    #find the word with observations
    if i in observations_d.keys():
        if (count == 0 ):
            for obs_key in observations_d[i].keys():
                #look for .->tags transition.
                #for key in nxt_state_d['.']:
                if obs_key in nxt_state_d['.'].keys():
                    print(". ->",key,":",nxt_state_d['.'][obs_key],"possible tag:",obs_key)
                    col1_d[obs_key] = nxt_state_d['.'][obs_key] * observations_d[i][obs_key] 
                    print("possible vt value:",col1_d[obs_key])
                else:
                    print("Next tag not found in.->",obs_key,"transition")

        else:
            for obs_key in observations_d[i].keys():
                #look for previous tag-> new tags transition.
                #for key in nxt_state_d[prv_tag]:
                if obs_key in nxt_state_d[prev_tag].keys():
                    print(prev_tag,"->",key,":",nxt_state_d[prev_tag][obs_key],"possible tag:",obs_key)
                    col1_d[obs_key] = nxt_state_d[prev_tag][obs_key] * observations_d[i][obs_key] *prev_tag_max_vt 
                    print("possible vt value:",col1_d[obs_key])                    
                else:
                    print("Next tag not found in",prev_tag,"->",obs_key,"transition")            

        #select the max from all the possible entries in col1
        if ( not col1_d):
            print("Empty Dict")
            predicted_tags[count][i] = 'NN'
            prev_tag = 'NN'
            prev_tag_max_vt = 0.000000001
        else:
            predicted_tags[count][i] = max(col1_d, key=col1_d.get)
            prev_tag_max_vt = col1_d[predicted_tags[count][i]]
            prev_tag = predicted_tags[count][i]
        print( "Tag is:", predicted_tags[count][i], "vt valu:", prev_tag_max_vt)
    elif ( i == '.'):
        prev_tag_max_vt = 1
        prev_tag = i
    else:
        print("word not found in the observation list")
    count = count + 1

#for i in predicted_tags:
#    print ("\n",predicted_tags[i])        
#print(actual_tags)

count = 0
#for i,j in zip(predicted_tags[count],actual_tags):
for j in actual_tags:
    for i in predicted_tags[count]:
        if (predicted_tags[count][i] != j):
            print(i,":",predicted_tags[count][i],"vs",j)
    count = count + 1


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
