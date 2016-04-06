from nltk.corpus import wordnet as wn
from itertools import chain
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk.stem.porter import PorterStemmer
import nltk
from coreNlpUtil import *
from sklearn.metrics import jaccard_similarity_score
#from nltk.parse.stanford import StanfordDependencyParser
import editdistance
from sklearn.metrics.pairwise import cosine_similarity;
from paramsHeader import *;
import num2words
from django.utils import encoding;
from copy import deepcopy

def convert_unicode_to_string(x):
    """
    >>> convert_unicode_to_string(u'ni\xf1era')
    'niera'
    """
    return encoding.smart_str(x, encoding='ascii', errors='ignore')


def flatten(l):
    return list( chain ( *l ) )

def get_variants(word):
    synonyms = []
    for syn in wn.synsets(word):
        synonyms.extend( syn.lemmas() )
	for s in syn.similar_tos():
		synonyms.extend(s.lemmas())
	for s in syn.hypernyms():
		synonyms.extend(s.lemmas())
	for s in syn.hyponyms():
		synonyms.extend(s.lemmas())
    return synonyms

def get_hypernyms(word):
    hypernyms=[]
    for syn in wn.synsets(word):
	for s in syn.hypernyms():
		hypernyms.extend(s.lemmas())
    return sorted( [v.name() for v in hypernyms] )

def get_hyponyms(word):
    h=[]
    for syn in wn.synsets(word):
	for s in syn.hyponyms():
		h.extend(s.lemmas())
    return sorted( [v.name() for v in h] )

def get_editdistance(sent1,sent2):
    return editdistance.eval(sent1, sent2)

def isSubsetOf(sent1,sent2):
    s1=sent1.lower().strip();
    s2=sent2.lower().strip();
    words1=filter(None, s1.split());
    words2=filter(None, s2.split());
    flag1=1;
    for word in words1:
	if word not in s2:
		flag1=0;
    
    flag2=1;
    for word in words2:
	if word not in s1:
		flag2=0;
    if flag1==1 and flag2==1:
	return 0;
    elif flag1==1 and flag2==0:
	return 1;
    elif flag1==0 and flag2==1:
	return 2;
    elif flag1==0 and flag2==0:
	return 3;

def containsNumber(sent1):
    s1=sent1.lower().strip();
    words1=filter(None, s1.split());
    for word in words1:
	if word.isdigit():
		return 1;
    return 0;

#is sentance2 a hypernym of sentance 1
def is_hypernym(sent1,sent2):
    s1=deepcopy(convert_unicode_to_string(sent1));
    s2=deepcopy(convert_unicode_to_string(sent2));
    s1=s1.lower().strip();
    s2=s2.lower().strip();
    
    words1=filter(None, s1.split());
    words2=filter(None, s2.split());
	
    hyperset=[];
    for word in words1:
	wss=wn.synsets(word)
	if len(wss)<1:
		return 0;
	wss=wss[0];
	
	wordhyper=  set([i for i in wss.closure(lambda s:s.hypernyms())])
	for word2 in words2:
		wss2= wn.synsets(word2)
		if len(wss2)<1:
			continue;

		wss2=wss2[0];
		if wss2 in wordhyper:
			return 1;
    return 0;

#is sentance2 a hypernym of sentance 1
def is_hyponym(sent1,sent2):
    s1=deepcopy(convert_unicode_to_string(sent1));
    s2=deepcopy(convert_unicode_to_string(sent2));
    s1=s1.lower().strip();
    s2=s2.lower().strip();
    
    words1=filter(None, s1.split());
    words2=filter(None, s2.split());
	
    hyposet=[];
    for word in words1:
	wss=wn.synsets(word)
	if len(wss)<1:
		return 0;
	wss=wss[0];
	wordhypo=  set([i for i in wss.closure(lambda s:s.hyponyms())])
	for word2 in words2:
		wss2= wn.synsets(word2)
		if len(wss2)<1:
			continue;
		wss2=wss2[0];
		if wss2 in wordhypo:
			return 1;
    return 0;

	

def get_normalized_editscore_sentences(sent1,sent2,dictParams=dictHeadlines):
    s1=sent1.lower().strip();
    s2=sent2.lower().strip();
    
    words1=filter(None, s1.split());
    words2=filter(None, s2.split());
    if len(words1)<len(words2):
	tmp=words1;
	words1=words2;
	words2=tmp;
    #words2=s2.split();
    dsum=0;
    
    for w1 in words1:
	d=-1;
    	for w2 in words2:
		ed=get_normalized_editscore_words(w1.strip(),w2.strip());
		if ed>d:
			d=ed;
	if d!=-1:
		dsum=dsum+d;
    score=(dsum*1.0/len(words1));
    #if score<0.5: #For students dataest
    #if score<0.7: #For headlines dataset
    if score<dictParams['EditClamp']:
	score=-1;
    return(score);
    	
	
def get_normalized_editscore_words(sent1,sent2):
    dist=editdistance.eval(sent1, sent2)
    l1=len(sent1);
    l2=len(sent2);
    l=l1;
    if l2>l1:
	l=l2;
    return ( 1-dist/(1.0*l) );
 

def aggregage_wordnet_similarity(sent1,sent2):
    return ((1.32)*common_words(sent1,sent2)-(0.27)*antonym_count(sent1,sent2)+(0.85)*synonym_count(sent1,sent2)+(-1.93)*get_wordnet_sent_similarity(sent1,sent2))/4.0

def bigramSimilarity(vectorizer,chunk1,chunk2):

    r1= vectorizer.transform([chunk1]).toarray().tolist()[0]
    r2= vectorizer.transform([chunk2]).toarray().tolist()[0]
    sim=cosine_similarity(r1,r2);
    return(sim[0][0]);
	
def common_words(sent1,sent2):
    # remove stop words, lemmatise and return count of common words
    porter = PorterStemmer()
    #stop = stopwords.words('english')
    s1_words =  [porter.stem(i.lower()) for i in wordpunct_tokenize(sent1)  ]
    s2_words =  [porter.stem(i.lower()) for i in wordpunct_tokenize(sent2)  ]
    s1 = set(s1_words)
    s2 = set(s2_words)
    return len(s1.intersection(s2)) / ((len(s1)+0.1+len(s2))/2.0) # normalised 

def antonym_count(sent1,sent2):
    porter = PorterStemmer()
    #stop = stopwords.words('english')
    s1_words =  [i.lower() for i in wordpunct_tokenize(sent1) ]
    s2_words =  [i.lower() for i in wordpunct_tokenize(sent2) ]
    s1_all = []
    s2_all = []

    for w in s1_words:
        s1_all.extend(get_antonyms(w))
    for w in s2_words:
        s2_all.extend(get_antonyms(w))
    
    
    s1_w = set(s1_words)
    s2_w = set(s2_words)
    s1_a = set(s1_all)
    s2_a = set(s2_all)
    #print len(s1.intersection(s2))
    return len(s1_w.intersection(s2_a))+len(s2_w.intersection(s1_a))/ ((len(s1_all)+0.1+len(s2_all))/2.0) # normalised 

def hyponym_count(sent1,sent2):
    
    s1_words =  [i.lower() for i in wordpunct_tokenize(sent1) ]
    s2_words =  [i.lower() for i in wordpunct_tokenize(sent2) ]
    s1_all = []
    s2_all = []

    for w in s1_words:
        s1_all.extend(get_hyponyms(w))
    for w in s2_words:
	s2_all.extend(get_hyponyms(w))
    w1_hyponym = len(set(s1_words).intersection(set(s2_all)))
    w2_hyponym = len(set(s2_words).intersection(set(s1_all)))
    return w1_hyponym-w2_hyponym

def hypernym_count(sent1,sent2):

    s1_words =  [i.lower() for i in wordpunct_tokenize(sent1) ]
    s2_words =  [i.lower() for i in wordpunct_tokenize(sent2) ]
    s1_all = []
    s2_all = []

    for w in s1_words:
        s1_all.extend(get_hypernyms(w))
    for w in s2_words:
	s2_all.extend(get_hypernyms(w))
    w1_hypernym = len(set(s1_words).intersection(set(s2_all)))
    w2_hypernym = len(set(s2_words).intersection(set(s1_all)))
    return w1_hypernym-w2_hypernym

def synonym_count(sent1,sent2):
    porter = PorterStemmer()
    #stop = stopwords.words('english')
    s1_words =  [i.lower() for i in wordpunct_tokenize(sent1) ]
    s2_words =  [i.lower() for i in wordpunct_tokenize(sent2) ]
    s1_all = []
    s2_all = []

    for w in s1_words:
        s1_all.extend(get_synonyms(w))
    for w in s2_words:
        s2_all.extend(get_synonyms(w))
    s1_all.extend(s1_words)    
    s2_all.extend(s2_words)
    
    s1 = set(s1_all)
    s2 = set(s2_all)
    #print len(s1.intersection(s2))
    return len(s1.intersection(s2))/ ((len(s1_all)+0.1+len(s2_all))/2.0) # normalised 
def get_hyponym(word):
    return sorted( [v.name() for v in get_variants(word)] )

def get_synonyms(word):
    return sorted( [v.name() for v in get_variants(word)] )

def get_antonyms(word):
    antonyms = flatten( [v.antonyms() for v in get_variants(word)] )
    return sorted( [a.name() for a in antonyms] )

def get_wordnet_sent_similarity(sent1,sent2):
    sim = 0.0
    for w1 in sent1.split():
        for w2 in sent2.split():
            #print get_path_similarity(w1,w2)
            sim=sim+get_path_similarity(w1,w2)
    sim = sim/(len(sent1)+len(sent2))
    return sim


def get_path_similarity(w1,w2):
    # get synsets and then path similarity with them 
    sim=0.0
    w1_syn = wn.synsets(w1)
    if len(w1_syn)>=1:
        w2_syn = wn.synsets(w2)
        if len(w2_syn)>=1:
            sim = w1_syn[0].path_similarity(w2_syn[0])
            if sim ==None:
                sim=0.0
            return sim
    
    return 0.0


def DistJaccard(l1, l2):
    str1 = set(l1)
    str2 = set(l2)
    if len(str1)>0 and len(str2)>0:
        return float(len(str1 & str2)) / len(str1 | str2)
    else:
        return 0.0

# checks if one of the chunks has negation term
def is_negation(sent1,sent2):
    if 'not' in sent1.lower() and 'not' not in sent2.lower():
        return 1
    elif 'not' not in sent1.lower() and 'not' in sent2.lower():
        return 1
    elif 'n\'t' in sent1.lower() and 'not' not in sent2.lower():
        return 1
    elif 'n\'t' not in sent1.lower() and 'not' in sent2.lower():
        return 1
    elif 'never' in sent1.lower() and 'not' not in sent2.lower():
        return 1
    elif 'never' not in sent1.lower() and 'not' in sent2.lower():
        return 1
    else:
        return 0    


# counts the difference of adjectives and adverbs in two sentences
def adjective_count_diff(sent1,sent2):
    s1_pos=[a[1] for a in get_pos_tags(sent1)]
    s2_pos=[a[1] for a in get_pos_tags(sent2)]
    ad = (s1_pos.count('RB')+s1_pos.count('RBS')+s1_pos.count('RBR')-s2_pos.count('RB')-s2_pos.count('RBS')-s2_pos.count('RBR'))/max(len(s1_pos),len(s2_pos))
    return ad+(s1_pos.count('JJ')+s1_pos.count('JJS')+s1_pos.count('JJR')-s2_pos.count('JJ')-s2_pos.count('JJS')-s2_pos.count('JJR'))/max(len(s1_pos),len(s2_pos))

def get_pos_tags(sent):
    text=nltk.word_tokenize(sent)
    return nltk.pos_tag(text)

#----------- new features ------------------
def normalise(chunk):
    words = chunk.split()
    out =[]
    prev_word =""
    for word in words:
        # removing numbers
        if word.isdigit():
            out.append(num2words.num2words(float(word)).replace('-',' '))
        # removing short form of million
        elif word[:-1].isdigit() and word[-1].lower()=='m':
            out.append(word[:-1]+" million")
        # removing short form of million
        elif word.lower()=='m' and prev_word.isdigit():
            out.append('million')
        # billion 
        elif word[:-1].isdigit() and word[-1].lower()=='bn':
            out.append(word[:-1]+" billion")
        # removing short form of billion
        elif word.lower()=='bn' and prev_word.isdigit():
            out.append('billion')
        elif word.lower()=='wk':
            out.append('week')
        elif word.lower()=='pc' or word.lower()=='pct' or word.lower()=='pcd':
            out.append('percent')
        elif word.lower()=='p.m' or word.lower()=='p.m.':
            out.append('PM') 
        elif word.lower()=='a.m' or word.lower()=='a.m.':
            out.append('AM') 
        elif word.lower()=='pair' or word.lower()=='couple':
            out.append('two')
        elif word.lower()=='u.s.' or word.lower()=='u.s' or word.lower()== 'u.s.a'or word.lower()== 'u.s.a.' or word.lower()=='united states' or word.lower()=='united states of america':
            out.append('USA')
        else:
            out.append(" "+word)
        prev_word=word
    return " ".join(out)


print get_normalized_editscore_sentences("My name","the name is");

# -------- taken from monolingual aligner 

################################################################################
def loadPPDB(ppdbFileName = './monolingual-word-aligner-master/Resources/ppdb-1.0-xxxl-lexical.extended.synonyms.uniquepairs'):

    global ppdbSim
    global ppdbDict

    count = 0
    
    ppdbFile = open(ppdbFileName, 'r')
    for line in ppdbFile:
        if line == '\n':
            continue
        tokens = line.split()
        tokens[1] = tokens[1].strip()
        ppdbDict[(tokens[0], tokens[1])] = ppdbSim
        count += 1

################################################################################

def presentInPPDBword(sent1,sent2):
    score = 0
    for word1 in sent1.split():
        for word2 in sent2.split():
            score+=presentInPPDB(word1,word2)
    return score
################################################################################
def presentInPPDB(word1, word2):

    global ppdbDict

    if (word1.lower(), word2.lower()) in ppdbDict:
        return ppdbDict[(word1.lower(), word2.lower())]
    if (word2.lower(), word1.lower()) in ppdbDict:
        return ppdbDict[(word2.lower(), word1.lower())]
    return 0.0
    
################################################################################
