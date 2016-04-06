#from glove import Glove, Corpus;
import gensim;
import numpy;
from numpy import *
from wordnet_utils import *;
from read_text_embeddings import *;
from sklearn.metrics.pairwise import cosine_similarity;


def get_text_embeddings(filename):
	model = read_text_embeddings(filename);
	return (model);

def get_glove_model(filename):
	model = Glove.load_stanford(filename)
	print "loaded glove"
	assert model.word_vectors is not None
	return (model);

def get_glove_vector(word,model):
	vec=[];
	try:
		vec=model.word_vectors[model.dictionary[word]]
	except:
		vec=[0]*300;
	return (vec);

def glove_test():
	word="hello";
	#model=get_glove_model('w2vmodels/glove.6B.300d.txt');
	#print get_glove_vector(word,model);
	model=getEmbeddingModel(3);
	print getWordEmbedding(word,model,3);

def embedding_test_word_embeddings():
	embeddingtype=2;
	ph1="eastern Afghan explosion";
	ph2="At least Chinese coal mine explosion";
	model=getEmbeddingModel(embeddingtype);
	res=getEmbeddingSimilarity(ph1,ph2,model,embeddingtype,0.7);
	print res
	

def getWordVecW2v(word,w2v_model):
        samp=w2v_model['computer']; 
        vec=[0]*len(samp);
        try:
                vec=w2v_model[word];
        except: 
                vec=[0]*len(samp);
        return (vec)

def getWordEmbedding(word,embeddingmodel,embeddingtype):
	if embeddingtype==3:
		return  (get_glove_vector(word,embeddingmodel));
	else:
		return  (getWordVecW2v(word,embeddingmodel));
	


def getUnnormalizedPhraseEmbedding(phrase,embeddingmodel,embeddingtype):
	#phrasenew=phrase.strip().lower();
	phrasenew=phrase.strip();
	samp=getWordEmbedding('computer',embeddingmodel,embeddingtype);
        vec=numpy.array([0]*len(samp));
	den=0;
        for word in phrasenew.split():
               	den=den+1;
	        vec=vec+numpy.array(getWordEmbedding(word,embeddingmodel,embeddingtype));
        #vec=vec/den;
        return (vec.tolist());

def normalizeEmbedding(vec):
        vec=numpy.array(vec);
        norm=numpy.linalg.norm(vec);
        if norm!=0:
                vec=vec/numpy.linalg.norm(vec)
        return (vec.tolist());

def getPhraseEmbedding(phrase,embeddingmodel,embeddingtype):
	vec=getUnnormalizedPhraseEmbedding(phrase,embeddingmodel,embeddingtype);
        vec=numpy.array(vec);
        norm=numpy.linalg.norm(vec);
        if norm!=0:
                vec=vec/len(phrase.split());
                #vec=vec/numpy.linalg.norm(vec)
        return (vec.tolist());

def getUnnormalizedEmbeddingSimilarity(chunk1, chunk2, embeddingmodel,embeddingtype):
        chvec1=getUnnormalizedPhraseEmbedding(chunk1,embeddingmodel,embeddingtype);
        chvec2=getUnnormalizedPhraseEmbedding(chunk2,embeddingmodel,embeddingtype);
        sim=numpy.dot(numpy.array(chvec1), numpy.array(chvec2));
        if math.isnan(sim) or sim<0.4:
        #if math.isnan(sim) or sim<0.4:
                sim=-1;
                #sim=0;
        return (sim);

def getPhraseEmbeddingVectors(phrase,embeddingmodel,embeddingtype):
	phrasenew=phrase.strip();
	samp=getWordEmbedding('computer',embeddingmodel,embeddingtype);
        vec=numpy.array([0]*len(samp));
	embeddingvecs=[];
        for word in phrasenew.split():
	        embeddingvecs.append(numpy.array(getWordEmbedding(word,embeddingmodel,embeddingtype)));
	return (embeddingvecs);

def getEmbeddingSimilarityMax(chunk1, chunk2, embeddingmodel,embeddingtype,cutoff=0.35):
	vecs1=getPhraseEmbeddingVectors(chunk1,embeddingmodel,embeddingtype);
	vecs2=getPhraseEmbeddingVectors(chunk2,embeddingmodel,embeddingtype);
	maxsim=-1;
	for vec1 in vecs1:
		for vec2 in vecs2: 
			sim=cosine_similarity(vec1,vec2)[0][0];
			if sim>maxsim:
				maxsim=sim;
	if maxsim<cutoff:
		maxsim=-1;
	return(maxsim);

def getEmbeddingSimilarity(chunk1, chunk2, embeddingmodel,embeddingtype,cutoff=0.35):
        chvec1=getPhraseEmbedding(chunk1,embeddingmodel,embeddingtype);
        chvec2=getPhraseEmbedding(chunk2,embeddingmodel,embeddingtype);
	sim=cosine_similarity(numpy.array(chvec1), numpy.array(chvec2))[0][0];
        if math.isnan(sim) or sim<cutoff:
        #if math.isnan(sim) or sim<0.4: #current best
                sim=-1;
                #sim=0;
        return (sim);

def getAggregatedSimilarityOld(chunk1, chunk2,embeddingmodel,embeddingtype):
        res1=getEmbeddingSimilarity(chunk1, chunk2,embeddingmodel,embeddingtype);
        if res1 > 0.8:
		return res1;
        res=aggregage_wordnet_similarity(chunk1,chunk2);
        if res<0.15:
                res=-1;
        return res;

def getWordnetSimilarity(chunk1, chunk2):
	res=aggregage_wordnet_similarity(chunk1,chunk2);
	if res<0.15:
		res=-1;
	return res;

def getEmbeddingModel(embeddingtype):
	if embeddingtype==3:
		print "loading glove embeddings...."
	        model=get_glove_model('w2vmodels/glove.6B.300d.txt');
	elif embeddingtype==1:
		print "loading google news w2v embeddings...."
		model=gensim.models.Word2Vec.load_word2vec_format('w2vmodels/GoogleNews-vectors-negative300.bin', binary= True);
	elif embeddingtype==2:
		print "loading text8 w2v embeddings...."
	        model=gensim.models.Word2Vec.load('text8.w2v.bin')
	else:
		print "Not any known embeddings...."
		model=None;
	return (model);

#glove_test();
#embedding_test_word_embeddings();
#embedding_test_phrase_embeddings();
