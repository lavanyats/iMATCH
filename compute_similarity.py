#from glove import Glove, Corpus;
import gensim;
import numpy;
import sys;
from numpy import *
from wordnet_utils import *;
from read_text_embeddings import *;
from embedding_funcs import *;
sys.path.insert(0, '../../resources/monolingual-word-aligner')
from aligner import *
from paramsHeader import *;




def getAggregatedSimilarity(chunk1, chunk2,sent1,sent2,embeddingmodel,embeddingtype,matchtype,debug=0,vectorizer=None,cutoff=0.35,dictParams=dictHeadlines):
	res=-1;

	if chunk1.strip() == chunk2.strip():
		return (1);
	if matchtype==1: #embedding
	        res=getEmbeddingSimilarity(chunk1, chunk2,embeddingmodel,embeddingtype);
		if debug==1:
			print "getEmbeddingSimilarity",res;
	elif matchtype==4:  
		res=((1.32)*common_words(chunk1,chunk2)
			-(0.27)*antonym_count(chunk1,chunk2)
			+(0.85)*synonym_count(chunk1,chunk2)
			+(-1.93)*get_wordnet_sent_similarity(chunk1,chunk2)
			+(0)*getEmbeddingSimilarity(chunk1, chunk2,embeddingmodel,embeddingtype)
			)/5.0;
		if debug==1:
			print "common words",common_words(chunk1,chunk2);
			print "antonym_count",antonym_count(chunk1,chunk2);
			print "synonym_count",synonym_count(chunk1,chunk2);
			print "get_wordnet_sent_similarity",get_wordnet_sent_similarity(chunk1,chunk2);
			print "getEmbeddingSimilarity",getEmbeddingSimilarity(chunk1, chunk2,embeddingmodel,embeddingtype);
	        #res=aggregage_wordnet_similarity(chunk1,chunk2);
        	if res<0.15:
                	res=-1;
	elif matchtype==5: #Hierarchical match 
		res=getEmbeddingSimilarity(chunk1, chunk2,embeddingmodel,embeddingtype);
		if res<=0:
			res=common_words(chunk1,chunk2);
		if res<=0:
			res=antonym_count(chunk1,chunk2);
		if res<=0:
			res=synonym_count(chunk1,chunk2);
		if res>1:
			res=1;
		if res<0.0004:
			res=-1;
	elif matchtype==6: #maximum
		stype="w2v"		 
		#cutoff=0.35
		chunk1=normalise(chunk1);
		chunk2=normalise(chunk2);
		res=getEmbeddingSimilarity(chunk1, chunk2,embeddingmodel,embeddingtype,cutoff);
		stype="w2v embeddings";
		res1=getEmbeddingSimilarity(chunk1.lower(), chunk2.lower(),embeddingmodel,embeddingtype,cutoff)
		if res1>res: 
			res=res1;
			stype="lowercase w2v embedding";
		res1=common_words(chunk1,chunk2);
		if res1>res: 
			res=res1;
			stype="common_words";
		res1=antonym_count(chunk1,chunk2);
		if res1>res: 
			res=res1;
			stype="antonyms";
		res1=synonym_count(chunk1,chunk2);
		if res1>res: 
			res=res1;
			stype="synnonym_count";
		res1=get_normalized_editscore_sentences(chunk1,chunk2,dictParams);
		if res1>res: 
			res=res1;
			stype="edit distance";

		#dep_feat = get_dependency_context_similarity(sent1,sent2,chunk1,chunk2)
		#res1=((2)*dep_feat[1]+(2)*dep_feat[2]+(2)*dep_feat[3]+(2)*dep_feat[4]+(2)*dep_feat[5]+(2)*dep_feat[6]+(2)*dep_feat[7])/14;
		#if res1>res: 
		#	res=res1;
		#	stype="Dependency based features";
	        
		#res1=getEmbeddingSimilarityMax(chunk1, chunk2,embeddingmodel,embeddingtype,cutoff);
		#if res1>res: 
		#	res=res1;
		#	stype="embedding_max";
		
		if vectorizer != None:
			res1=bigramSimilarity(vectorizer,chunk1,chunk2);	
			if res1>res: 
				res=res1;
				stype="bigram similarity";
		else:
			print "None Vectorizer";
		if res>1:
			res=1;
		if res<cutoff:
			res=-1;
			stype="< Cutoff: No similarity worked";
		if debug==1:
			print "returning matchtype ",stype," ",res
	elif matchtype==7: # with sultan features - headlines
		#sent1=""
		#sent2=""
		dep_feat = get_dependency_context_similarity(sent1,sent2,chunk1,chunk2)
		res=((0.8)*common_words(chunk1,chunk2)
			-(0.17)*antonym_count(chunk1,chunk2)
			+(0.52)*synonym_count(chunk1,chunk2)
			+(-1.17)*get_wordnet_sent_similarity(chunk1,chunk2)
			+(-2.44)*aggregage_wordnet_similarity(chunk1,chunk2)
			+(0)*getEmbeddingSimilarity(chunk1, chunk2,embeddingmodel,embeddingtype)
			+(-2)*dep_feat[0]
			+(2)*dep_feat[1]
			+(2)*dep_feat[2]
			+(2)*dep_feat[3]
			+(2)*dep_feat[4]
			+(2)*dep_feat[5]
			+(2)*dep_feat[6]
			+(2)*dep_feat[7]
			)/14.0;
	elif matchtype==8: # with sultan features - images
		#sent1=""
		#sent2=""
		dep_feat = get_dependency_context_similarity(sent1,sent2,chunk1,chunk2)
		res=((2.6)*common_words(chunk1,chunk2)
			-(5.2)*antonym_count(chunk1,chunk2)
			+(1.66)*synonym_count(chunk1,chunk2)
			+(-3.8)*get_wordnet_sent_similarity(chunk1,chunk2)
			+(-7.81)*aggregage_wordnet_similarity(chunk1,chunk2)
			+(0)*getEmbeddingSimilarity(chunk1, chunk2,embeddingmodel,embeddingtype)
			+(-4.9)*dep_feat[0]
			+(4.9)*dep_feat[1]
			+(4.9)*dep_feat[2]
			+(4.9)*dep_feat[3]
			+(4.9)*dep_feat[4]
			+(4.9)*dep_feat[5]
			+(4.9)*dep_feat[6]
			+(4.9)*dep_feat[7]
			)/14.0;
	elif matchtype==9: # with sultan features - average of images and headlines
		#sent1=""
		#sent2=""
		dep_feat = get_dependency_context_similarity(sent1,sent2,chunk1,chunk2)
		res=((1.7)*common_words(chunk1,chunk2)
			-(0.33)*antonym_count(chunk1,chunk2)
			+(1.1)*synonym_count(chunk1,chunk2)
			+(-2.6)*get_wordnet_sent_similarity(chunk1,chunk2)
			+(-5)*aggregage_wordnet_similarity(chunk1,chunk2)
			+(0)*getEmbeddingSimilarity(chunk1, chunk2,embeddingmodel,embeddingtype)
			+(-3.5)*dep_feat[0]
			+(3.5)*dep_feat[1]
			+(3.5)*dep_feat[2]
			+(3.5)*dep_feat[3]
			+(3.5)*dep_feat[4]
			+(3.5)*dep_feat[5]
			+(3.5)*dep_feat[6]
			+(3.5)*dep_feat[7]
			)/14.0;

	elif matchtype==13: #embedding
	        res=getEmbeddingSimilarityMax(chunk1, chunk2,embeddingmodel,embeddingtype);
	else:
		res=-1;
        return res;

def test():
        embeddingtype=2;
	matchtype=6;
	debug=0;
	ph1="London bridge is falling down"
	ph2="A bridge in london is collapsing"
	sent1="London bridge is falling down"
	sent2="A bridge in london is collapsing"
        #ph1="eastern Afghan explosion";
        #ph2="At least Chinese coal mine explosion";
	print ph1;
	print ph2;
        model=getEmbeddingModel(embeddingtype,"","");
        res=getAggregatedSimilarity(ph1,ph2,sent1,sent2,model,embeddingtype,matchtype,debug);
        print "aggregated similarity is ",res," with match type ",matchtype;

#test();

