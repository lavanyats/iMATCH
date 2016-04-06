"""
The Simplified Whiskas Model Python Formulation for the PuLP Modeller
Authors: Antony Phillips, Dr Stuart Mitchell  2007
"""
# Import PuLP modeler functions
from pulp import *
import numpy;
import gensim;
import re
import math
import sys
sys.path.insert(0, '../../../resources/monolingual-word-aligner')
from aligner import *
from scipy import spatial;
#from embedding_funcs import *;
from compute_similarity import *;
import copy;
from paramsHeader import *;
from wordnet_utils import *;
from nltk.corpus import stopwords;

def multipleILPAlign(chunks1In, chunks2In,sent1,sent2, model,matchtype,embeddingtype,vectorizer=None,cutoff=0.35,clamp_param=1.5,dictParams=dictHeadlines):
	chunks1=copy.deepcopy(chunks1In);
	chunks2=copy.deepcopy(chunks2In);
	M=len(chunks1)
	N=len(chunks2)
	print M
	print N
	#for i in range(1,M):
	#	chunks1[i]=chunks1[i].lower();
	#for j in range(1,N):
	#	chunks2[j]=chunks2[j].lower();

	removestopwords=1;
	if removestopwords:
		stop = set(stopwords.words('english'));
		for i in range(1,M):
			oldchunk=chunks1[i];
			chunks1[i]=' '.join([w for w in chunks1[i].split() if w not in stop]);
			if len(chunks1[i])==0 or embeddingtype==4:
				chunks1[i]=oldchunk;
		for j in range(1,N):
			oldchunk=chunks2[j];
			chunks2[j]=' '.join([w for w in chunks2[j].split() if w not in stop]);
			if len(chunks2[j])==0 or embeddingtype==4:
				chunks2[j]=oldchunk;
	print chunks1
	print chunks2
	chunks1.append("");
	chunks2.append("");
	# Create the 'prob' variable to contain the problem data
	prob = LpProblem("The alignment Problem",LpMinimize)
	single_clamp_factor=clamp_param; #headlines w2v
	print "single clamp factor: ",single_clamp_factor,"cutoff= ",cutoff;

	# The 2 variables Beef and Chicken are created with a lower limit of zero
	z_ind=[];
	scores={};
	for i in range(0,M):
		for j in range(i+1,M+1):
			for k in range(0,N):
				for l in range(k+1,N+1):		
					j1=0;l1=0;
					i1=1;
					if j<M:	j1=0.5;
					k1=1;
					if l<N:	l1=0.5;
					weight=i1+j1+k1+l1;
					z_ind.append(str(i)+'_'+str(j)+'_'+str(k)+'_'+str(l))
					#if matchtype==1:
					if matchtype!=3:
						scores[str(i)+'_'+str(j)+'_'+str(k)+'_'+str(l)]=getAggregatedSimilarity(chunks1[i]+" "+chunks1[j],chunks2[k]+" "+chunks2[l],sent1,sent2,model,embeddingtype,matchtype,0,vectorizer,cutoff,dictParams=dictParams);

						if dictParams['TrigramChunks']==1:
							if j==M and l==N and M>2 and N>2: #single-single case : Take context features: trigram chunks insted of 1-1 
								previ=(chunks1[i+2] if i==0 else chunks1[i-1]);
								prevk=(chunks2[k+2] if k==0 else chunks2[k-1]);
								sc1=getEmbeddingSimilarity(previ+" "+chunks1[i+1],prevk+" "+chunks2[k+1],model,embeddingtype);
								#sc1=getEmbeddingSimilarity(previ+" "+chunks1[i]+" "+chunks1[i+1],prevk+" "+chunks2[k]+" "+chunks2[k+1],model,embeddingtype);
								#sc1=getAggregatedSimilarity(previ+" "+chunks1[i]+" "+chunks1[i+1],prevk+" "+chunks2[k]+" "+chunks2[k+1],sent1,sent2,model,embeddingtype,matchtype,0,vectorizer,cutoff);
								#if sc1>0.9:
								if scores[str(i)+'_'+str(j)+'_'+str(k)+'_'+str(l)]>0.6 and sc1>0.9:
								#if sc1>0.8:
									scores[str(i)+'_'+str(j)+'_'+str(k)+'_'+str(l)]=1;
									#scores[str(i)+'_'+str(j)+'_'+str(k)+'_'+str(l)]=sc1;
								if scores[str(i)+'_'+str(j)+'_'+str(k)+'_'+str(l)]==1:
									scores[str(i)+'_'+str(j)+'_'+str(k)+'_'+str(l)]=scores[str(i)+'_'+str(j)+'_'+str(k)+'_'+str(l)]+0.05/(1+abs(i-k));
						if dictParams['NoCombineProp']==1:
							if j!=M:
								words=filter(None,chunks1[i].strip().split());
								pt=get_pos_tags(words[0])[0][1];
								if len(words)==1 and (pt=='IN' or pt=='DT'):
									scores[str(i)+'_'+str(j)+'_'+str(k)+'_'+str(l)]=-1;
							if l!=N:
								words=filter(None,chunks2[k].strip().split());
								pt=get_pos_tags(words[0])[0][1];
								if len(words)==1 and (pt=='IN' or pt=='DT'):
									scores[str(i)+'_'+str(j)+'_'+str(k)+'_'+str(l)]=-1;

						if j==M or l==N:
#							scores[str(i)+'_'+str(j)+'_'+str(k)+'_'+str(l)]=scores[str(i)+'_'+str(j)+'_'+str(k)+'_'+str(l)]*1.2;
							scores[str(i)+'_'+str(j)+'_'+str(k)+'_'+str(l)]=scores[str(i)+'_'+str(j)+'_'+str(k)+'_'+str(l)]*single_clamp_factor;
					else:
						scores[str(i)+'_'+str(j)+'_'+str(k)+'_'+str(l)]=-1;
	#print scores
	z_vars = LpVariable.dicts("z_vars",z_ind,0,1)
	# The objective function is added to 'prob' first
	prob += lpSum([-1*scores[ind]*z_vars[ind] for ind in z_vars]), 'objective function to minimize'

	# The constraints are entered
	for i in range(0,M):
		ikeys1=[key for key in z_ind if re.match(str(i)+'_[0-9]+_[0-9]+_[0-9]+', key)]
		ikeys2=[key for key in z_ind if re.match('[0-9]+_'+str(i)+'_[0-9]+_[0-9]+', key)]
		prob+=lpSum([z_vars[key] for key in ikeys1+ikeys2])  <= 1, "singlematchi"+str(i)
	for k in range(0,N):
		ikeys1=[key for key in z_ind if re.match('[0-9]+_[0-9]+_'+str(k)+'_[0-9]+', key)]
		ikeys2=[key for key in z_ind if re.match('[0-9]+_[0-9]+_[0-9]+_'+str(k), key)]
		prob+=lpSum([z_vars[key] for key in ikeys1+ikeys2])  <= 1, "singlematchk"+str(k)

	# The problem data is written to an .lp file
	LpSolverDefault.msg = 1
	prob.writeLP("multipleAlignModel.lp")

	# The problem is solved using PuLP's choice of Solver
	prob.solve()

	# The status of the solution is printed to the screen
	print("Status:", LpStatus[prob.status])

	# Each of the variables is printed with it's resolved optimum value
	#for v in prob.variables():
	#    print(v.name, "=", v.varValue)
    
	# The optimised objective function value is printed to the screen
	print("Total Objective value = ", value(prob.objective))

	boolaligned1=[-1]*M;
	boolaligned2=[-1]*N;
	tmpalign1=[];
	tmpalign2=[];
	aligntype=[];
	alignscore=[];
	totalign=0;

	for v in prob.variables():
		#print v.name
		i=int(v.name.split("_")[2]);
		j=int(v.name.split("_")[3]);
		k=int(v.name.split("_")[4]);
		l=int(v.name.split("_")[5]);
		j1=0;l1=0;
		i1=1;
		if j<M:	j1=0.5;
		k1=1;
		if l<N:	l1=0.5;
		weight=i1+j1+k1+l1;
		#print(v.name, "=", v.varValue, " sc= ",scores[str(i)+'_'+str(j)+'_'+str(k)+'_'+str(l)]," mult= ",weight, " i1j1k1l1 ",i1,j1,k1,l1, " MN ",M,N, " ijkl ",i,j,k,l)
	
	numalign=0;
	for v in prob.variables():
		if v.varValue==1:
			numalign=numalign+1;
			totalign=totalign+2;
			i=v.name.split("_")[2];
			j=v.name.split("_")[3];
			k=v.name.split("_")[4];
			l=v.name.split("_")[5];
			print chunks1[int(i)]+" "+chunks1[int(j)]
			print chunks2[int(k)]+" "+chunks2[int(l)]
			alignscore.append(scores[str(i)+'_'+str(j)+'_'+str(k)+'_'+str(l)]);
			if j==str(M):
				j="";
			else: 
				boolaligned1[int(j)]=1;
				totalign=totalign+1;
			if l==str(N):
				l="";
			else:
				boolaligned2[int(l)]=1;
				totalign=totalign+1;
			tmpalign1.append(i+" "+j);
			tmpalign2.append(k+" "+l);
			aligntype.append("EQUI");
			boolaligned1[int(i)]=1;
			boolaligned2[int(k)]=1;
	for i in range(0,M):
		if boolaligned1[i]==-1:
			tmpalign1.append(str(i));
			tmpalign2.append(str(-1));
			alignscore.append(0);
			aligntype.append("NOALI");
	for j in range(0,N):
		if boolaligned2[j]==-1:
			tmpalign1.append(str(-1));
			tmpalign2.append(str(j));
			alignscore.append(0);
			aligntype.append("NOALI");

	totscore=0;
	if numalign>0:
		totscore=value(prob.objective)*-1/numalign;
		#totscore=1.0*totalign/(M+N);
	#print chunks1
	#print chunks2
	#print tmpalign1
	#print tmpalign2
		
	return (tmpalign1, tmpalign2, aligntype,alignscore,totscore)


def main_multiali():
	embeddingtype=2;
	matchtype=1;
        chunksfile1="STSint.input.images.sent1.chunk.txt";
        chunksfile2="STSint.input.images.sent2.chunk.txt";

	model=getEmbeddingModel(embeddingtype);
	chunks1=['Smiling women','with a brick','sitting','outside','as a pair'];
	#chunks1=['A smiling woman','with a beer','sitting','outside','with another smiling woman'];
	chunks2=['Two women','sitting','outside','laughing'];
	sent1='Smiling women with a brick sitting outside as a pair';
	sent2='Two women sitting outside laughing';

	#chunks1=[ 'Iran' , 'says' , 'it', 'captures', 'drone', ';','U.S.', 'denies','losing','one' ];
	#chunks2=[ 'Iran' ,'says','it','has seized','U.S. drone',';','U.S.', 'says','it','\'s not true' ]


	#chunks1=[ 'A group', 'of people', ' sitting ', ' around a table ', ' with food ' ,' on it ', ' . '];
	#chunks2=[ 'A group ',' of people ',' sit ', ' around a table ', ' with food and beer ' ,' . '];
	#chunks1=[' Shinzo Abe ', 'removed', 'Japan', "s Prime Minister"]
	#chunks2=[' Shinzo Abe ',  'Japan', "s prime minister"]
	
	(tmpalign1, tmpalign2, aligntype,alignscore,totscore)=multipleILPAlign(chunks1, chunks2, sent1,sent2,model,matchtype,embeddingtype);
	print tmpalign1 
	print tmpalign2
	print aligntype
	print alignscore
	print totscore


main_multiali();
