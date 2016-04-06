"""
The Simplified Whiskas Model Python Formulation for the PuLP Modeller
Authors: Antony Phillips, Dr Stuart Mitchell  2007
"""
# Import PuLP modeler functions
from pulp import *
import numpy;
import gensim;
import re
sys.path.insert(0, '../../../resources/monolingual-word-aligner')
from aligner import *
import math
import sys
from scipy import spatial;
#from embedding_funcs import *;
from compute_similarity import *;
from nltk.corpus import stopwords;
import copy;
from paramsHeader import *;

import warnings
warnings.filterwarnings("ignore")

def coverunderscore(string):
	return "_"+string+"_";

def getIndexAndPhrases(sset1,sset2,chunks1,chunks2):
	indstr="";
	count=0;
	ch1="";
	ch2="";
	for ele in sset1:
		ch1=ch1+chunks1[ele]+" ";
		if count==0:
			indstr=coverunderscore(str(ele));
			count=count+1;
			continue;
		indstr=indstr+"_"+coverunderscore(str(ele));
		count=count+1;
	indstr=indstr+"_N";
	for ele in sset2:
		indstr=indstr+"_"+coverunderscore(str(ele));
		ch2=ch2+chunks2[ele]+" ";
	ch1=ch1.strip();
	ch2=ch2.strip();
	return (indstr,ch1,ch2);



def recurseComb(num,M,arr,sets,ind):
	if ind==num:
		sets.append(list(arr));
		#print sets;
		return (sets);
	start=0;
	if ind==0:
		start=0;
	else: 
		start=arr[ind-1]+1;
	for i in range(start,M):
		arr[ind]=i;
		sets=recurseComb(num,M,arr,sets,ind+1);
	return(sets);
		

def getAdjacentSets(size,M,sets,maxsize):
	print(sets);	
	for i in range(0,M-size):
		setsingle=range(i,i+size);
		for j in range(size,maxsize):
			setsingle.append(M);
		sets.append(setsingle);
	return(sets);

def getCandidateSetsForAlignment(chunks1,chunks2,maxsize):
	M=len(chunks1)
	N=len(chunks2)
	
	sets1=[];	
	sets2=[];	


	if maxsize>=1:	
		sets1=recurseComb(1,M,[M]*maxsize,sets1,0);
		sets2=recurseComb(1,N,[N]*maxsize,sets2,0);
	if maxsize>=2:	
		sets1=recurseComb(2,M,[M]*maxsize,sets1,0);
		sets2=recurseComb(2,N,[N]*maxsize,sets2,0);
	if maxsize>=3:
		for i in range(3,maxsize+1):
			sets1=getAdjacentSets(i,M,sets1,maxsize);
		for i in range(3,maxsize+1):
			sets2=getAdjacentSets(i,N,sets2,maxsize);

	#for i in range(1,maxsize+1):
		#sets1=recurseComb(i,M,[M]*maxsize,sets1,0);
	#for i in range(1,maxsize+1):
	#	sets2=recurseComb(i,N,[N]*maxsize,sets2,0);
	
	#print "sets1",sets1;
	#print "sets2",sets2;
	return (sets1, sets2);

def getAlignWeightInd(sset1,sset2,M,N,alignweights):
	s1=sum(sset1!=M);
	s2=sum(sset2!=N);
	#print(1.0*(s1+s2));
	return (1.0*(s1+s2));


	s1=sum(sset1==M)-1;
	s2=sum(sset2==N)-1;
	if(s1<s2):
		return (alignweights[s1]);
	else:
		return (alignweights[s2]);


def manyILPAlign(chunks1In, chunks2In,sent1,sent2, model,matchtype,embeddingtype,vectorizer=None,cutoff=0.35,alignsize=2,alignweights=[1.3,1],dictParams=dictHeadlines):
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
	# Create the 'prob' variable to contain the problem data
	prob = LpProblem("The alignment Problem",LpMinimize)

	z_ind=[];
	scores={};
	[sets1, sets2]=getCandidateSetsForAlignment(chunks1,chunks2,alignsize);
	chunks1.append("");
	chunks2.append("");
	print "sets1",sets1
	print "sets2",sets2
	for sset1 in sets1:
		for sset2 in sets2:
			#construct index:
			[indstr,ch1,ch2]=getIndexAndPhrases(sset1,sset2,chunks1,chunks2);
			z_ind.append(indstr);
			
			scores[indstr]=getAlignWeightInd(sset1,sset2,M,N,alignweights)*getAggregatedSimilarity(ch1,ch2,sent1,sent2,model,embeddingtype,matchtype,0,vectorizer,cutoff,dictParams=dictParams);
			
			#j1=0;l1=0;
			#i1=1;
			#if j<M:	j1=0.5;
			#k1=1;
			#if l<N:	l1=0.5;
			#weight=i1+j1+k1+l1;
			#if j==M or l==N:
#			#	scores[str(i)+'_'+str(j)+'_'+str(k)+'_'+str(l)]=scores[str(i)+'_'+str(j)+'_'+str(k)+'_'+str(l)]*1.2;
			#	scores[str(i)+'_'+str(j)+'_'+str(k)+'_'+str(l)]=scores[str(i)+'_'+str(j)+'_'+str(k)+'_'+str(l)]*single_clamp_factor;
			#	#scores[str(i)+'_'+str(j)+'_'+str(k)+'_'+str(l)]=weight*scores[str(i)+'_'+str(j)+'_'+str(k)+'_'+str(l)];

	#print scores
	z_vars = LpVariable.dicts("z_vars",z_ind,0,1)
	# The objective function is added to 'prob' first
	prob += lpSum([-1*scores[ind]*z_vars[ind] for ind in z_vars]), 'objective function to minimize'

	# The constraints are entered
	for i in range(0,M):
		ikeys=[key for key in z_ind if re.search('_'+str(i)+'_', key.split("N")[0])]
		#ikeys1=[key for key in z_ind if re.match(str(i)+'_[0-9]+_[0-9]+_[0-9]+', key)]
		#ikeys2=[key for key in z_ind if re.match('[0-9]+_'+str(i)+'_[0-9]+_[0-9]+', key)]
		prob+=lpSum([z_vars[key] for key in ikeys])  <= 1, "matchfirst"+str(i)
	for k in range(0,N):
		ikeys=[key for key in z_ind if re.search('_'+str(k)+'_', key.split("N")[1])]
		#ikeys1=[key for key in z_ind if re.match('[0-9]+_[0-9]+_'+str(k)+'_[0-9]+', key)]
		#ikeys2=[key for key in z_ind if re.match('[0-9]+_[0-9]+_[0-9]+_'+str(k), key)]
		prob+=lpSum([z_vars[key] for key in ikeys])  <= 1, "matchsecond"+str(k)

	# The problem data is written to an .lp file
	LpSolverDefault.msg = 1
	prob.writeLP("manyAlignModel.lp")

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
	totalignscore=0;

	print M
	print N
	numalign=0;
	for v in prob.variables():
		if v.varValue==1:
			print "vname",v.name
			p1=v.name.split("N")[0].split("s")[1];
			p2=v.name.split("N")[1];
			s1=p1.split("_");
			s1 = [x for x in s1 if x != '']
			s2=p2.split("_");
			s2 = [x for x in s2 if x != '']
			s1 = [int(i) for i in s1]
			s2 = [int(i) for i in s2]
			[indstr,ch1,ch2]=getIndexAndPhrases(s1,s2,chunks1,chunks2);
	
			numalign=numalign+1;
			totalign=totalign+2;

			den=sum(s1!=M)+sum(s2!=N);
			alignscore.append(scores[indstr]/den);
			totalignscore=totalignscore+(scores[indstr]/den);
			str1="";str2="";
			nonempty=0;
			for ind in s1:
				if ind!=M: 
					boolaligned1[ind]=1;
					str1=str1+str(ind)+" ";
					nonempty=1;
			tmpalign1.append(str1);
			if nonempty==1:
				totalign=totalign+1;
			nonempty=0;
			for ind in s2:
				if ind!=N:
					boolaligned2[ind]=1;
					str2=str2+str(ind)+" ";
					#tmpalign2.append(str(ind)+" ");
			tmpalign2.append(str2);
			if nonempty==1:
				totalign=totalign+1;
			aligntype.append("EQUI");
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
		#totscore=totalignscore/(M+N);
		#totscore=value(prob.objective)*-1/numalign;
		totscore=1.0*totalign/(M+N);
	#print chunks1
	#print chunks2
	#print tmpalign1
	#print tmpalign2
		
	return (tmpalign1, tmpalign2, aligntype,alignscore,totscore)


def main_manyali():
	embeddingtype=2;
	matchtype=1;
        chunksfile1="STSint.input.images.sent1.chunk.txt";
        chunksfile2="STSint.input.images.sent2.chunk.txt";

	model=getEmbeddingModel(embeddingtype);
	#chunks1=['A smiling woman','with a beer','sitting','outside','with another smiling woman'];
	chunks1=['Smiling women','with a brick','sitting','outside','as a pair'];
	chunks2=['Two women','sitting','outside','laughing'];
	sent1='Smiling women with a brick sitting outside as a pair';
	sent2='Two women sitting outside laughing';

	#chunks1=[ 'Iran' , 'says' , 'it', 'captures', 'drone', ';','U.S.', 'denies','losing','one' ];
	#chunks2=[ 'Iran' ,'says','it','has seized','U.S. drone',';','U.S.', 'says','it','\'s not true' ]

	#['0 ', '1 ', '2 ', '4 ', '5 ', '6 ', '7 ', '3', '8', '9', '-1']
	#['0 ', '7 ', '8 ', '4 ', '5 ', '6 ', '1 ', '3 ', '9 ', '-1', '-1', '-1', '2']
	#['EQUI', 'EQUI', 'EQUI', 'EQUI', 'EQUI', 'EQUI', 'EQUI', 'NOALI', 'NOALI', 'NOALI', 'NOALI']
	#[-1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0]


	#chunks1=[ 'A group', 'of people', ' sitting ', ' around a table ', ' with food ' ,' on it ', ' . '];
	#chunks2=[ 'A group ',' of people ',' sit ', ' around a table ', ' with food and beer ' ,' . '];
	#chunks1=[' Shinzo Abe ', 'removed', 'Japan', "s Prime Minister"]
	#chunks2=[' Shinzo Abe ',  'Japan', "s prime minister"]
	alignsize=3;	
	alignweight=[1.35,1.1,1];

	(tmpalign1, tmpalign2, aligntype,alignscore,totscore)=manyILPAlign(chunks1, chunks2, sent1,sent2,model,matchtype,embeddingtype,None,0.35,alignsize,alignweight);
	print tmpalign1 
	print tmpalign2
	print aligntype
	print alignscore
	print totscore


main_manyali();
