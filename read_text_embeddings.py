import numpy;
from django.utils import encoding;
from subprocess import call;
import os;

def read_text_embeddings(filename):
	dic={};
	with open(filename) as f:
	    lines = f.readlines()

	for line in lines:
		parts=line.split(',',1);
		key=parts[0].strip().lower();
		
		value=numpy.fromstring(parts[1], dtype=float, sep=",");
		dic[key]=value;
	return (dic);

#d=read_text_embeddings("test.txt");
#print d['hello folks']

def convert_unicode_to_string(x):
    """
    >>> convert_unicode_to_string(u'ni\xf1era')
    'niera'
    """
    return encoding.smart_str(x, encoding='ascii', errors='ignore')


def writechunks(chunks1,chunks2,fileout):
	f = open(fileout,'w')
	for chunks in chunks1: 
		for chunk in chunks:
			f.write(chunk+"\n");
	for chunks in chunks2: 
		for chunk in chunks:
			f.write(chunk+"\n");
	
	f.close() # you can omit


def readchunkInput(inputfile1,inputfile2):
	chunks1=[];
	chunks2=[];
	#Labels for each chunked sentence
	chunk1labels=[]
	chunk2labels=[]
	sent1=[]
	sent2=[]
	breakiter=50000;

	i=0;
	for line in open(inputfile1,'r').readlines():
		#chunks1=chunks1.append([])
	        line=convert_unicode_to_string(line);		
		line=line.replace(']','[');
		spl=line.strip(' \n').split('[');
		spl= filter(lambda name: name.strip(), spl)
		chunks1.append(spl);
		sent1.append(''.join(chunks1[i]));
		#Some code to get the index number strig for each chunk as shown in the output
		j=1;
		labels=[];
		for chunk in spl:
			chlab="";
			for word in chunk.split():
				chlab=chlab+' '+str(j);	
				j=j+1;
			labels.append(chlab);
		chunk1labels.append(labels);
		#print chunks1[i];
		i=i+1;
		if breakiter>0 and i>breakiter:
			break;

	i=0;
	for line in open(inputfile2,'r').readlines():
		#chunks2.append([])	
	        line=convert_unicode_to_string(line);		
		line=line.replace(']','[');
		spl=line.strip(' \n').split('[');
		spl= filter(lambda name: name.strip(), spl)
		chunks2.append(spl);
		sent2.append(''.join(chunks2[i]));
		#Some code to get the index number strig for each chunk as shown in the output
		j=1;
		labels=[];
		for chunk in spl:
			chlab="";
			for word in chunk.split():
				chlab=chlab+' '+str(j);	
				j=j+1;
			labels.append(chlab);
		chunk2labels.append(labels);
		#print chunks2[i];
		i=i+1;
		if breakiter>0 and i>breakiter:
			break;
	return(chunks1,chunks2,sent1,sent2,chunk1labels,chunk2labels);

def get_phrase_text_embeddings(chunkfile1,chunkfile2):
	outfilechunks=chunkfile1+".chunks";
	outfileembeddings=chunkfile1+".embeddings";
	(chunks1,chunks2,sent1,sent2,chunk1labels,chunk2labels)=readchunkInput(chunkfile1,chunkfile2);
	writechunks(chunks1,chunks2,outfilechunks);
	progdir="/home/lavanya/res/semeval/task-b-semeval2016/resources/codeRAEVectorsNIPS2011/";
	currdir=os.getcwd()+"/";
	print currdir;
	print progdir;
	call(["cp",outfilechunks,progdir+"input.txt"]);
	os.chdir(os.path.dirname(progdir))
	call(["./phrase2Vector.sh"]);
	os.chdir(os.path.dirname(currdir))
	emf=open(progdir+"outVectors.txt");
	cf=open(progdir+"input.txt");
	finalf=open(outfileembeddings,"w");
	lineso=emf.readlines();
	linesc=cf.readlines();
	
	count=0;
	for line in lineso:	
		finalf.write(linesc[count].rstrip()+","+lineso[count]);
		count=count+1;
	#call(["cp",progdir+"outVectors.txt",outfileembeddings]);
	print "wrote output embeddings of ",outfilechunks," to ",outfileembeddings
	return (outfileembeddings);
	

def test_main():	 
	chunkfile1="STSint.input.images.sent1.chunk.txt";
	chunkfile2="STSint.input.images.sent2.chunk.txt";
	outfile=get_phrase_text_embeddings(chunkfile1,chunkfile2);	
	model=read_text_embeddings(outfile);
	print len(model['a cat']);
	

#test_main();

