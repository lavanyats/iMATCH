#!/usr/bin/python

#EditClamp: Is the cutoff belowwhich editscore is not considered
#TrigramChunks : Some rules to (1) if neighbors align perfectly and inbetween chunks 'sort-of' align, match inbetween chunks by setting score to 1 (2) Let chunks with similar indices have slightly higher probability of aligning rather than far away indices in case of multiple perfect matches of chunk from a sentence with chunks in the other sentence
#NoCombineProp : 
dictHeadlines = {'EditClamp': 0.7 , 'TrigramChunks': 1, 'NoCombineProp':0};
dictImages = {'EditClamp': 0.7 , 'TrigramChunks': 1, 'NoCombineProp':0};
#dictImages = {'EditClamp': 0.7 , 'TrigramChunks': 0};
dictStudents = {'EditClamp': 0.5 , 'TrigramChunks': 1, 'NoCombineProp':0};

