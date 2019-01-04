#!/usr/bin/env python

"""Clean comment text for easier parsing."""
from __future__ import print_function

import re
import string
import argparse
import io
import json
import bz2




__author__ = ""
__email__ = ""

# Some useful data.
_CONTRACTIONS = {
    "tis": "'tis",
    "aint": "ain't",
    "amnt": "amn't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hell": "he'll",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "id": "i'd",
    "ill": "i'll",
    "im": "i'm",
    "ive": "i've",
    "isnt": "isn't",
    "itd": "it'd",
    "itll": "it'll",
    "its": "it's",
    "mightnt": "mightn't",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "oclock": "o'clock",
    "ol": "'ol",
    "oughtnt": "oughtn't",
    "shant": "shan't",
    "shed": "she'd",
    "shell": "she'll",
    "shes": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "somebodys": "somebody's",
    "someones": "someone's",
    "somethings": "something's",
    "thatll": "that'll",
    "thats": "that's",
    "thatd": "that'd",
    "thered": "there'd",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "wasnt": "wasn't",
    "wed": "we'd",
    "wedve": "wed've",
    "well": "we'll",
    "were": "we're",
    "weve": "we've",
    "werent": "weren't",
    "whatd": "what'd",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whereve": "where've",
    "whod": "who'd",
    "whodve": "whod've",
    "wholl": "who'll",
    "whore": "who're",
    "whos": "who's",
    "whove": "who've",
    "whyd": "why'd",
    "whyre": "why're",
    "whys": "why's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "yall": "y'all",
    "youd": "you'd",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've"
}

# You may need to write regular expressions.


def sanitize(text):

    """Do parse the text in variable "text" according to the spec, and return
    a LIST containing FOUR strings 
    1. The parsed text: 
    2. The unigrams: 
    3. The bigrams
    4. The trigrams
    """

    # YOUR CODE GOES BELOW:
 
    # Replace new lines and tab characters with a single space
    parsed_text = re.sub(r'[\r\n\t]', ' ', text)
    parsed_text = re.sub(r'(\.){2,}', ' . ', parsed_text)
    parsed_text = re.sub(r'(\!){2,}', ' ! ', parsed_text)
    parsed_text = re.sub(r'(\?){2,}', ' ? ', parsed_text)
    parsed_text = re.sub(r'(\,){2,}', ' , ', parsed_text)
    # parsed_text = re.sub(r'\;', ' ; ', parsed_text)
    
    # Remove URLs.Replace them with the empty string.
    parsed_text = re.sub(r'(\()?((https?:\/\/)|(www\.))\S*(\.[A-Za-z]{2,5})?(\))?', ' ', parsed_text)

    # Split text on a single space.
    parsed_text = parsed_text.strip();
    parsed_text = re.sub(r' +', ' ', parsed_text)

    # Separate all external punctuation such as periods, commas, etc. 
    h = re.split(r"([\.,:!?;](?= |$))", parsed_text)
    parsed_text =(' '.join(h))

    # Remove all punctuation: delete all non-alphanumerics except . ! ? , ; : " '
    
    parsed_text = re.sub(r"([^\.\,\:\!\?\;\'\w\$\%]+(?!\w))|((?<!\w)[^\.\,\:\!\?\;\'\w\$\%]+)|(\$(?!\d))|((?<!\d)\%)", ' ', parsed_text)
    # parsed_text = re.sub(r'\/', ' ', parsed_text)
    parsed_text = re.sub(r'\(', ' ', parsed_text)
    # Split text on a single space.
    parsed_text = parsed_text.strip();
    parsed_text = re.sub(r' +', ' ', parsed_text)
    
    # Convert all text to lowercase.
    parsed_text = parsed_text.lower()
#     seperate words/puctuations from parsed_text
    words = parsed_text.split()
    # print ('parsed_text: '+ parsed_text)
    
    unigrams = re.sub(r' [.!?,;:] ', ' ', parsed_text)
    unigrams = re.sub(r' [.!?,;:]$', '', unigrams)
    # print ('unigrams: '+ unigrams)

#     split short sentences from words based on puctuation segment
    small = list()
    big = list()  
    for j in words:
        if j not in [',','.','?','!',':',';']:
            small.append(j)
            flag =1
        else:
            big.append(small)
            small = list()
            flag=0
    if len(big)==0 or flag==1:
        big.append(small)
        
    sentence=list()
    for short in big:
        n=list()
        for i in range(0,len(short)-1):
            temp = short[i]+"_"+short[i+1]
            n.append(temp)
        sentence.append(n)    
        
    bigram=list()
    for i in range(0,len(sentence)):
        for w in sentence[i]:
            bigram.append(w)
        
    r3 = (' '.join(bigram))
    # print ('bigrams: '+ r3)
    bigrams = r3
    
    sentence2=list()
    for short in big:
        n2=list()
        for i in range(0,len(short)-2):
            temp = short[i]+"_"+short[i+1]+"_"+short[i+2]
            n2.append(temp)
        sentence2.append(n2)    

    trigram=list()
    for i in range(0,len(sentence2)):
        for w in sentence2[i]:
            trigram.append(w)

    r4 = (' '.join(trigram))
    # print ('trigrams: '+ r4)
    trigrams = r4
    cleand_text= unigrams+ " " +bigrams+" " +trigrams

    return cleand_text




if __name__ == "__main__":
    # This is the Python main function.
    # You should be able to run
    # python cleantext.py <filename>
    # and this "main" function will open the file,
    # read it line by line, extract the proper value from the JSON,
    # pass to "sanitize" and print the result as a list.

    # YOUR CODE GOES BELOW.


    #  read json
    value=list()
    with io.open('sample-comments.json',  encoding='utf-8') as f:
        for line in f:        
            l=json.loads(line)
            comment=l['body']
            value.append(comment)
    f.close()

    # result=list()
  
    for i in value:
       print (str(sanitize(i)))

    # read bz2
    # value=list()
    # source_file = bz2.BZ2File("comments-minimal.json.bz2", "r")
    # count = 0 
    # for line in source_file:
    #     count += 1 
    #     if count <= 3:         
    #         l=json.loads(line)       
    #         value.append(l['body'])
    #     else:
    #         break
    # source_file.close()     
    # result=list()
    # for i in value:
    #     result.append(sanitize(i))



