## Text Preprocessing for Amharic

When working with NLP, preprocessing text is one of important process to get clean and formatted data before passing it to the model.
Most of resourced languages suach as English and other European countries has tools such as NLTK, that allows to perform text preprocessing , but the same history is not true for Amharic. Amharic is an offical language of the Ethiopian government spoken by more than 100M people arroung the world and all over Ethiopia. Amharic script is not latin it uses geez script and this made the steps a bit challenging to use.

The aim of this notebook is to support researchers working in NLP tasks for Amharic. The following preprocessing steps are included:
* Short form expansion
* Multi-word detection
* Character level miss-match normalization
* Number mismatch normalization

### Short Form Expansion and Character Level Normalization 

 To deal multi-word short form representation, the list of short forms in Amharic language are consulted to expand a short form expression to its long form. For example, ትምህርት ቤት can also be represented as ት/ቤት in Amharic text.

``` python
import re
class normalize(object):
    expansion_file_dir='' # assume you have file with list of short forms with their expansion as gazeter
    short_form_dict={}
    # Constructor
    def __init__(self):
       self.short_form_dict=self.get_short_forms()
        

    def get_short_forms(self):
         text=open(self.expansion_file_dir,encoding='utf8')
         exp={}
         for line in iter(text):
            line=line.strip()
            if not line:  # line is blank
                continue
            else:
                expanded=line.split("-")
                exp[expanded[0].strip()]=expanded[1].replace(" ",'_').strip()
         return exp
                     

    # method that expand short form file
    def expand_short_form(self,input_short_word):
        if input_short_word in self.short_form_dict:              
            return self.short_form_dict[input_short_word]
        else:
            return input_short_word

```
The following function performs character level mismatch normalization task. 
Amharic has different characters that are interchangeably used in writing and reading such as (ሀ, ኀ, ሐ, and ኸ), (ሰ and ሠ), (ጸ and ፀ), (ው and ዉ) and (አ and ዓ). For example, ጸሀይ to mean sun can also be written as ፀሐይ. In addition, Amharic words with suffix such as ቷል are also written as ቱዋል. So, I will normalize any character under such category to common canonical representation.
``` python
def normalize_char_level_missmatch(input_token):
        rep1=re.sub('[ሃኅኃሐሓኻ]','ሀ',input_token)
        rep2=re.sub('[ሑኁዅ]','ሁ',rep1)
        rep3=re.sub('[ኂሒኺ]','ሂ',rep2)
        rep4=re.sub('[ኌሔዄ]','ሄ',rep3)
        rep5=re.sub('[ሕኅ]','ህ',rep4)
        rep6=re.sub('[ኆሖኾ]','ሆ',rep5)
        rep7=re.sub('[ሠ]','ሰ',rep6)
        rep8=re.sub('[ሡ]','ሱ',rep7)
        rep9=re.sub('[ሢ]','ሲ',rep8)
        rep10=re.sub('[ሣ]','ሳ',rep9)
        rep11=re.sub('[ሤ]','ሴ',rep10)
        rep12=re.sub('[ሥ]','ስ',rep11)
        rep13=re.sub('[ሦ]','ሶ',rep12)
        rep14=re.sub('[ዓኣዐ]','አ',rep13)
        rep15=re.sub('[ዑ]','ኡ',rep14)
        rep16=re.sub('[ዒ]','ኢ',rep15)
        rep17=re.sub('[ዔ]','ኤ',rep16)
        rep18=re.sub('[ዕ]','እ',rep17)
        rep19=re.sub('[ዖ]','ኦ',rep18)
        rep20=re.sub('[ጸ]','ፀ',rep19)
        rep21=re.sub('[ጹ]','ፁ',rep20)
        rep22=re.sub('[ጺ]','ፂ',rep21)
        rep23=re.sub('[ጻ]','ፃ',rep22)
        rep24=re.sub('[ጼ]','ፄ',rep23)
        rep25=re.sub('[ጽ]','ፅ',rep24)
        rep26=re.sub('[ጾ]','ፆ',rep25)
        #Normalizing words with Labialized Amharic characters such as በልቱዋል or  በልቱአል to  በልቷል  
        rep27=re.sub('(ሉ[ዋአ])','ሏ',rep26)
        rep28=re.sub('(ሙ[ዋአ])','ሟ',rep27)
        rep29=re.sub('(ቱ[ዋአ])','ቷ',rep28)
        rep30=re.sub('(ሩ[ዋአ])','ሯ',rep29)
        rep31=re.sub('(ሱ[ዋአ])','ሷ',rep30)
        rep32=re.sub('(ሹ[ዋአ])','ሿ',rep31)
        rep33=re.sub('(ቁ[ዋአ])','ቋ',rep32)
        rep34=re.sub('(ቡ[ዋአ])','ቧ',rep33)
        rep35=re.sub('(ቹ[ዋአ])','ቿ',rep34)
        rep36=re.sub('(ሁ[ዋአ])','ኋ',rep35)
        rep37=re.sub('(ኑ[ዋአ])','ኗ',rep36)
        rep38=re.sub('(ኙ[ዋአ])','ኟ',rep37)
        rep39=re.sub('(ኩ[ዋአ])','ኳ',rep38)
        rep40=re.sub('(ዙ[ዋአ])','ዟ',rep39)
        rep41=re.sub('(ጉ[ዋአ])','ጓ',rep40)
        rep42=re.sub('(ደ[ዋአ])','ዷ',rep41)
        rep43=re.sub('(ጡ[ዋአ])','ጧ',rep42)
        rep44=re.sub('(ጩ[ዋአ])','ጯ',rep43)
        rep45=re.sub('(ጹ[ዋአ])','ጿ',rep44)
        rep46=re.sub('(ፉ[ዋአ])','ፏ',rep45)
        rep47=re.sub('[ቊ]','ቁ',rep46) #ቁ can be written as ቊ
        rep48=re.sub('[ኵ]','ኩ',rep47) #ኩ can be also written as ኵ  
        
        return rep48
    
```

replacing any existance of special character or punctuation to null  Amharic puncutation marks: =፡።፤;፦፧፨፠፣ 
``` python
def remove_punc_and_special_chars(text): 
    normalized_text = re.sub('[\!\@\#\$\%\^\«\»\&\*\(\)\…\[\]\{\}\;\“\”\›\’\‘\"\'\:\,\.\‹\/\<\>\?\\\\|\`\´\~\-\=\+\፡\።\፤\;\፦\፥\፧\፨\፠\፣]', '',text) 
    return normalized_text

#remove all ascii characters and Arabic and Amharic numbers
def remove_ascii_and_numbers(text_input):
    rm_num_and_ascii=re.sub('[A-Za-z0-9]','',text_input)
    return re.sub('[\'\u1369-\u137C\']+','',rm_num_and_ascii)
```

### Multi-word detection using with collocation finder

In natural language token can be formed from single or multi words. Thus, in order to consider those tokens formed from multi-words, component that dedicated to detect their existence is highly required under preprocessing stage. First I have tokenized the each sentences into list of tokens. 

In Amharic, the individual words in a sentence are separated by two dots (: ሁለትነጥብ). The end of a sentence is marked by Amharic full stop (። አራት ነጥብ). The symbol (፣ ነጠላ ሰረዝ) represents a comma, while (፤ ድርብ ሰረዝ) correspond to a semicolon. ‘!’ and ‘?’ punctuations are used to end exclamatory and interogative sentence respectively. 

Then using n-gram multi word detection approach, multiwords are detected.
* The first process in this component is forming all possible bi-grams from tokenized input text.
* Next, chi-square computation is applied to detect multi-words from the possible bigrams those their chi-square value is greater than experimentally chosen threshold value.

``` python
from nltk import BigramCollocationFinder
import nltk.collocations 
import io
import re
import os
class normalize(object):
    def tokenize(self,corpus):
        print('Tokenization ...')
        all_tokens=[]
        sentences=re.compile('[!?።(\፡\፡)]+').split(corpus)
        for sentence in sentences:
            tokens=sentence.split() # expecting non-sentence identifies are already removed
            all_tokens.extend(tokens)
        
        return all_tokens

    def collocation_finder(self,tokens,bigram_dir):
   
        bigram_measures = nltk.collocations.BigramAssocMeasures()
        
        #Search for bigrams with in a corpus
        finder = BigramCollocationFinder.from_words(tokens)
        
        #filter only Ngram appears morethan 3+ times
        finder.apply_freq_filter(3)
        
        frequent_bigrams = finder.nbest(bigram_measures.chi_sq,5) # chi square computer 
        print(frequent_bigrams)
        PhraseWriter = io.open(bigram_dir, "w", encoding="utf8")
        
        for bigram in frequent_bigrams:
            PhraseWriter.write(bigram[0]+' '+bigram[1] + "\n")
   
    def normalize_multi_words(self,tokenized_sentence,bigram_dir,corpus):
      #bigram_dir: is the directory to store multi-words
        bigram=set()
        sent_with_bigrams=[]
        index=0
        if not os.path.exists(bigram_dir):
            self.collocation_finder(self.tokenize(corpus),bigram_dir)
            #calling itsef
            self.normalize_multi_words(tokenized_sentence,bigram_dir,corpus)
        else:
            text=open(bigram_dir,encoding='utf8')
            for line in iter(text):
               line=line.strip()
               if not line:  # line is blank
                   continue
               else:
                   bigram.add(line)
            if len(tokenized_sentence)==1:
                sent_with_bigrams=tokenized_sentence
            else:
                while index <=len(tokenized_sentence)-2:
                    mword=tokenized_sentence[index]+' '+tokenized_sentence[index+1]
                    if mword in bigram:
                        sent_with_bigrams.append(tokenized_sentence[index]+''+tokenized_sentence[index+1])
                        index+=1
                    else:
                        sent_with_bigrams.append(tokenized_sentence[index])
                    index+=1
                    if index==len(tokenized_sentence)-1:
                        sent_with_bigrams.append(tokenized_sentence[index])
                
            return sent_with_bigrams
    
```
### Normalize Geez and Arabic Number Mismatch
This code snippet allows you to expand decimal form numbers to text representation. It also automatically normalize arabic numbers to Geez form. For example, 1=፩, 2=፪, ...

``` python
def arabic2geez(arabicNumber):
        ETHIOPIC_ONE= 0x1369
        ETHIOPIC_TEN= 0x1372
        ETHIOPIC_HUNDRED= 0x137B
        ETHIOPIC_TEN_THOUSAND = 0x137C
        arabicNumber=str(arabicNumber)
        n = len(arabicNumber)-1 #length of arabic number
        if n%2 == 0:
            arabicNumber = "0" + arabicNumber
            n+=1
        arabicBigrams=[arabicNumber[i:i+2] for i in range(0,n,2)] #spliting bigrams
        reversedArabic=arabicBigrams[::-1] #reversing list content
        geez=[]
        for index,pair in enumerate(reversedArabic):
            curr_geez=''
            artens=pair[0]#arrabic tens
            arones=pair[1]#arrabic ones
            amtens=''
            amones=''
            if artens!='0':
                amtens=str(chr((int(artens) + (ETHIOPIC_TEN - 1)))) #replacing with Geez 10s [፲,፳,፴, ...]
            else:
                if arones=='0': #for 00 cases
                    continue
            if arones!='0':       
                    amones=str(chr((int(arones) + (ETHIOPIC_ONE - 1)))) #replacing with Geez Ones [፩,፪,፫, ...]
            if index>0:
                if index%2!= 0: #odd index
                    curr_geez=amtens+amones+ str(chr(ETHIOPIC_HUNDRED)) #appending ፻
                else: #even index
                    curr_geez=amtens+amones+ str(chr(ETHIOPIC_TEN_THOUSAND)) # appending ፼
            else: #last bigram (right most part)
                curr_geez=amtens+amones
            
            geez.append(curr_geez)
        
        geez=''.join(geez[::-1])
        if geez.startswith('፩፻') or geez.startswith('፩፼'):
            geez=geez[1:]
        
        if len(arabicNumber)>=7:
            end_zeros=''.join(re.findall('([0]+)$',arabicNumber)[0:])
            i=int(len(end_zeros)/3)
            if len(end_zeros)>=(3*i):
                if i>=3:
                    i-=1
                for thoushand in range(i-1):
                    print(thoushand)                
                    geez+='፼'

        return geez
    def getExpandedNumber(self,number):
        if '.' not in str(number):
            return arabic2geez(number)
        else:
            num,decimal=str(number).split('.')
            if decimal.startswith('0'):
                decimal=decimal[1:]
                dot=' ነጥብ ዜሮ '
            else:
                dot=' ነጥብ '
            return arabic2geez(num)+dot+self.arabic2geez(decimal)
````

Your comments are my teacher. So drop any comments.
