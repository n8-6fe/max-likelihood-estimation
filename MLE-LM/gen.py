from nltk.util import pad_sequence
from nltk.util import bigrams
from nltk.util import ngrams
from nltk.util import everygrams
from nltk.lm.preprocessing import pad_both_ends
from nltk.lm.preprocessing import flatten

from nltk import word_tokenize, sent_tokenize
from nltk.lm.preprocessing import padded_everygram_pipeline


#### example sentence to show how words are broken up/ tokenized
text = [['I','need','to','book', 'ticket', 'to', 'Australia' ], 
        ['I', 'want', 'to' ,'read', 'a' ,'book', 'of' ,'Shakespeare']]
# creating bigrams of the first element in the text array
print(list(bigrams(text[0])))
#creating trigrams of the second element in the text array
print(list(ngrams(text[1], n=3)))
####


#grab data to use to create ngrams
import pandas as pd
df = pd.read_csv("C:\\Users\\User\\Documents\\brushUp\\repub\\republic_book7.csv")
df.head()

#use the data to create a corpus by tokenizing all the words from the text
repub_corpus = list(df['content'].apply(word_tokenize))

# number for creating n-grams
# in this case we're creating tri-grams
n = 3
train_data, padded_sents = padded_everygram_pipeline(n, repub_corpus)

#create a language model and train it with our corpus/ngrams
from nltk.lm import MLE
repub_model = MLE(n)
repub_model.fit(train_data, padded_sents)

#from the trained language model we can generate new sentences based on trigrams
from nltk.tokenize.treebank import TreebankWordDetokenizer

detokenize = TreebankWordDetokenizer().detokenize

def generate_sent(model, num_words, random_seed=42):
    """
    :param model: An ngram language model from `nltk.lm.model`.
    :param num_words: Max no. of words to generate.
    :param random_seed: Seed value for random.
    """
    #create a sentence by getting the next word, if it is not an empty place holder
    # detokenize it and add it to the sentence
    content = []
    for token in model.generate(num_words, random_seed=random_seed):
        if token == '<s>':
            continue
        if token == '</s>':
            break
        content.append(token)
    return detokenize(content)



sent1 = generate_sent(repub_model, num_words=20, random_seed=10)
#because of the Great State of Pennsylvania, for himself, the “ Congressional Slush Fund, ” said Ralph

sent2 = generate_sent(repub_model, num_words=6, random_seed=42)
#for 200 years . Thank you

sent3 = generate_sent(repub_model, num_words=10, random_seed=0)
#she treated me fairly, they will be incredible—best in

print(sent1)