import os
from pathlib import Path

# Imports for preprocessing of Russian language texts
import csv
import nltk
import re
import csv
from string import punctuation
from pymystem3 import Mystem

# Imports for ELMO
import torch
from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.fields import TextField, LabelField
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import ElmoTokenEmbedder
from allennlp.modules.seq2vec_encoders import  PytorchSeq2VecWrapper
import torch.optim as optim
import torch.nn.functional
from torch import nn
from allennlp.training.trainer import Trainer
from allennlp.predictors import SentenceTaggerPredictor


torch.manual_seed(1)


russian_stopwords = nltk.corpus.stopwords.words("russian")
russian_stemmer = nltk.stem.snowball.SnowballStemmer("russian")
lemmatizer = Mystem()


# The function is to create from the given names of subcorpora a train and test dataset. There are quite a few functions which
#split in more efficient way, but I wanted to ensure that the elements even from the smallest corpora (as RT_media_corpus) will
#get in both test and train.

def create_train_and_test_folder(corpora_subset,new_train_filename,new_test_filename):
    # TODO: Not the most elegant solution, how to avoid chdir without getting os dependant?
    os.chdir("..")
    corpath =  os.getcwd() + "\\corpora\\annotated\\"
    new_path = os.getcwd() + '\\collect_corpora\\TemporalCorpora\\'

    corpora_subset = [corpus.strip() for corpus in corpora_subset.split(',')]
    sex_tr = sex_te = non_sex_tr = non_sex_te = 0
    with open(new_path+new_train_filename, 'w',encoding="utf-8") as writeTrainFile,  open(new_path+new_test_filename, 'w',encoding="utf-8") as writeTestFile:
        fieldnames = ['Label', 'Text']
        writer_train = csv.DictWriter(writeTrainFile, fieldnames=fieldnames)
        writer_test = csv.DictWriter(writeTestFile, fieldnames=fieldnames)
        writer_train.writeheader()
        writer_test.writeheader()
        for corpus in corpora_subset:
            with open((corpath+corpus),encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                tmp_counter_sex = tmp_counter_nonsex = 0
                for row in reader:
                    label = row['Label']
                    if label=="sexist":
                        if tmp_counter_sex>10:
                            writer_test.writerow({'Label': str(label+";"),'Text':row['Text']})
                            sex_te+=1
                            tmp_counter_sex = 0
                        else:
                            writer_train.writerow({'Label': str(label+";"), 'Text': row['Text']})
                            sex_tr+=1
                            tmp_counter_sex+=1
                    elif label=="non_sexist":
                        if tmp_counter_nonsex > 15:
                            writer_test.writerow({'Label': str(label+";"), 'Text': row['Text']})
                            non_sex_te+=1
                            tmp_counter_nonsex = 0
                        else:
                            writer_train.writerow({'Label': str(label+";"), 'Text': row['Text']})
                            non_sex_tr+=1
                            tmp_counter_nonsex+=1
        print("It is over")
        print("Your train set has {} sexist comments and {} non_sexist comments".format(sex_tr,non_sex_tr))
        print("Your test set has {} sexist comments and {} non_sexist comments".format(sex_te, non_sex_te))


# vk preprocessing - remove with regex IDs and quote prefixes

def remove_with_regex(text):
    regex = re.compile("\[[i][d][0-9]*\|.*],")
    regex1 = re.compile('^.*[ писал(а):]$')
    words = [re.sub(regex,"",word) and re.sub(regex1,"",word) for word in text.split()]
    return ' '.join(words)

# punctuation removed and words are lowercased

def remove_punctuation_and_make_lowercase(text):
    sentences = [(sentence.translate(str.maketrans('', '', punctuation))).lower() for sentence in text.split()]
    return ' '.join(sentences)

# Remove stopwords

def remove_stowords(text):
    text = [word for word in text.split() if word.lower() not in russian_stopwords]
    return ' '.join(text)

# Lemmatization is performed

def perform_lemmatization(text):
    text = lemmatizer.lemmatize(text)
    return ''.join(text)

# Stemmatization is performed

def stemmatize(text):
    text = [russian_stemmer.stem(word) for word in text.split()]
    return ' '.join(text)

# Three preprocessing phases. It is not very efficient, and can be improved.

def preprocessing(text, punct_low_case, stop_regex, lemm, stemm):
    if punct_low_case == True:
        text = remove_punctuation_and_make_lowercase(text)
    if stop_regex == True:
        text = remove_with_regex(text)
        text = remove_stowords(text)
    if lemm == True:
        text = perform_lemmatization(text)
    elif stemm == True:
        text = stemmatize(text)
    return text

# I prefer to have all different preprocessing stage corpora avaliable

def save_preprocessing_phase(filename,filename_out,punctuationBool,stopwordsBool,lemmBool):
     with open('TemporalCorpora\\'+ filename +'.csv', encoding="utf-8") as main_file, open('TemporalCorpora\\'+filename_out+'.csv',"w",encoding='utf-8') as file_no_stop:
         fieldnames = ['Label', 'Text']
         csv_reader = csv.DictReader(main_file)
         no_stopwords = csv.DictWriter(file_no_stop, fieldnames=fieldnames)
         no_stopwords.writeheader()
         for row in csv_reader:
             no_stopwords.writerow({'Label':row['Label'],'Text': preprocessing(row['Text'],punctuationBool,stopwordsBool,lemmBool,False)})

# Some would say it would be much smarter to do it earlier. Some may be are correct.

def cleaning_up_the_nulls(filename,filename_new):
    with open('TemporalCorpora\\' + filename, encoding="utf-8") as file_to_check, open('TemporalCorpora\\'+ filename_new,'w', encoding="utf-8") as file_to_write_in:
        fieldnames = ['Label', 'Text']
        csv_reader = csv.DictReader(file_to_check)
        new_file = csv.DictWriter(file_to_write_in, fieldnames=fieldnames)
        new_file.writeheader()
        for row in csv_reader:
            if row['Text'] is None or row['Text']=="":
                continue
            else:
                new_file.writerow({'Label':row['Label'],'Text':row['Text']})

# way to speed up the process, optional
USE_GPU = torch.cuda.is_available()
cleaning_up_the_nulls("tmp_train_no_stopwords.csv","train_no_stopwords.csv")
cleaning_up_the_nulls("tmp_test_no_stopwords.csv",'test_no_stopwords.csv')

class SexistDataReader(DatasetReader):

    def __init__(self, token_indexers = None):
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": ELMoTokenCharactersIndexer()}

    def text_to_instance(self,tokens,labels):
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"tokens": sentence_field}

        if labels:
            label_field = LabelField(labels)
            fields["label"] = label_field
        return Instance(fields)

    def _read(self, file_path):
        with open(file_path,encoding="utf-8") as f:
            csv_reader = csv.DictReader(f)
            for row in csv_reader:
                label = row['Label']
                sentence = row['Text']
                print(len(sentence))
                print(len(nltk.word_tokenize(sentence, language='russian')))
                yield self.text_to_instance([Token(x) for x in nltk.word_tokenize(sentence, language='russian')], label)


test = SexistDataReader()
DATA_ROOT = Path("TemporalCorpora")
train_ds, test_ds = (test.read(DATA_ROOT / fname) for fname in ["train_no_stopwords.csv", "test_no_stopwords.csv"])
# does it work?
print(vars(train_ds[7].fields['tokens']))
vocab = Vocabulary.from_instances(train_ds + test_ds)


from allennlp.data.iterators import BucketIterator
iterator = BucketIterator(batch_size=64,
                          biggest_batch_first=True,
                          sorting_keys=[("tokens", "num_tokens")],
                         )
iterator.index_with(vocab)

batch = next(iter(iterator(train_ds)))
print(batch['tokens']['tokens'].shape)


class BaselineModel(Model):
    def __init__(self, word_embeddings,
                 encoder,
                 out_sz= 2):
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.projection = nn.Linear(self.encoder.get_output_dim(), out_sz)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, tokens, label):
        print(tokens['tokens'])
        mask = get_text_field_mask(tokens)
        print(mask.shape)
        embeddings = self.word_embeddings(tokens)
        print(embeddings.shape)
        state = self.encoder(embeddings, mask)
        print(state.shape)
        class_logits = self.projection(state)
        output = {"class_logits": class_logits}
        print(class_logits.shape, torch.nn.functional.one_hot(label,2).shape)
        output["loss"] = self.loss(class_logits, torch.nn.functional.one_hot(label,2).float())
        return output


class LSTM_Sexist_Model(Model):
    def __init__(self, word_embeddings, encoder, vocab):
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.hidden2tag = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                          out_features=vocab.get_vocab_size('label'))
        self.accuracy = CategoricalAccuracy()


    def forward(self,tokens,label):
        mask = get_text_field_mask(tokens)
        embeddings = self.word_embeddings(tokens)
        encoder_out = self.encoder(embeddings, mask)
        tag_logits = self.hidden2tag(encoder_out)
        output = {"tag_logits": tag_logits}
        if label is not None:
            print(tag_logits)
            self.accuracy(tag_logits, label, mask)
            output["loss"] = sequence_cross_entropy_with_logits(tag_logits, label, mask)
        return output



    def get_metrics(self, reset):
        return {"accuracy": self.accuracy.get_metric(reset)}


EMBEDDING_DIM = 256
HIDDEN_DIM = 6


options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json'
weight_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5'

elmo_embedder = ElmoTokenEmbedder(options_file, weight_file)
word_embeddings = BasicTextFieldEmbedder({"tokens": elmo_embedder})
print('embed dim', word_embeddings.get_output_dim())
lstm = PytorchSeq2VecWrapper(torch.nn.LSTM(word_embeddings.get_output_dim(), HIDDEN_DIM, batch_first=True, bidirectional=True))
model = BaselineModel(word_embeddings,lstm)
#model = LSTM_Sexist_Model(word_embeddings, lstm, vocab)
if torch.cuda.is_available():
    cuda_device = 0
    model = model.cuda(cuda_device)

from allennlp.nn import util as nn_util

batch = nn_util.move_to_device(batch, 0 )

# tokens = batch['tokens']
# labels = batch
# print(tokens)
# mask = get_text_field_mask(tokens)
# print(mask.shape)
# embeddings = model.word_embeddings(tokens)
# state = model.encoder(embeddings, mask)
# class_logits = model.projection(state)
# print(class_logits.shape)
#
# print(model(**batch))

optimizer = optim.Adam(model.parameters(), lr=0.1)
trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=train_ds,
                  validation_dataset=test_ds,
                  patience=10,
                  num_epochs=1000,
                  cuda_device=0)
trainer.train()
predictor = SentenceTaggerPredictor(model, dataset_reader=test)
train_preds = predictor.predict(train_ds)
test_preds = predictor.predict(test_ds)
with open("/tmp/model.th", 'wb') as f:
    torch.save(model.state_dict(), f)
vocab.save_to_files("/tmp/vocabulary")

#media_corpora = 'media_1.csv, media_2.csv, media_3.csv'
#forum_corpora = 'ant_1.csv, ns_1.csv, ant_2.csv'
#all_corpora = media_corpora + ', ' + forum_corpora
#create_train_and_test_folder(all_corpora,'tmp_train.csv','tmp_test.csv')
#save_preprocessing_phase('tmp_test','tmp_test_no_stopwords',False,True,False)
#save_preprocessing_phase('tmp_test_no_stopwords','tmp_test_no_punctuation',True,False,False)
#save_preprocessing_phase('tmp_test','tmp_test_just_lemm_and_no_punct',True,False,True)
#save_preprocessing_phase('tmp_test_no_punctuation','tmp_test_lemmatized',False,False,True)
#save_preprocessing_phase('tmp_train','tmp_train_no_stopwords',False,True,False)
#save_preprocessing_phase('tmp_train_no_stopwords','tmp_train_no_punctuation',True,False,False)
#save_preprocessing_phase('tmp_train_no_punctuation','tmp_train_lemmatized',False,False,True)
#save_preprocessing_phase('tmp_train','tmp_train_just_lemm_and_no_punct',True,False,True)
