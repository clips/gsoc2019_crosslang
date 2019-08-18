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


# way to speed up the process, optional
USE_GPU = torch.cuda.is_available()

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
