from pathlib import Path

from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

# Imports for ELMO
import csv
import nltk
import torch
import numpy as np
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

torch.manual_seed(1)


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
                label = row['Label'].replace(';', '')
                sentence = row['Text']
                yield self.text_to_instance([Token(x) for x in nltk.word_tokenize(sentence, language='russian')[:100]], label)


test = SexistDataReader()
DATA_ROOT = Path("TemporalCorpora")
train_ds, test_ds = (test.read(DATA_ROOT / fname) for fname in ["train_no_stopwords.csv", "test_no_stopwords.csv"])
# does it work?
vocab = Vocabulary.from_instances(train_ds + test_ds)


from allennlp.data.iterators import BucketIterator
iterator = BucketIterator(batch_size=64,
                          biggest_batch_first=True,
                          sorting_keys=[("tokens", "num_tokens")],
                         )
iterator.index_with(vocab)

batch = next(iter(iterator(train_ds)))


class BaselineModel(Model):
    def __init__(self, word_embeddings,
                 encoder,
                 out_sz= 2):
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.projection = nn.Linear(self.encoder.get_output_dim(), out_sz)
        self.loss = nn.BCEWithLogitsLoss(weight=torch.Tensor([0.1, 0.9]))
 #       self.loss = nn.BCEWithLogitsLoss()


    def forward(self, tokens, label):
        mask = get_text_field_mask(tokens)
        embeddings = self.word_embeddings(tokens)
        state = self.encoder(embeddings, mask)
        class_logits = self.projection(state)
        output = {"class_logits": class_logits}
        output["loss"] = self.loss(class_logits, torch.nn.functional.one_hot(label,2).float())
        return output


def get_labels(filename):
    with open('TemporalCorpora\\' + filename, encoding="utf-8") as main_file:
        csv_reader = csv.DictReader(main_file)
        true_labels = ""
        for row in csv_reader:
            true_labels = true_labels + row['Label']
        return true_labels
true_labels = get_labels('test_no_stopwords.csv')
true_labels_ls = true_labels.split(';')


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
            self.accuracy(tag_logits, label, mask)
            output["loss"] = sequence_cross_entropy_with_logits(tag_logits, label, mask)
        return output



    def get_metrics(self, reset):
        return {"accuracy": self.accuracy.get_metric(reset)}


EMBEDDING_DIM = 256
HIDDEN_DIM = 64


options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json'
weight_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5'

elmo_embedder = ElmoTokenEmbedder(options_file, weight_file)
word_embeddings = BasicTextFieldEmbedder({"tokens": elmo_embedder})
lstm = PytorchSeq2VecWrapper(torch.nn.LSTM(word_embeddings.get_output_dim(), HIDDEN_DIM, batch_first=True, bidirectional=True))
model = BaselineModel(word_embeddings,lstm)
#model = LSTM_Sexist_Model(word_embeddings, lstm, vocab)

from allennlp.nn import util as nn_util

batch = nn_util.move_to_device(batch, 0 )

train_dataset, val_dataset = train_test_split(train_ds, train_size=0.9, test_size=0.1, shuffle=True)

optimizer = optim.RMSprop(model.parameters(), lr=0.01)
if torch.cuda.is_available():
    cuda_device = 0
    model = model.cuda(cuda_device)

trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=train_dataset,
                  validation_dataset=val_dataset,
                  patience=10,
                  num_epochs=50,
                  cuda_device=0)
print(trainer.train())

from tqdm import tqdm
from scipy.special import expit  # the sigmoid function


def tonp(tsr): return tsr.detach().cpu().numpy()


class Predictor:
    def __init__(self, model, iterator,
                 cuda_device= -1):
        self.model = model
        self.iterator = iterator
        self.cuda_device = cuda_device

    def _extract_data(self, batch):
        out_dict = self.model(**batch)
        return expit(tonp(out_dict["class_logits"]))

    def predict(self, ds):
        pred_generator = self.iterator(ds, num_epochs=1, shuffle=False)
        self.model.eval()
        pred_generator_tqdm = tqdm(pred_generator,
                                   total=self.iterator.get_num_batches(ds))
        preds = []
        with torch.no_grad():
            for batch in pred_generator_tqdm:
                batch = nn_util.move_to_device(batch, self.cuda_device)
                preds.append(self._extract_data(batch))
        return np.concatenate(preds, axis=0)


from allennlp.data.iterators import BasicIterator

# iterate over the dataset without changing its order
seq_iterator = BasicIterator(batch_size=64)
seq_iterator.index_with(vocab)

predictor = Predictor(model, seq_iterator, cuda_device=0 if USE_GPU else -1)
test_preds = predictor.predict(test_ds)
pred_ids = np.argmax(test_preds, axis=-1)
mapper = {'sexist':  1, 'non_sexist': 0}
true_labels_ls.pop(-1)
print(pred_ids)
print(pred_ids, np.asarray(list(map(mapper.get, true_labels_ls))))
print(balanced_accuracy_score(np.asarray(list(map(mapper.get, true_labels_ls))) ,pred_ids))

vocab.save_to_files("/tmp/vocabulary")
