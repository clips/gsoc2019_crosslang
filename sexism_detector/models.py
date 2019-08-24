from pathlib import Path

import nltk
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from allennlp.data.iterators import BucketIterator
from allennlp.nn import util as nn_util
import torch
import numpy as np
from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.fields import TextField, LabelField
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.nn.util import get_text_field_mask
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import ElmoTokenEmbedder
from allennlp.modules.seq2vec_encoders import  PytorchSeq2VecWrapper
import torch.optim as optim
import torch.nn.functional
from torch import nn
from allennlp.training.trainer import Trainer
from tqdm import tqdm
from allennlp.data.iterators import BasicIterator


def tonp(tsr): return tsr.detach().cpu().numpy()


torch.manual_seed(1)
USE_GPU = torch.cuda.is_available()


class SexistDataReader(DatasetReader):

    def __init__(self, token_indexers = None):
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": ELMoTokenCharactersIndexer()}

    def text_to_instance(self,tokens,labels):
        if not tokens:
            tokens=[Token('')]
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"tokens": sentence_field}
        if labels:
            label_field = LabelField(labels)

            fields["label"] = label_field
        return Instance(fields)

    def _read(self, file_path):
        data = pd.read_csv(file_path)
        for row in data.itertuples():
            label = row.Label
            sentence = row.Text
            yield self.text_to_instance([Token(x) for x in nltk.word_tokenize(sentence, language='russian')[:200]], label)


class Predictor:
    def __init__(self, model, iterator,
                 cuda_device= -1):
        self.model = model
        self.iterator = iterator
        self.cuda_device = cuda_device

    def _extract_data(self, batch):
        out_dict = self.model(**batch)
        return tonp(out_dict["class_logits"])

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


class BaselineModel(Model):
    def __init__(self, word_embeddings_a,
                 encoder,
                 out_sz= 2):
        super().__init__(vocab)
        self.word_embeddings = word_embeddings_a
        self.encoder = encoder
        self.activation = nn.Softmax()
        self.projection = nn.Linear(self.encoder.get_output_dim(), out_sz)
        self.loss = nn.CrossEntropyLoss(weight=torch.Tensor([0.05,0.95]))


    def forward(self, tokens, label):
        mask = get_text_field_mask(tokens)
        embeddings = self.word_embeddings(tokens)
        state = self.encoder(embeddings, mask)
        class_logits = self.projection(state)
        output = {"class_logits": self.activation(class_logits), "loss": self.loss(class_logits, label)}
        return output

# TODO: make it universal somehow?
reader = SexistDataReader()
sexism_test = pd.read_csv('TemporalCorpora\\test.csv')
sexism_train = pd.read_csv('TemporalCorpora\\train.csv')
sexism_full = pd.concat([sexism_test, sexism_train], ignore_index=True)
sexism_full.to_csv('sexist.csv',encoding='utf-8')
full_ds = reader.read('sexist.csv')
labels_enc = LabelEncoder()
y_full = labels_enc.fit_transform(sexism_full['Label'])
train_ds, test_ds, y_train, y_test = train_test_split(full_ds, y_full, shuffle=True, stratify=y_full, train_size=0.9, test_size=0.1)
vocab = Vocabulary.from_instances(train_ds+test_ds)
iterator = BucketIterator(batch_size=32,
                          biggest_batch_first=True,
                          sorting_keys=[("tokens", "num_tokens")],
                          padding_noise=.15)
iterator.index_with(vocab)
batch = next(iter(iterator(train_ds)))
EMBEDDING_DIM = 256
HIDDEN_DIM = 64
# These files are trained by us, for pretrained ELMO just to take pretrained ones
options_file = 'forELMO\\options.json'
weight_file = 'forELMO\\corp_trained.hdf5'
elmo_embedder = ElmoTokenEmbedder(options_file, weight_file)
word_embeddings = BasicTextFieldEmbedder({"tokens": elmo_embedder})
lstm = PytorchSeq2VecWrapper(torch.nn.LSTM(word_embeddings.get_output_dim(), HIDDEN_DIM, batch_first=True, bidirectional=True))
model = BaselineModel(word_embeddings, lstm)
batch = nn_util.move_to_device(batch, 0)
train_dataset, val_dataset = train_test_split(train_ds, train_size=0.9, test_size=0.1, shuffle=True, stratify=y_train)
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
                  num_epochs=100,
                  cuda_device=0)
seq_iterator = BasicIterator(batch_size=32)
seq_iterator.index_with(vocab)
predictor = Predictor(model, seq_iterator, cuda_device=0 if USE_GPU else -1)
test_preds = predictor.predict(test_ds)
pred_ids = np.argmax(test_preds, axis=-1)
# Results are yet again given in balanced accuracy => imbalanced corpus + for consistency sake
print(balanced_accuracy_score(y_test ,pred_ids))

