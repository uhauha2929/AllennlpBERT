import logging
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer

from typing import *
from overrides import overrides

import csv

logger = logging.getLogger(__name__)


@DatasetReader.register("text_classification_csv")
class TextClassificationCSVReader(DatasetReader):

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 delimiter: str = ',',
                 max_sequence_length: int = None,
                 lazy: bool = False) -> None:

        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._delimiter = delimiter
        self._max_sequence_length = max_sequence_length

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        with open(file_path, 'r') as text_file:
            reader = csv.DictReader(text_file)
            next(reader)
            for row in reader:
                # https://www.kaggle.com/crowdflower/twitter-airline-sentiment
                yield self.text_to_instance(row.get('text', ''), row['airline_sentiment'])

    def _truncate(self, tokens):
        """
        truncate a set of tokens using the provided sequence length
        """
        if len(tokens) > self._max_sequence_length:
            tokens = tokens[:self._max_sequence_length]
        return tokens

    @overrides
    def text_to_instance(self, text: str, label: Union[str, int] = None) -> Instance:
        """
        Parameters
        ----------
        text : ``str``, required.
            The text to classify
        label : ``str``, optional, (default = None).
            The label for this text.
        Returns
        -------
        An ``Instance`` containing the following fields:
            tokens : ``TextField``
                The tokens in the sentence or phrase.
            label : ``LabelField``
                The label label of the sentence or phrase.
        """
        fields: Dict[str, Field] = {}
        text_tokens = self._tokenizer.tokenize(text)
        if self._max_sequence_length is not None:
            text_tokens = self._truncate(text_tokens)
        fields['tokens'] = TextField(text_tokens, self._token_indexers)
        if label:
            fields['label'] = LabelField(label)

        return Instance(fields)
