from typing import Dict
import json
import logging

from overrides import overrides

import tqdm

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset import Dataset
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("full_context_relations")
class FullContextRelationDatasetReader(DatasetReader):
    """
    Reads a JSON-lines file containing entity to entity, relative positions, left context, right context, POS tags, NER tags and relation labels.

    Expected format for each input line: {"text": "", "relpos1": [], "relpos2" : [], "left_context" : "", 'right_context" : "", "pos" : [], "ner" :[], "relation" : "" }

    The JSON could have other fields, too, but they are ignored.

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the title and abstrct into words or other kinds of tokens.
        Defaults to ``WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input token representations. Defaults to ``{"tokens":
        SingleIdTokenIndexer()}``.
    """
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def read(self, file_path):
        instances = []
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line_num, line in enumerate(tqdm.tqdm(data_file.readlines())):
                line = line.strip("\n")
                if not line:
                    continue
                paper_json = json.loads(line)
                e2e = paper_json["e2e"]["text"]
                left_context = paper_json["left_context"]["text"]
                right_context = paper_json["right_context"]["text"]
                pos_tags = paper_json["e2e"]["pos"]
                ner_tags = paper_json["e2e"]["ner"]
                relpos1 = paper_json["e2e"]["relpos1"]
                relpos2 = paper_json["e2e"]["relpos2"]
                relation_label = paper_json["relation"]

                # e2e tokens.
                e2e_tokens = [ Token(e2e_token) for e2e_token in e2e]
                e2e_sequence = TextField(e2e_tokens, self._token_indexers)

                # Left context tokens.
                left_context_tokens = [ Token(token) for token in left_context]
                left_context_sequence = TextField(left_context_tokens, self._token_indexers)
                
                # Right context tokens.
                right_context_tokens = [ Token(token) for token in right_context]
                right_context_sequence = TextField(right_context_tokens, self._token_indexers)

                instance_fields = {"e2e_tokens" : e2e_sequence, "left_context_tokens": left_context_sequence, "right_context_tokens" : right_context_sequence}

                # Add POS tags.
                instance_fields["pos_tags"] = SequenceLabelField(pos_tags, e2e_sequence, "pos_tags")

                # Add NER tags.
                instance_fields["ner_tags"] = SequenceLabelField(ner_tags, e2e_sequence, "ner_tags")

                # Add relpos1.
                instance_fields["relpos1"] = SequenceLabelField(relpos1, e2e_sequence, "relpos1")

                # Add relpos2.
                instance_fields["relpos2"] = SequenceLabelField(relpos2, e2e_sequence, "relpos2")

                # Add relation label.
                instance_fields["relation_label"] = LabelField(relation_label)

                instances.append(Instance(instance_fields))
        if not instances:
            raise ConfigurationError("No instances read!")
        return Dataset(instances)

    @overrides
    # TODO: rewrite this.
    def text_to_instance(self, title: str, abstract: str, venue: str = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        tokenized_title = self._tokenizer.tokenize(title)
        tokenized_abstract = self._tokenizer.tokenize(abstract)
        title_field = TextField(tokenized_title, self._token_indexers)
        abstract_field = TextField(tokenized_abstract, self._token_indexers)
        fields = {'title': title_field, 'abstract': abstract_field}
        if venue is not None:
            fields['label'] = LabelField(venue)
        return Instance(fields)

    @classmethod
    def from_params(cls, params: Params) -> 'FullContextRelationDatasetReader':
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        params.assert_empty(cls.__name__)
        return cls(tokenizer=tokenizer, token_indexers=token_indexers)
