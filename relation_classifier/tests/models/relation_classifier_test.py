# pylint: disable=invalid-name,protected-access
from flaky import flaky
import pytest

from allennlp.common.testing import ModelTestCase
from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.models import Model


class RelationClassifierTest(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model('relation_classifier/experiments/concat_reprs.json',
                          'data/SemEval2017_FullContext/train.json')

    def test_simple_tagger_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    @flaky
    def test_batch_predictions_are_consistent(self):
        self.ensure_batch_predictions_are_consistent()

    def test_forward_pass_runs_correctly(self):
        training_tensors = self.dataset.as_tensor_dict()
        output_dict = self.model(**training_tensors)
       # assert len(tags) == 2
       # assert len(tags[0]) == 7
       # assert len(tags[1]) == 7
       # for example_tags in tags:
       #     for tag_id in example_tags:
       #         tag = self.model.vocab.get_token_from_index(tag_id, namespace="labels")
       #         assert tag in {'O', 'I-ORG', 'I-PER', 'I-LOC'}

