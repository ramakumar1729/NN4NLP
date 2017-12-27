from relation_classifier.dataset_readers.full_context_relations import FullContextRelationDatasetReader
from allennlp.common.testing import AllenNlpTestCase

class TestFullContextRelationDatasetReader(AllenNlpTestCase):
    def test_default_format(self):
        reader = FullContextRelationDatasetReader()
        dataset = reader.read("data/SemEval2017_FullContext/train.json")

        instance_fields = dataset.instances[0].fields
        assert [t.text for t in instance_fields["e2e_tokens"].tokens] == ["inorganic", "flocculating", "agents", ",", "including", "FeSO4"] 
