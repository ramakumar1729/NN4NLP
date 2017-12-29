from typing import Dict, Optional

import numpy
from overrides import overrides
import torch
import torch.nn.functional as F

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy, F1Measure


@Model.register("relation_classifier")
class RelationClassifier(Model):
    """
    This ``Model`` performs classification for a relation between pair of entities.  

    The basic model structure: we'll embed span between entities, left and right contexts, POS, NER tags, and encode each of them with
    separate Seq2VecEncoders, getting a single vector representing the content of each.  We'll then
    concatenate those two vectors, and pass the result through a feedforward network, the output of
    which we'll use as our scores for each label.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    e2e_encoder : ``Seq2VecEncoder``
        The encoder that we will use to convert the entity to entity span to a vector.
    left_context_encoder : ``Seq2VecEncoder``
        The encoder that we will use to convert the left context to a vector.
    right_context_encoder : ``Seq2VecEncoder``
        The encoder that we will use to convert the right context to a vector.
    relative_position_encoder : ``Seq2VecEncoder``
        Encoder for relative position.
    classifier_feedforward : ``FeedForward``
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 e2e_encoder: Seq2VecEncoder,
                 left_context_encoder: Seq2VecEncoder,
                 right_context_encoder: Seq2VecEncoder,
                 relative_position_encoder : Seq2VecEncoder,
                 classifier_feedforward: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(RelationClassifier, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.e2e_encoder = e2e_encoder
        self.left_context_encoder = left_context_encoder
        self.right_context_encoder = right_context_encoder
        self.relative_position_encoder = relative_position_encoder
        self.classifier_feedforward = classifier_feedforward

        if text_field_embedder.get_output_dim() != e2e_encoder.get_input_dim():
            raise ConfigurationError("The output dimension of the text_field_embedder must match the "
                                     "input dimension of the title_encoder. Found {} and {}, "
                                     "respectively.".format(text_field_embedder.get_output_dim(),
                                                            title_encoder.get_input_dim()))
        if text_field_embedder.get_output_dim() != left_context_encoder.get_input_dim():
            raise ConfigurationError("The output dimension of the text_field_embedder must match the "
                                     "input dimension of the abstract_encoder. Found {} and {}, "
                                     "respectively.".format(text_field_embedder.get_output_dim(),
                                                            abstract_encoder.get_input_dim()))
        self.metrics = {
                "accuracy": CategoricalAccuracy(),
                "accuracy3": CategoricalAccuracy(top_k=3),
                "F11": F1Measure(positive_label=1),
                "F12": F1Measure(positive_label=2),
                "F13": F1Measure(positive_label=3)
        }
        self.loss = torch.nn.CrossEntropyLoss()

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                e2e_tokens: Dict[str, torch.LongTensor],
                left_context_tokens: Dict[str, torch.LongTensor],
                right_context_tokens: Dict[str, torch.LongTensor],
                pos_tags:  Dict[str, torch.LongTensor],
                entity_tags: Dict[str, torch.LongTensor],
                relpos1_tags: Dict[str, torch.LongTensor],
                relpos2_tags: Dict[str, torch.LongTensor],
                relation_label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters (TODO)
        ----------
        title : Dict[str, Variable], required
            The output of ``TextField.as_array()``.
        abstract : Dict[str, Variable], required
            The output of ``TextField.as_array()``.
        label : Variable, optional (default = None)
            A variable representing the label for each instance in the batch.

        Returns
        -------
        An output dictionary consisting of:
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_classes)`` representing a distribution over the
            label classes for each instance.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        embedded_e2e = self.text_field_embedder(e2e_tokens)
        e2e_mask = util.get_text_field_mask(e2e_tokens)
        encoded_e2e = self.e2e_encoder(embedded_e2e, e2e_mask)

        embedded_left_context = self.text_field_embedder(left_context_tokens)
        left_context_mask = util.get_text_field_mask(left_context_tokens)
        encoded_left_context = self.e2e_encoder(embedded_left_context, left_context_mask)

        embedded_right_context = self.text_field_embedder(right_context_tokens)
        right_context_mask = util.get_text_field_mask(right_context_tokens)
        encoded_right_context = self.e2e_encoder(embedded_right_context, right_context_mask)

        logits = self.classifier_feedforward(torch.cat([encoded_e2e, encoded_left_context, encoded_right_context], dim=-1))
        # logits = self.classifier_feedforward(torch.cat([encoded_e2e], dim=-1))
        output_dict = {'logits': logits}
        if relation_label is not None:
            loss = self.loss(logits, relation_label.squeeze(-1))
            for metric in self.metrics.values():
                metric(logits, relation_label.squeeze(-1))
            output_dict["loss"] = loss

        return output_dict

    # TODO
    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the class probabilities, converts indices to string labels, and
        adds a ``"label"`` key to the dictionary with the result.
        """
        class_probabilities = F.softmax(output_dict['logits'])
        output_dict['class_probabilities'] = class_probabilities

        predictions = class_probabilities.cpu().data.numpy()
        argmax_indices = numpy.argmax(predictions, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace="labels")
                  for x in argmax_indices]
        output_dict['label'] = labels
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        d1 = {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items() if metric_name[0] != 'F'}
        for metric_name, metric in self.metrics.items():
            if metric_name[0] == 'F':
                precision, recall, f1_measure =  metric.get_metric(reset)
                d1["{}.{}".format(metric_name, "f1_measure")] = f1_measure
        return d1

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'RelationClassifier':
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)
        e2e_encoder = Seq2VecEncoder.from_params(params.pop("e2e_encoder"))
        left_context_encoder = Seq2VecEncoder.from_params(params.pop("left_context_encoder"))
        right_context_encoder = Seq2VecEncoder.from_params(params.pop("right_context_encoder"))
        relative_position_encoder =Seq2VecEncoder.from_params(params.pop("relative_position_encoder"))
        classifier_feedforward = FeedForward.from_params(params.pop("classifier_feedforward"))

        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))

        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   e2e_encoder=e2e_encoder,
                   left_context_encoder=left_context_encoder,
                   right_context_encoder=right_context_encoder,
                   relative_position_encoder=relative_position_encoder,
                   classifier_feedforward=classifier_feedforward,
                   initializer=initializer,
                   regularizer=regularizer)
