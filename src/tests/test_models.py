# tests/embedding/test_models.py
import unittest
import torch
from transformers import BertConfig
from src.embedding.models import EmbeddingBert, BertLMPredictionHead, BertPredictionHeadTransform, embedding_model_factory  # Absolute import
import optuna

class TestEmbeddingModels(unittest.TestCase):

    def setUp(self):
        self.config = BertConfig.from_pretrained("bert-base-uncased")
        self.config.hidden_dropout_prob = 0.1  # Set a specific value
        self.config.attention_probs_dropout_prob = 0.1 # Set a specific value


    def test_embedding_bert_initialization(self):
        model = EmbeddingBert(config=self.config)
        self.assertIsInstance(model, EmbeddingBert)
        self.assertIsInstance(model.bert, BertModel)
        self.assertIsInstance(model.cls, BertEmbeddingHead)

    def test_embedding_bert_forward_pass(self):
        model = EmbeddingBert(config=self.config)
        input_ids = torch.randint(0, self.config.vocab_size, (1, 16))
        attention_mask = torch.ones((1, 16), dtype=torch.long)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        self.assertIsInstance(outputs, dict)
        self.assertTrue('logits' in outputs)
        self.assertEqual(outputs['logits'].shape, (1, 16, self.config.vocab_size))


    def test_embedding_bert_forward_pass_with_labels(self):
      model = EmbeddingBert(config=self.config)
      input_ids = torch.randint(0, self.config.vocab_size, (1, 16))
      attention_mask = torch.ones((1, 16), dtype=torch.long)
      labels = torch.randint(0, self.config.vocab_size, (1, 16))  # Provide some labels
      outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
      self.assertIsInstance(outputs, dict)
      self.assertTrue('loss' in outputs)
      self.assertTrue('logits' in outputs)
      self.assertEqual(outputs['logits'].shape, (1, 16, self.config.vocab_size))
      self.assertIsInstance(outputs['loss'], torch.Tensor)


    def test_bert_lm_prediction_head(self):
        head = BertLMPredictionHead(config=self.config)
        self.assertIsInstance(head, BertLMPredictionHead)
        hidden_states = torch.randn(1, 16, self.config.hidden_size)
        prediction_scores = head(hidden_states)
        self.assertEqual(prediction_scores.shape, (1, 16, self.config.vocab_size))

    def test_bert_prediction_head_transform(self):
        transform = BertPredictionHeadTransform(config=self.config)
        self.assertIsInstance(transform, BertPredictionHeadTransform)
        hidden_states = torch.randn(1, 16, self.config.hidden_size)
        transformed_states = transform(hidden_states)
        self.assertEqual(transformed_states.shape, (1, 16, self.config.hidden_size))

    def test_embedding_model_factory(self):
        config = {
            'model': {
                'name': 'bert-base-uncased',
                'tie_weights': True,
                'hidden_dropout_prob': 0.2,  # Add explicit values
                'attention_probs_dropout_prob': 0.2 # Add explicit values
            },
            'hyperparameters': { #Added
                'hidden_dropout_prob': {'min': 0.0, 'max': 0.3},
                'attention_probs_dropout_prob': {'min': 0.0, 'max': 0.3}
            }
        }
        model = embedding_model_factory(config)
        self.assertIsInstance(model, EmbeddingBert)
        self.assertEqual(model.config.hidden_dropout_prob, 0.2)
        self.assertEqual(model.config.attention_probs_dropout_prob, 0.2)

    def test_embedding_model_factory_with_trial(self):
        config = {
            'model': {
                'name': 'bert-base-uncased',
                'tie_weights': True
            },
            'hyperparameters': {
                'hidden_dropout_prob': {'min': 0.0, 'max': 0.3},
                'attention_probs_dropout_prob': {'min': 0.0, 'max': 0.3}
            }
        }
        def objective(trial):
            model = embedding_model_factory(config, trial=trial)
            # In a real test, you'd also evaluate the model here and return a score
            return model.config.hidden_dropout_prob  # Return a dummy value for testing
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=2) # Run a couple trials for test
        self.assertTrue(len(study.trials) == 2)

    def test_get_set_output_embeddings(self):
      model = EmbeddingBert(config=self.config)
      original_embeddings = model.get_output_embeddings()
      self.assertIsInstance(original_embeddings, nn.Linear)

      # Create new embeddings
      new_embeddings = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
      model.set_output_embeddings(new_embeddings)
      self.assertTrue(torch.equal(model.get_output_embeddings().weight, new_embeddings.weight))
      

if __name__ == '__main__':
    unittest.main()