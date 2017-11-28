from embedding_evaluation.evaluate_similarity import EvaluationSimilarity
from embedding_evaluation.evaluate_feature_norm import EvaluationFeatureNorm
from embedding_evaluation.evaluate_concreteness import EvaluationConcreteness


class Evaluation:

    def __init__(self, vocab_path=None, entity_subset=None, vocab=None):
        self.sim = EvaluationSimilarity(entity_subset=entity_subset)
        self.conc = EvaluationConcreteness(entity_subset=None)
        self.fn = None
        if vocab_path is not None:
            self.fn = EvaluationFeatureNorm(entity_subset=entity_subset, vocab_path=vocab_path, vocab=vocab)

    def evaluate(self, my_embedding):
        results = {}
        results["similarity"] = self.sim.evaluate(my_embedding)
        if self.fn is not None:
            results["feature_norm"] = self.fn.evaluate(my_embedding)
        results["concreteness"] = self.conc.evaluate(my_embedding)
        return results

