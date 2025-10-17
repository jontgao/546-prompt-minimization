###############################################################################
# Scorer class
###############################################################################
class Scorer():
    """Base Scorer class to define a standard scoring interface."""
    
    def __init__(self, scorer=None):
        self.scorer = scorer

    def compute_score(self, output, expected_output):
        """
        Compute score between actual and expected output.
        
        :param output: List of strings
        :param expected_output: List of strings
        :returns: tensor of floats, each representing the score for the corresponding pair of output and expected_output
        """
        raise NotImplementedError("Subclasses must implement this method.")

###############################################################################
# Semantic Similarity Metrics
###############################################################################
class BERTScoreScorer(Scorer):
    def __init__(self, lang='en'):
        from bert_score import BERTScorer
        super().__init__(scorer=BERTScorer(lang=lang))

    def compute_score(self, output, expected_output):
        _, _, F1 = self.scorer.score(output, expected_output)
        return F1

# class BLEURTScorer(Scorer):
#     def __init__(self, checkpoint='BLEURT-20'):
#         from bleurt import score as bleurt_score
#         super().__init__(scorer=bleurt_score.BleurtScorer(checkpoint=checkpoint))

#     def compute_score(self, output, expected_output):
#         return self.scorer.score(references=expected_output, candidates=output)

class SentenceBERTCosineScorer(Scorer):
    def __init__(self, model='all-MiniLM-L6-v2'):
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        super().__init__(scorer=SentenceTransformer(model))
        self._cos = cosine_similarity

    def compute_score(self, output, expected_output):
        cand_emb = self.scorer.encode(output)
        ref_emb = self.scorer.encode(expected_output)
        scores = self._cos(cand_emb, ref_emb)
        return scores

###############################################################################
# Compression Metrics
###############################################################################
class CompressionLengthScorer(Scorer):
    """Compression scorer based on string length (not token counts)."""
    def compute_score(self, output, expected_output):
        if len(expected_output) == 0:
            return float('inf')
        return len(output) / len(expected_output)

class CompressionTokenScorer(Scorer):
    """Compression scorer based on token counts (not string length)."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def compute_score(self, output, expected_output):
        output_tokens = self.tokenizer.encode(output)
        expected_tokens = self.tokenizer.encode(expected_output)
        if len(expected_tokens) == 0:
            return float('inf')
        return len(output_tokens) / len(expected_tokens)
    