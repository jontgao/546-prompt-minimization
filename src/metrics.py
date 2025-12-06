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

        This is a wrapper function to handle different input types. Permissible input types are:
            1. List of strings for both output and expected_output (of same length)
                Returns list of scores for each corresponding pair
            2. Single string for both output and expected_output
                Returns score between the two strings
            3. List of strings for output and single string for expected_output
                Returns list of scores for each output against the single expected_output
        
        :param output: List of strings OR single string
        :param expected_output: List of strings OR single string
        :returns: tensor of floats, each representing the score for the corresponding pair of output and expected_output
        """
        if isinstance(output, list) and isinstance(expected_output, list):
            assert len(output) == len(expected_output), "Output and expected_output lists must be of the same length."
            return self._compute_score_output_expected_pairs(output, expected_output)
        elif isinstance(output, str) and isinstance(expected_output, str):
            return self._compute_score_output_expected_pairs([output], [expected_output])[0]
        elif isinstance(output, list) and isinstance(expected_output, str):
            return self._compute_score_output_expected_pairs(output, [expected_output] * len(output))
        else:
            raise TypeError("Types of output and expected_output not supported.")

    def _compute_score_output_expected_pairs(self, outputs, expected_outputs):
        """
        Compute score between actual and expected output.
        
        :param output: List of strings
        :param expected_output: List of strings
        :returns: tensor of floats, each representing the score for the corresponding pair of output and expected_output
        """
        assert len(outputs) == len(expected_outputs), "Output and expected_output lists must be of the same length."
        raise NotImplementedError("Subclasses must implement this method.")


###############################################################################
# Semantic Similarity Metrics
###############################################################################
class BERTScoreScorer(Scorer):
    def __init__(self, lang='en'):
        from bert_score import BERTScorer
        super().__init__(scorer=BERTScorer(lang=lang))

    def _compute_score_output_expected_pairs(self, outputs, expected_outputs):
        _, _, F1 = self.scorer.score(outputs, expected_outputs)
        return F1.tolist()


class BLEURTScorer(Scorer):
    def __init__(self, checkpoint='BLEURT-20'):
        from bleurt import score as bleurt_score
        super().__init__(scorer=bleurt_score.BleurtScorer(checkpoint=checkpoint))

    def _compute_score_output_expected_pairs(self, outputs, expected_outputs):
        return self.scorer.score(references=expected_outputs, candidates=outputs)


class SentenceBERTCosineScorer(Scorer):
    def __init__(self, model='all-MiniLM-L6-v2'):
        from sentence_transformers import SentenceTransformer
        super().__init__(scorer=SentenceTransformer(model))

    def _compute_score_output_expected_pairs(self, outputs, expected_outputs):
        cand_emb = self.scorer.encode(outputs)
        ref_emb = self.scorer.encode(expected_outputs)
        return self.scorer.similarity_pairwise(cand_emb, ref_emb).tolist()

###############################################################################
# Compression Metrics
###############################################################################
class CompressionLengthScorer(Scorer):
    """Compression scorer based on string length (not token counts)."""

    def _compute_score_output_expected_pairs(self, outputs, expected_outputs):
        return [float('inf') if len(single_expected_output) == 0 else len(single_output) / len(single_expected_output)
                for single_output, single_expected_output in zip(outputs, expected_outputs)]


class CompressionTokenScorer(Scorer):
    """Compression scorer based on token counts (not string length)."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def _compute_score_output_expected_pairs(self, outputs, expected_outputs):
        # TODO: vectorize
        output_tokens = [self.tokenizer.encode(single_output) for single_output in outputs]
        expected_tokens = [self.tokenizer.encode(single_expected_output) for single_expected_output in expected_outputs]

        return [float('inf') if len(single_expected_output_tokens) == 0 else len(single_output_tokens) / len(
            single_expected_output_tokens)
                for single_output_tokens, single_expected_output_tokens in zip(output_tokens, expected_tokens)]


if __name__ == '__main__':
    from transformers import AutoTokenizer

    scorers = [BERTScoreScorer(), BLEURTScorer(), SentenceBERTCosineScorer(), CompressionLengthScorer(),
               CompressionTokenScorer(tokenizer=AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0'))]
    for scorer in scorers:
        output = ["The cat sat on the mat.", "A quick brown fox jumps over the lazy dog."]
        expected_output = ["The cat is sitting on the mat.", "A fast brown fox leaps over a lazy dog."]

        print("\n\nSCORER: ", type(scorer).__name__)

        print("Single string input, single string output:", output[0], expected_output[0])
        score = scorer.compute_score(output[0], expected_output[0])
        assert isinstance(score, float)
        print(score)

        print("List of strings input, single string output:", output, expected_output[0])
        score = scorer.compute_score(output, expected_output[0])
        assert isinstance(score, list) and len(score) == 2
        print(score)

        print("List of strings input, List of strings output:", output, expected_output)
        score = scorer.compute_score(output, expected_output)
        assert isinstance(score, list) and len(score) == 2
        print(score)
