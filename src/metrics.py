from bert_score import BERTScorer
# from bleurt import score as bleurt_score
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

###############################################################################
# Semantic Similarity Metrics
###############################################################################
bert_scorer = BERTScorer(lang='en')
def bert_scoring(output, expected_output):
    """
    Compute BERTScore F1 between actual and expected output.
    
    :output: List of strings
    :expected_output: List of strings
    :returns: tensor of floats, each representing the BERTScore F1 for the corresponding pair of output and expected_output
    """
    _, _, F1 = bert_scorer.score(output, expected_output)
    return F1

# bleurt_scorer = bleurt_score.BleurtScorer(checkpoint='BLEURT-20')
# def bleurt_scoring(output, expected_output):
#     """
#     Compute BLEURT scores between actual and expected output.
    
#     :output: List of strings
#     :expected_output: List of strings
#     :returns: tensor of floats, each representing the BLEURT score for the corresponding pair of output and expected_output
#     """
#     return bleurt_scorer.score(references=expected_output, candidates=output)

bert_cosine_model = SentenceTransformer('all-MiniLM-L6-v2')
def compute_sentence_bert_cosine(output, expected_output):
    """
    Compute cosine similarity between actual and expected output using Sentence-BERT.

    :param output: List of strings
    :param expected_output: List of strings
    :returns: List of floats, each representing the cosine similarity for the corresponding pair of output and expected_output
    """
    cand_emb = bert_cosine_model.encode(output)
    ref_emb = bert_cosine_model.encode(expected_output)
    scores = cosine_similarity(cand_emb, ref_emb)
    return scores

###############################################################################
# Compression Metrics
###############################################################################
def compression_score(output, expected_output):
    """
    Compute the compression ratio of the output compared to the expected output.
    Note this compares the lengths of the strings, not their token counts.
    """
    return len(output) / len(expected_output)

def compression_score_tokens(output, expected_output, tokenizer):
    """
    Compute the compression ratio of the output compared to the expected output.
    Note this compares the token counts, not the lengths of the strings.
    """
    output_tokens = tokenizer.encode(output)
    expected_tokens = tokenizer.encode(expected_output)

    if len(output_tokens) == 0:
        return float('inf')

    return len(expected_tokens) / len(output_tokens)