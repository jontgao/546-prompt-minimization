from bert_score import BERTScorer
# from bleurt import score as bleurt_score
# from sentence_transformers import SentenceTransformer, util
# import torch

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

def compute_sentence_bert_cosine(output, expected_output, model_name='all-MiniLM-L6-v2'):
    """
    Compute cosine similarity between actual and expected output using Sentence-BERT.
    """
    model = SentenceTransformer(model_name)
    cand_emb = model.encode(output, convert_to_tensor=True)
    ref_emb = model.encode(expected_output, convert_to_tensor=True)
    scores = [util.pytorch_cos_sim(c, r).item() for c, r in zip(cand_emb, ref_emb)]
    return scores


###############################################################################
# Compression Metrics
###############################################################################
def compression_ratio(output, expected_output):
    """
    Compute the compression ratio of the output compared to the expected output.
    Note this compares the lengths of the strings, not their token counts.
    """
    return len(output) / len(expected_output)

# def compression_ratio_tokens(output, expected_output):
#     """
#     Compute the compression ratio of the output compared to the expected output.
#     Note this compares the token counts, not the lengths of the strings.
#     """
#     pass