import argparse

from minimize import MultiStageOptimization

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Minimization parser')

    parser.add_argument('--dataset', type=str, choices=[''], help='Datasets available')
    parser.add_argument('--run_folder', type=str, default='runs', help='Datasets available')
    parser.add_argument('--model', type=str, default='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                        help='Model to use for the inference')
    parser.add_argument('--temp', type=float, default=1.0, help='Temperature to use for the inference')
    parser.add_argument('--top_p', type=float, default=0.95, help='Top p parameter for the inference')
    parser.add_argument('--seed', type=int, default=0, help='Seed to use')
    parser.add_argument('--bert_score_weight', type=float, default=10.0, help='BERT score weight')
    parser.add_argument('--compression_weight', type=float, default=1.0, help='Compression score weight')
    parser.add_argument('--num_iterations', type=int, default=10, help='Max iterations number for optimization')
    parser.add_argument('--top_n', type=int, default=1,
                        help='Use the top n best scoring prompts as seed for the next sampling')
    parser.add_argument('--batch_size', type=int, default=10, help='The number of prompts to generate per generation')
    parser.add_argument('--max_token_length', type=int, default=1_000, help='Max token length to output')
    parser.add_argument('--method', type=str, choices=['ga', 'agent'], default='ga',
                        help='The optimization algorithm to use')
    parser.add_argument("--use_initial_output", type=bool, default=True, action=argparse.BooleanOptionalAction,
                        help="Use the output given, otherwise generate the output from the LLM.")

    config = parser.parse_args()

    prompt = 'Who was Kyle Van Zyl playing against when he scored 36 of hisa teams 61 points?'

    output = '''Van Zyl joined the Eastern Province Kings Academy, where he played for the Eastern Province U19 side in the 2010 Under-19 Provincial Championship. He was a key player for the Eastern Province U21 side in the 2012 Under-21 Provincial Championship, scoring 71 points in eight appearances. Van Zyl was under the Top SARU Performers, scoring the most tries at 6 in the 2012 Provincial Under 21 in the Rugby Junior Provincials.

    This included a record and a remarkable personal haul in their opening match, when he scored 36 of his team's points in a 61–3 victory over Boland U21, consisting of four tries and eight conversions and was awarded Man of the Match.'''

    if config.method == 'ga':
        temp = MultiStageOptimization(config)

        temp(prompt, output)
    else:
        raise NotImplementedError()
