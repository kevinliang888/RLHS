import argparse
import os
from marketplaceChat import MarketplaceChatBot
from marketplace import Marketplace
from llm import load_model
import json
import pdb
import time
import torch

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory {path} created.")
    else:
        print(f"Directory {path} already exists.")


def gen_rlhf_train_data(ai_model, human_model, output, index, thres):
    """This is the main one for training."""
    chatbot = MarketplaceChatBot(ai_model, human_model)
    output_path = f"{output}/results_{ai_model}_{human_model}_{index}.json"
    final_output_path = f"{output}/final_results_{ai_model}_{human_model}_{index}.json"
    marketplace = Marketplace(chatbot)
    all_results = marketplace.generate_scenarios()
    rlhf_data = marketplace.generate_rlhf_data(all_results, output_path, thres=thres)
    with open(final_output_path, 'w') as file:
        json.dump(rlhf_data, file)


def gen_test_data_opt(ai_model_names, human_model_name, test_data, output, index, load_in_8bit, rlhf_type):

    output_path = f"{output}/test_data_{index}.json"
    final_output_path = f"{output}/final_test_data_{index}.json"

    human_model, human_tokenizer = load_model(human_model_name, load_in_8bit=load_in_8bit)
    all_results = json.load(open(test_data, 'r'))
    create_directory(output)

    for ai_model_name in ai_model_names:
        ai_model, ai_tokenizer = load_model(ai_model_name)
        chatbot = MarketplaceChatBot(ai_model_name, human_model_name, ai_model, ai_tokenizer, human_model, human_tokenizer)
        marketplace = Marketplace([chatbot])
        all_results = marketplace.generate_test_data(all_results, output_path=output_path, rlhf_type=rlhf_type)
        output_path2 = f"{output}/test_data_{ai_model_name}_{index}.json"
        with open(output_path2, 'w') as file:
            json.dump(all_results, file)

    with open(final_output_path, 'w') as file:
        json.dump(all_results, file)



def test_inference_opt(ai_model_name, human_model_name, ai_model_directory, ckpts, test_data, output, index, load_in_8bit, rlhf_type):
     """This is the main one for inference."""
    output_path = f"{output}/test_inf_{ai_model_name}_{index}.json"
    final_output_path = f"{output}/final_test_inf_{ai_model_name}_{index}.json"

    human_model, human_tokenizer = load_model(human_model_name, load_in_8bit=load_in_8bit)
    all_results = json.load(open(test_data, 'r'))
    create_directory(output)

    for ckpt in ckpts:
        ckpts_path = f"{ai_model_directory}/{ckpt}"
        ai_model, ai_tokenizer = load_model(ai_model_name, model_ckpts=ckpts_path)
        chatbot = MarketplaceChatBot(ai_model_name, human_model_name, ai_model, ai_tokenizer, human_model, human_tokenizer)
        marketplace = Marketplace([chatbot])
        all_results = marketplace.generate_test_data(all_results, ckpts_list=[ckpt], output_path=output_path, rlhf_type=rlhf_type)
        output_path2 = f"{output}/test_inf_{ai_model_name}_{ckpt}.json"
        with open(output_path2, 'w') as file:
            json.dump(all_results, file)
        
        del ai_model
        del ai_tokenizer
        torch.cuda.empty_cache()

    with open(final_output_path, 'w') as file:
        json.dump(all_results, file)


def gen_new(ai_model_name, ai_model_directory, ckpts, test_data, output, index, rlhf_type):
    """This is just for human study"""

    output_path = f"{output}/test_inf_{ai_model_name}_{index}.json"
    final_output_path = f"{output}/final_test_inf_{ai_model_name}_{index}.json"

    all_results = json.load(open(test_data, 'r'))
    create_directory(output)

    for ckpt in ckpts:
        ckpts_path = f"{ai_model_directory}/{ckpt}"
        ai_model, ai_tokenizer = load_model(ai_model_name, model_ckpts=ckpts_path)
        chatbot = MarketplaceChatBot(ai_model_name, "None", ai_model, ai_tokenizer, "None", "None")
        marketplace = Marketplace([chatbot])
        all_results = marketplace.generate_new(all_results, ckpts_list=[ckpt], output_path=output_path, rlhf_type=rlhf_type)
        output_path2 = f"{output}/test_inf_{ai_model_name}_{ckpt}.json"
        with open(output_path2, 'w') as file:
            json.dump(all_results, file)
        
        del ai_model
        del ai_tokenizer
        torch.cuda.empty_cache()

    with open(final_output_path, 'w') as file:
        json.dump(all_results, file)


def main(args):
    if args.task == "gen_rlhf_train_data":
        gen_rlhf_train_data(args.ai_model[0], args.human_model, args.output, args.index, args.thres)
    elif args.task == "gen_test_data":
        gen_test_data_opt(args.ai_model, args.human_model, args.test_data, args.output, args.index, args.load_in_8bit, args.rlhf_type)
    elif args.task == "test_inference":
        test_inference_opt(args.ai_model[0], args.human_model, args.ai_model_directory, args.ai_model_ckpts, args.test_data, args.output, args.index, args.load_in_8bit, args.rlhf_type)
    else:
        gen_new(args.ai_model[0], args.ai_model_directory, args.ai_model_ckpts, args.test_data, args.output, args.index, args.rlhf_type)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Support model: llama-2-7b, llama-2-13b, llama-3-8b, llama-3-70b
    parser.add_argument('--ai_model', nargs='+', default=['llama-3-8b'], help="List of AI models or a single AI model")
    parser.add_argument('--ai_model_directory', default=None)
    parser.add_argument('--ai_model_ckpts', nargs='+', default=None, help="List of checkpoints")
    parser.add_argument('--human_model', default='llama-3.1-70b')
    parser.add_argument('--load_in_8bit', action='store_true', default=False, help="Whether the human model load in 8bit")
    parser.add_argument('--output', default='./data')
    parser.add_argument('--thres', default=15, type=int)
    parser.add_argument('--index', default=1, type=int)
    parser.add_argument('--task', choices=['gen_rlhf_train_data', "gen_test_data", "test_inference", "gen_new"], default='gen_rlhf_train_data', help="Task to be performed")

    # Specific only to test_inference
    # parser.add_argument('--rlhf_type', choices=['all', 'base', 'phs', 'fhs', 'phs_v0', 'fhs_v0'], default='all', help="Type of RLHF is used")

    parser.add_argument('--rlhf_type', nargs='+', default=['base', 'phs', 'fhs', 'phs_v0', 'fhs_v0'], help="List of different type of RLHF is used")

    # Generate test data. If the test file is specified, then use this test file instead of generating all new scenarios.
    parser.add_argument('--test_data', default='test_data/test.json')

    args = parser.parse_args()
    main(args)