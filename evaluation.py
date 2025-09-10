from main_utilities import *
from tqdm.auto import tqdm
import torch
import numpy as np
import random
import math
import os
import pandas as pd
import csv


def get_abc_new(abc: str, zero_pad=False, data_format="plain", binary=False, mode: str = "compute_gold"):
    """Unified parser: mode='compute_gold' computes the groudtruth on the fly;
       mode='read_gold_as_str' reads the groundtruth from the evaluation files (testing, validation) to do string matching.
    Returns either
      (operands_str, result_int, operation)            # v1
    or
      (operands_str, result_int, result_str, operation)  # v2
    """
    if '+' in abc:
        operation = '+'
    elif '-' in abc:
        operation = '-' 
    elif '*' in abc:
        operation = '*'
    elif "#" in abc:
        operation = ','
    else:
        print(f'operation not found, abc: {abc}')
        return None, None, None

    # Split the input string into parts
    parts = abc.split('#')
    if len(parts) != 2:
        print(f'Invalid format, expected "a+b+c...=result", got: {abc}')
        return None, None, None

    # Get the operands part (before =)
    operands_str = parts[0]
    if operands_str[0] == '$':
        operands_str = operands_str[1:]
    if operands_str.startswith('Input:\n'):
        operands_str = operands_str.split('Input:\n')[-1]
    if 'Target' in operands_str:
        operands_str = operands_str.split('\nTarget')[0]

    # Split into individual operands
    operands = [op.strip() for op in operands_str.split(operation)]
    
    # Clean up operands
    operands = [op.replace(' ', '') for op in operands]
    
    if binary:
        # Convert all operands to binary and sum
        result = sum(int(op, 2) for op in operands)
        return operands_str, result, operation

    if zero_pad:
        operands = [remove_zero_pad(op) for op in operands]

    # version 1: compute the result
    if mode == "compute_gold":
        if operation == '+':
            result = sum(int(op) for op in operands)
        elif operation == '-':
            result = int(operands[0]) - sum(int(op) for op in operands[1:])
        elif operation == '*':
            result = 1
            for op in operands:
                result *= int(op)
        elif operation == ',':
            if len(operands) != 2:
                raise ValueError(f"Invalid number of operands for operation ',': {len(operands)}")
            if operands[0] > operands[1]:
                result = '>'
            elif operands[0] < operands[1]:
                result = '<'
            else:
                result = '='
        else:
            raise ValueError(f"Unsupported operation: {operation}")

        return operands_str, result, operation
    # version 2: read the groundtruth from the evaluation files
    if mode == "read_gold_as_str":
        # parts[1] is the result part, which may contain a trailing '$' or newline
        result_str = parts[1].strip()
        if result_str.endswith('\n'):
            result_str = result_str[:-1].strip()
        if result_str.endswith('$'):
            result_str = result_str[:-1].strip()
        if data_format == "reverse":
            sign = ''
            if result_str.startswith('-') or result_str.startswith('+'):
                sign = result_str[0]
                result_str = result_str[1:]
            result_str = sign + result_str[::-1]  # reverse the result string if needed

        return operands_str, result_str, operation

_precomputed_batches = {}
def prepare_addition_batches(config, encode, num_digit=3, zero_pad=False, binary=False, data_type='binary',
                               operator='+', data_format='plain', add_space=False, simple=False,
                               mode: str = "compute_gold"):
    device = config['device']
    test_batch_size = config['test_batch_size'] if 'test_batch_size' in config.keys() else 128
    start = config['start'] if 'start' in config.keys() else "FILE:prompt/prompt_addition_pad_test_0.01.txt"
    print(f"Preparing batches from: {start}")

    if start.startswith('FILE:'):  # start is just the test file path
        with open(start[5:], 'r', encoding='utf-8') as f:
            lines = [line.rstrip() for line in f]
    else:
        lines = start.splitlines()

    total = len(lines)
    print(f'Preparing batches for {total} examples from: {start}')

    # Process all lines and group by prompt length
    prompt_dict = {}
    for line in lines:
        # split off gold answer
        # e.g. line = "123+456=579"
        prompt_str = line.split('#')[0] + '#'  # "123+456="
        prompt_ids = encode(prompt_str)
        x = torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, ...]
        prompt_length = x.size(1)

        # parse out gold for evaluation later
        operands, result, op = get_abc_new(
            line,
            zero_pad=zero_pad,
            data_format=data_format,
            binary=binary,
            mode=mode
        )

        entry = (x, operands, result)
        prompt_dict.setdefault(prompt_length, []).append(entry)

    # Construct batches of prompts
    batch_list = []
    for prompt_length in prompt_dict.keys():
        input_tuple_list = prompt_dict[prompt_length]
        for batch_idx in range(math.ceil(len(input_tuple_list) / test_batch_size)):
            batch_list.append(input_tuple_list[batch_idx * test_batch_size:(batch_idx + 1) * test_batch_size])

    print(f'Created {len(batch_list)} batches')

    # Cache the batches using a hash of the configuration
    config_hash = hash(frozenset({k: str(v) for k, v in config.items() if k != 'device'}.items()))
    batch_key = f"{config_hash}_{data_type}_{operator}_{num_digit}_{zero_pad}_{data_format}_{add_space}"
    _precomputed_batches[batch_key] = (batch_list, total)

    return batch_list, total

# Modified evaluation function that uses pre-created batches
def evaluate_addition_precomputed(config, model, ctx, decode, batch_list, total,
                                  verbose=False, num_digit=3, zero_pad=False, data_format='plain',
                                  add_space=False, operator='+', verbose_correct=False, analyze=False, mode: str = "compute_gold"):
    model.eval()
    device = config['device']
    max_new_tokens = config['max_new_tokens'] if 'max_new_tokens' in config.keys() else num_digit+2
    temperature = config['temperature'] if 'temperature' in config.keys() else 0.8
    top_k = config['top_k'] if 'top_k' in config.keys() else 200

    if add_space:
        max_new_tokens = 2 * num_digit + 3

    correct = 0

    if analyze:
        # analyze various metrics
        error_dict = {'y': [], 'y_hat': [], 'accuracy_eps0': [], 'accuracy_eps5e-4': [],
                      'accuracy_eps5e-3': [], 'mse': [], 'normalized_mse': [],
                      'digit_wise_difference': [], 'incorrect_digit_count': []}
        list_not_num = []
        list_outlier_num = []
    op = operator
    correct_examples = []
    incorrect_examples = []
    print(f"Max number of tokens {max_new_tokens}.")
    for batch_idx in tqdm(range(len(batch_list))):
        batch = batch_list[batch_idx]
        x_list = [input_tuple[0] for input_tuple in batch]
        x = torch.cat(x_list, dim=0)

        # Run generation
        with torch.no_grad():
            with ctx:
                eos_id = config['eos_id']
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                outcome_list = [decode(y_i.tolist()) for y_i in y]

                for i, outcome in enumerate(outcome_list):
                    _, operands, result = batch[i]
                    
                    if mode == "compute_gold":
                        c_hat = outcome.split('#')[1].split('$')[0].strip()

                        if zero_pad:
                            c_hat = remove_zero_pad(c_hat)

                        # plain addition
                        c_hat = c_hat.split('\n')[0]

                        if data_format == "reverse":
                            c_hat = reverse_string(c_hat)

                        if add_space:
                            c_hat = c_hat.replace(' ', '')

                        if is_number(c_hat):
                            if '.' in c_hat:
                                c_hat = float(c_hat)
                            else:
                                c_hat = int(c_hat)
                        else:  # c_hat is not a number
                            result = str(result)

                    if mode == "read_gold_as_str":
                        c_hat = outcome.split('#')[1].split('$')[0].strip()

                        if data_format == "reverse":
                            sign = ''
                            if c_hat.startswith('-') or c_hat.startswith('+'):
                                sign = c_hat[0]
                                c_hat = c_hat[1:]
                            c_hat = sign + c_hat[::-1]

                    # Check correctness
                    if result == c_hat:
                        correct += 1
                        correct_examples.append((operands, result, outcome, c_hat))
                        if verbose_correct:
                            print('outputs(o): ', outcome)
                            print(f'correct: {operands}={result}')
                    else:
                        incorrect_examples.append((operands, result, outcome, c_hat))
                        if verbose:
                            print('outputs(x): ', outcome)
                            print(f'wrong  : {operands}={c_hat}')
                            print(f'correct: {operands}={result}')
                    # Calculate metrics if analyzing
                    if analyze:
                        error_dict['y'].append(result)
                        error_dict['y_hat'].append(c_hat)

                        metric_types = ['mse', 'normalized_mse', 'digit_wise_difference', 'incorrect_digit_count']
                        for metric_type in metric_types:
                            error, list_not_num, list_outlier_num = get_error_metric(result, c_hat, metric_type, eps=config.get('eps', 0),
                                                                                    list_not_num=list_not_num, list_outlier_num=list_outlier_num)
                            error_dict[f'{metric_type}'].append(error)

                        error, _, _ = get_error_metric(result, c_hat, 'accuracy', eps=0, list_not_num=list_not_num, list_outlier_num=list_outlier_num)
                        error_dict[f'accuracy_eps0'].append(error * 100)
                        error, _, _ = get_error_metric(result, c_hat, 'accuracy', eps=5e-4, list_not_num=list_not_num, list_outlier_num=list_outlier_num)
                        error_dict[f'accuracy_eps5e-4'].append(error * 100)
                        error, _, _ = get_error_metric(result, c_hat, 'accuracy', eps=5e-3, list_not_num=list_not_num, list_outlier_num=list_outlier_num)
                        error_dict[f'accuracy_eps5e-3'].append(error * 100)

    accuracy = correct / total * 100
    print(f"accuracy of {total} examples: {correct}/{total} ({accuracy}%)")

    accuracy_dictionary = {}
    if analyze:
        error_df = pd.DataFrame(error_dict)
        result_dir = config.get('result_dir')
        if result_dir is None:
            result_dir = get_results_dir(config)
        error_df.to_csv(os.path.join(result_dir, 'error_df.csv'), index=False)

        error_mean_dict = {
            metric_type: np.nanmean(error_dict[f'{metric_type}'])
            for metric_type in ['accuracy_eps0', 'accuracy_eps5e-4', 'accuracy_eps5e-3',
                               'mse', 'normalized_mse', 'digit_wise_difference', 'incorrect_digit_count']
        }
        error_mean_dict['num_not_num'] = len(list_not_num) / len(metric_types)
        error_mean_dict['num_outlier_num'] = len(list_outlier_num) / len(metric_types)
        error_mean_dict['median_mse'] = error_df.mse.median()
        error_mean_dict['median_normalized_mse'] = error_df.normalized_mse.median()
        accuracy_dictionary.update(error_mean_dict)

    model.train()
    return accuracy, accuracy_dictionary, correct_examples, incorrect_examples

# Keep the original function for backward compatibility, but make it use the new functions
def evaluate_addition_batch(config, model, ctx, encode, decode, verbose=False, num_digit=3, zero_pad=False, 
                          data_type='binary', operator='+', 
                          data_format='plain', add_space=False, verbose_correct=False, analyze=False, mode: str = "compute_gold"):
    config_hash = hash(frozenset({k: str(v) for k, v in config.items() if k != 'device'}.items()))
    batch_key = f"{config_hash}_{data_type}_{operator}_{num_digit}_{zero_pad}_{data_format}_{add_space}"
    
    if batch_key in _precomputed_batches:
        print("Using precomputed batches")
        batch_list, total = _precomputed_batches[batch_key]
    else:
        print("Creating new batches")
        batch_list, total = prepare_addition_batches(
            config, encode, num_digit=num_digit, zero_pad=zero_pad,
            data_type=data_type, operator=operator, data_format=data_format, add_space=add_space, mode=mode
        )

    # Evaluate using the batches
    return evaluate_addition_precomputed(
        config, model, ctx, decode, batch_list, total, verbose=verbose,
        num_digit=num_digit, zero_pad=zero_pad, data_format=data_format,
        add_space=add_space, operator=operator, verbose_correct=verbose_correct, analyze=analyze, mode=mode
    )

def evaluate_multiple_files(config, model, ctx, encode, decode, test_file, iter_num, result_dir,
                          verbose=False, num_digit=3, zero_pad=False,
                          data_type='binary', operator='+', data_format='plain', add_space=False, analyze=False, mode: str = "compute_gold"):
    """
    Evaluate model on multiple test files and store results.
    Args:
        test_files: List of test file paths
        iter_num: Current iteration number
        result_dir: Directory to store results
    Returns:
        dict: Dictionary containing accuracies for each test file
    """
    
    # Get test file name without path and extension
    test_name = os.path.splitext(os.path.basename(test_file))[0]
    
    # Set the current test file as start
    config['start'] = f"FILE:{test_file}"
    
    # Run evaluation
    accuracy, metrics, correct, incorrect = evaluate_addition_batch(
        config, model, ctx, encode=encode, decode=decode,
        verbose=verbose, num_digit=num_digit, zero_pad=zero_pad,
        data_type=data_type, operator=operator,
        data_format=data_format, analyze=analyze, mode=mode
    )
    
    # Path for this test file's results
    results_file = os.path.join(result_dir, f'{test_name}_results.csv')
    
    # Combine correct and incorrect examples and sort by operands to maintain consistent order
    all_examples = correct + incorrect
    all_examples.sort(key=lambda x: x[0])  # Sort by operands
    
    # Create new DataFrame with operands and actual results
    new_df = pd.DataFrame({
        'operands': [ex[0] for ex in all_examples],
        'actual': [ex[1] for ex in all_examples],
        f'pred_iter_{iter_num}': [ex[3] for ex in all_examples]
    })
    
    # Read existing results if file exists and merge
    if os.path.exists(results_file):
        old_df = pd.read_csv(results_file)
        # # Merge based on operands, keeping all predictions
        # if 'operands' in old_df.columns:
        #     merged_df = pd.merge(old_df, new_df, on=['operands', 'actual'], how='outer')
        # else:
        #     merged_df = new_df
        # ── Normalize keys so they truly match ──
        for df in (old_df, new_df):
            # strip whitespace from the operands strings
            df['operands'] = df['operands'].str.strip()
            df['actual']   = df['actual'].str.strip()

        merged_df = pd.merge(
            old_df, new_df,
            on=['operands', 'actual'],
            how='outer'
        )
    else:
        merged_df = new_df
    
    # Save results
    merged_df.to_csv(results_file, index=False)
    
    # Save accuracy separately in a summary file
    accuracy_file = os.path.join(result_dir, f'{test_name}_accuracy.csv')
    if os.path.exists(accuracy_file):
        acc_df = pd.read_csv(accuracy_file)
    else:
        acc_df = pd.DataFrame(columns=['iteration', 'accuracy'])
    
    # Add new accuracy
    new_row = pd.DataFrame({'iteration': [iter_num], 'accuracy': [accuracy]})
    acc_df = pd.concat([acc_df, new_row], ignore_index=True)
    acc_df.to_csv(accuracy_file, index=False)
    
    return test_name, accuracy, metrics, correct, incorrect