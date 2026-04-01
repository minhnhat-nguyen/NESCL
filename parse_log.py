import re
import argparse

def parse_log(file_path):
    epochs = []
    best_epoch = -1
    test_result = {}

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        # Match training epoch line: epoch 0 training [time: 20.21s, train loss: 135.7317]
        train_match = re.search(r'epoch (\d+) training \[.*?train loss: ([\d.]+)\]', line)
        if train_match:
            epoch = int(train_match.group(1))
            train_loss = float(train_match.group(2))
            epochs.append({'epoch': epoch, 'train_loss': train_loss})
            continue

        # Match evaluating line
        eval_match = re.search(r'epoch (\d+) evaluating \[.*?valid_score: ([\d.]+)\]', line)
        if eval_match:
            epoch = int(eval_match.group(1))
            valid_score = float(eval_match.group(2))
            # Just ensure we augment the existing epoch dict
            if epochs and epochs[-1]['epoch'] == epoch:
                epochs[-1]['valid_score'] = valid_score
            continue

        # Match valid result
        if 'valid result:' in line and i + 1 < len(lines):
            result_line = lines[i+1]
            recall_match = re.search(r'recall@10 : ([\d.]+)', result_line)
            ndcg_match = re.search(r'ndcg@10 : ([\d.]+)', result_line)
            
            if recall_match and ndcg_match and epochs:
                epochs[-1]['recall@10'] = float(recall_match.group(1))
                epochs[-1]['ndcg@10'] = float(ndcg_match.group(1))

        # Match best valid result
        if 'best eval result in epoch' in line:
            best_match = re.search(r'best eval result in epoch (\d+)', line)
            if best_match:
                best_epoch = int(best_match.group(1))

        # Match test result
        if 'test result:' in line:
            # Using ast or regex to parse the dictionary string safely
            # e.g., {'recall@10': 0.0586, 'mrr@10': 0.13, ...}
            matches = re.findall(r"'([^']+)': ([\d.]+)", line)
            for key, val in matches:
                test_result[key] = float(val)

    return epochs, best_epoch, test_result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse RecBole log file')
    parser.add_argument('--log_file', type=str, required=True, help='Path to the log file')
    args = parser.parse_args()

    epochs, best_epoch, test_result = parse_log(args.log_file)
    
    print("=== Training Summary ===")
    print("Epoch | Train Loss | Valid NDCG@10 | Valid Recall@10")
    print("-" * 55)
    for data in epochs:
        print(f"{data['epoch']:^5} | {data.get('train_loss', 0):^10.4f} | {data.get('ndcg@10', 0):^13.4f} | {data.get('recall@10', 0):^15.4f}")

    print("\n=== Best Model ===")
    print(f"Best Validation Epoch: {best_epoch}")
    
    print("\n=== Test Results (from Best Epoch) ===")
    for metric, val in test_result.items():
        print(f"{metric}: {val}")
