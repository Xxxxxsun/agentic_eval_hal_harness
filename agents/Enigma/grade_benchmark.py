import json
import argparse



def grade_benchmark(all_preds_json_path):

    with open(all_preds_json_path, 'r') as f:
        results = [json.loads(line) for line in f]  # Process each JSON line separately
    acc_count = 0
    for result in results:
        ground_truth_path = result['challenge_path']
        with open(ground_truth_path, 'r') as f:
            ground_truth = json.load(f)
        ground_truth = ground_truth['flag']
        acc_count += int(result['model_patch'] == ground_truth)

    return len(results), acc_count, acc_count / len(results)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--all_preds_json_path', type=str, default='all_preds.json')
    args = args.parse_args()
    taks_num, acc_count, accuracy = grade_benchmark(args.all_preds_json_path)
    print(f"Total: {taks_num}, Correct: {acc_count}, Accuracy: {accuracy}")