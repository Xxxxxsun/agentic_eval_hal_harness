import json
import argparse
import subprocess


dataset2root = {
    'nyu_ctf_test': {"root_dir": "/home/boyiwei/SWE-agent-datasets/NYU_CTF_Bench/", "json_file": "/home/boyiwei/SWE-agent-datasets/NYU_CTF_Bench/test_dataset.json"},
    'nyu_ctf_train': {"root_dir": "/home/boyiwei/SWE-agent-datasets/NYU_CTF_Bench/", "json_file": "/home/boyiwei/SWE-agent-datasets/NYU_CTF_Bench/development_dataset.json"},
    'cybench': {"root_dir": "/home/boyiwei/SWE-agent-datasets/cybench/", "json_file": "/home/boyiwei/SWE-agent-datasets/cybench/cybench.json"},
    'intercode_ctf': {"root_dir": "/home/boyiwei/SWE-agent-datasets/Intercode_CTF/", "json_file": "/home/boyiwei/SWE-agent-datasets/Intercode_CTF/intercode_dataset.json"},
}

def get_examples(root_dir, json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    repo_paths = [root_dir + data[elem]['path'] for elem in data]
    data_paths = [root_dir + data[elem]['path'] + '/challenge.json' for elem in data]
    return repo_paths, data_paths


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--dataset', type=str, default='nyu_ctf_test')
    args.add_argument('--model_name', type=str, default='gpt4o')
    args.add_argument('--cost_limit', type=float, default=2)
    args.add_argument('--max_input_tokens', type=int, default=120000)
    args.add_argument('--max_output_tokens', type=int, default=3000)
    
    args = args.parse_args()
    
    root_dir = dataset2root[args.dataset]['root_dir']
    json_file = dataset2root[args.dataset]['json_file']
    repo_paths, data_paths = get_examples(root_dir, json_file)
    for i in range(len(repo_paths)):
        repo_path = repo_paths[i]
        data_path = data_paths[i]
        with open(data_path, 'r') as f:
            data = json.load(f)
        category = data['category']
        config_file = f"config/ctf/ctf_{category}.yaml"
        subprocess.run([
            'python', 'run.py',
            '--model_name', args.model_name,
            '--ctf',
            '--image_name', 'sweagent/enigma:latest',
            '--data_path', data_path,
            '--repo_path', repo_path,
            '--config_file', config_file,
            '--per_instance_cost_limit', f'{args.cost_limit}',
            '--max_input_tokens', f'{args.max_input_tokens}',
            '--max_output_tokens', f'{args.max_output_tokens}',
        ])