import re

def main():
    filepath = "log/log.dynamic"

    pattern = re.compile(r"Loop time of ([\d.]+) on \d+ procs? for (\d+) steps? with (\d+) atoms")
    
    last_match = None

    with open(filepath, 'r') as file:
        for riga in file:
            match = pattern.search(riga)
            if match:
                last_match = match

    if last_match:
        total_time = float(last_match.group(1))
        num_steps = int(last_match.group(2))
        num_atoms = int(last_match.group(3))

        inference_time = total_time / (num_atoms * num_steps) * 1000 # convert in ms

    with open("model_info.yaml", 'a') as f:
        f.write(f"inference_time: {inference_time}")

if __name__ == "__main__":
    main()