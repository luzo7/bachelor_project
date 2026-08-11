import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import os
import shutil
import warnings
import re
import cocopp

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "coco_data", "exdata")
UNIFIED_DIR = os.path.join(BASE_DIR, "coco_data", "exdata_unified")
REPORT_DIR = os.path.join(BASE_DIR, "coco_data", "coco_reports", "report_no_baseline")

BASELINE_ALGORITHMS = [
    # "bbob-mixint/2019/RANDOMSEARCH", 
    # "bbob-mixint/2019/DE-scipy", 
    # "bbob-mixint/2019/CMA-ES-pycma", 
    # "bbob-mixint/2019/TPE-hyperopt"
]

def get_base_name(folder_name):
    """
    Remove instance suffix (es. '-001', '-002') 
    to group different seeds under the same algorithm.
    Es: 'SAASBO_f15_40D-002' -> 'SAASBO_f15_40D'
    """
    return re.sub(r'-\d+$', '', folder_name)

def main():
    if not os.path.exists(RAW_DIR):
        print(f"Errore: {RAW_DIR} not found.")
        return

    print("Data aggregation...")
    
    if os.path.exists(UNIFIED_DIR):
        shutil.rmtree(UNIFIED_DIR)
    os.makedirs(UNIFIED_DIR, exist_ok=True)

    folders = [d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d))]
    
    # Grouping by algorithm
    # { 'MyAlgo_f15': ['MyAlgo_f15', 'MyAlgo_f15-001'], ... }
    groups = {}
    for folder in folders:
        base_name = get_base_name(folder)
        if base_name not in groups:
            groups[base_name] = []
        groups[base_name].append(folder)

    print(f"Found {len(groups)} distinct algorithms: {list(groups.keys())}")

    # Creating the structure for each algorithm
    parent_dirs = []
    
    for base_name, instances in groups.items():
        # Create a father folder for each algorithm
        target_algo_dir = os.path.join(UNIFIED_DIR, base_name)
        os.makedirs(target_algo_dir, exist_ok=True)
        parent_dirs.append(target_algo_dir) # Saving the path for cocopp

        # Copying seeds inside
        for folder in instances:
            src_path = os.path.join(RAW_DIR, folder)
            dest_path = os.path.join(target_algo_dir, folder)
            
            shutil.copytree(src_path, dest_path)
            
            # Filling suite param beacuase there's a bug that writes garbage in it
            for file in os.listdir(dest_path):
                if file.endswith(".info"):
                    info_path = os.path.join(dest_path, file)
                    with open(info_path, 'r', errors='ignore') as f:
                        content = f.read()
                    
                    content = re.sub(r"suite\s*=\s*.*?(?=,\s*funcId|\n\s*%|$)", "suite = 'bbob-mixint'", content, flags=re.DOTALL)
                    
                    with open(info_path, 'w', encoding='utf-8') as f:
                        f.write(content)

    print(f"Unified data saved in: {UNIFIED_DIR}")

    # COCO plots generation
    print(f"\nGenerating comparative plots...")
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    try:
        args_str = ["-o", REPORT_DIR] + parent_dirs + BASELINE_ALGORITHMS
        cocopp.main(args_str)
        print(f"\nReport successfully generated!")
    except Exception as e:
        print(f"Error while post-processing: {e}")

if __name__ == "__main__":
    main()