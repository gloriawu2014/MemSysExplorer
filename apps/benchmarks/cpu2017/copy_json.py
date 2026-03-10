import os
import shutil

def copy_json_files(src_dir, dest_dir):
    for root, dirs, files in os.walk(src_dir):
        json_files = [f for f in files if f.endswith('.json')]

        for json_file in json_files:
            src_file = os.path.join(root, json_file)

            relative_path = os.path.relpath(root, src_dir)
            dest_folder = os.path.join(dest_dir, relative_path)

            os.makedirs(dest_folder, exist_ok=True)
            
            dest_file = os.path.join(dest_folder, json_file)

            shutil.copy(src_file, dest_file)

if __name__ == "__main__":
    src_dir = '/home/gwu28/MemSysExplorer/apps/benchmarks/cpu2017/spec_runs'
    dest_dir = '/home/gwu28/MemSysExplorer/apps/benchmarks/cpu2017/perf_l1_data'

    copy_json_files(src_dir, dest_dir)