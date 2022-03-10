import subprocess
from pathlib import Path


def get_diff_path(output_dir: Path):
    if not output_dir.joinpath("diff.patch").is_file():
        return output_dir.joinpath("diff.patch")
    idx = 1
    while output_dir.joinpath(f"diff{idx}.patch").is_file():
        idx += 1
    return output_dir.joinpath(f"diff{idx}.patch")

def dump_git_diff(output_dir: Path):
    output_dir.mkdir(exist_ok=True, parents=True)
    diff_path = get_diff_path(output_dir)
    cmd = [
        f"git status >> {diff_path}",
        f"git rev-parse HEAD >> {diff_path}",
        f"git diff >> {diff_path}",
    ]
    try:
        subprocess.call(" && ".join(cmd), shell=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e :
        print(e)
        pass


