import argparse
import datetime
import os
from pathlib import Path
import stat
import subprocess

from git import Repo
import numpy as np

default_log_dir = f"/work/dlclarge1/lagandua-vlawm/lumos/logs"
if default_log_dir == "/tmp":
    print("CAUTION: logging to /tmp")
parser = argparse.ArgumentParser(description="Parse slurm parameters and hydra config overrides")

parser.add_argument("--script", type=str, default="/work/dlclarge1/lagandua-vlawm/lumos/slurm_scripts/sbatch_train.sh")
parser.add_argument("--train_file", type=str, default="/work/dlclarge1/lagandua-vlawm/lumos/scripts/train_wm.py")
parser.add_argument("-l", "--log_dir", type=str, default=default_log_dir)
parser.add_argument("-j", "--job_name", type=str, default="patch_wm_job")
parser.add_argument("-g", "--gpus", type=int, default=1)
parser.add_argument("--mem", type=int, default=0)  # 0 means no memory limit
parser.add_argument("-v", "--venv", type=str, default="lumos")
parser.add_argument("-p", "--partition", type=str, default="rldlc2_gpu-l40s")
parser.add_argument("-x", "--exclude", type=str, help="Comma-separated list of nodes to exclude (e.g. 01,02,05)")
parser.add_argument("--no_clone", action="store_true", default=True, help="If set, do not clone the repo to the logdir")
args, unknownargs = parser.parse_known_args()

# Validate required parameters
if args.venv is None:
    raise ValueError("--venv parameter is required. Please specify the conda environment name.")

assert np.all(["gpu" not in arg for arg in unknownargs])
assert np.all(["hydra.run.dir" not in arg for arg in unknownargs])
assert np.all(["log_dir" not in arg for arg in unknownargs])
assert np.all(["hydra.sweep.dir" not in arg for arg in unknownargs])

log_dir = Path(args.log_dir).absolute() / f'{datetime.datetime.now().strftime("%Y-%m-%d/%H-%M-%S")}_{args.job_name}'
os.makedirs(log_dir)
args.script = Path(args.script).absolute()
args.train_file = Path(args.train_file).absolute()


def create_git_copy(repo_src_dir, repo_target_dir):
    repo = Repo(repo_src_dir)
    repo.clone(repo_target_dir)
    orig_cwd = os.getcwd()
    try:
        os.chdir(repo_target_dir)
        # Install using setup_local.py (no dependencies, faster for SLURM)
        result = subprocess.run(
            ["python", "setup_local.py", "develop", "--install-dir", "."], capture_output=True, text=True, check=True
        )
        print(f"Successfully installed lumos in development mode: {repo_target_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error installing package: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        raise
    finally:
        os.chdir(orig_cwd)


if not args.no_clone:
    repo_src_dir = Path(__file__).absolute().parents[1]
    repo_target_dir = log_dir / "lumos"
    create_git_copy(repo_src_dir, repo_target_dir)

    args.script = repo_target_dir / os.path.relpath(args.script, repo_src_dir)
    args.train_file = repo_target_dir / os.path.relpath(args.train_file, repo_src_dir)

if args.partition == "test":
    args.partition = "testdlc2_gpu-l40s"

args.time = "1-00:00"
if args.partition == "testdlc2_gpu-l40s":
    args.time = "01:00:00"

job_opts = {
    "script": f"{args.script.as_posix()} {args.venv} {args.train_file.as_posix()} {log_dir.as_posix()} {args.gpus} {' '.join(unknownargs)}",
    "partition": args.partition,
    "ntasks-per-node": str(args.gpus),
    "cpus-per-task": str(args.gpus * 8),
    "gres": f"gpu:{args.gpus}",
    "nodes": str(1),
    "output": os.path.join(log_dir, "%x.%N.%j.out"),
    "error": os.path.join(log_dir, "%x.%N.%j.err"),
    "job-name": args.job_name,
    "mail-type": "END,FAIL",
    "time": args.time,
}

# Only add memory specification if it's greater than 0
if args.mem > 0:
    job_opts["mem"] = args.mem

if args.exclude is not None:
    job_opts["exclude"] = ",".join(map(lambda x: f"dlc2gpu{int(x):02d}", args.exclude.split(",")))


def submit_job(job_info):
    # Construct sbatch command
    slurm_cmd = ["sbatch"]
    for key, value in job_info.items():
        # Check for special case keys
        if key == "script":
            continue
        slurm_cmd.append(f"--{key}={value}")
    slurm_cmd.append(job_info["script"])
    print("Generated slurm batch command: '%s'" % slurm_cmd)

    # Run sbatch command as subprocess.
    sbatch_output = None
    try:
        sbatch_output = subprocess.check_output(slurm_cmd)
        create_resume_script(slurm_cmd)
        print(sbatch_output.decode("utf-8"))
    except subprocess.CalledProcessError as e:
        # Print error message from sbatch for easier debugging, then pass on exception
        if sbatch_output is not None:
            print("ERROR: Subprocess call output: %s" % sbatch_output.decode("utf-8"))
        print(f"ERROR: sbatch command failed with return code {e.returncode}")
        if e.output:
            print(f"ERROR: Command output: {e.output.decode('utf-8')}")
        raise e


def create_resume_script(slurm_cmd):
    file_path = os.path.join(log_dir, "resume_training.sh")
    with open(file_path, "w") as file:
        file.write("#!/bin/bash\n")
        file.write(" ".join(slurm_cmd))
    st = os.stat(file_path)
    os.chmod(file_path, st.st_mode | stat.S_IEXEC)


submit_job(job_opts)
