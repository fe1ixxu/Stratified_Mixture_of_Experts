import argparse
import datetime
import os
import socket
import subprocess
from typing import Callable, List, MutableMapping, Optional
from urllib.parse import urlparse


def csv_str_list(x):
    return [y.strip() for y in x.split(",")]


def get_args(add_extra_options_func=None, input_args: Optional[List[str]] = None):
    """
    input_args (List[str]): strings to parse, defaults to sys.argv
    """
    parser = argparse.ArgumentParser("Script for launching hyperparameter sweeps ")
    parser.add_argument("--grid", help="grid function we used", default=None)
    parser.add_argument("--pair", help="language direction", default=None)

    parser.add_argument("-d", "--data", help="path to data directory")
    parser.add_argument(
        "-p",
        "--prefix",
        required=True,
        help="save checkpoints and logs in <checkpoints-dir>/<prefix>.<save_dir_key>",
    )
    parser.add_argument(
        "-t",
        "--num-trials",
        required=True,
        type=int,
        help="number of random hyperparam configurations to try (-1 for grid search)",
    )
    parser.add_argument(
        "-g", "--num-gpus", type=int, required=True, help="number of GPUs per node"
    )
    parser.add_argument(
        "-n",
        "--num-nodes",
        type=int,
        default=1,
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "--local-experts",
        type=int,
        default=1,
        help="number of local experts",
    )
    parser.add_argument(
        "--langs",
        type=str,
        default="en_XX",
        help="list of languages used in the language model, separate by commas",
    )
    parser.add_argument(
        "--update-freq",
        type=int,
        default=1,
        help="update freq",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--baseline-model", help="path to baseline model from which to resume training"
    )
    parser.add_argument(
        "--force-checkpoints-dir", help="force using a given checkpoint dir"
    )
    parser.add_argument(
        "--resume-failed",
        action="store_true",
        help="resume any runs that failed (assumes --num-trials and --seed are the same)",
    )
    parser.add_argument(
        "--resume-finished",
        action="store_true",
        help="force any runs that finished to begin again (uncommon)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="output only a list of actions to perform without performing them",
    )
    parser.add_argument("--local", action="store_true", help="run job locally")
    parser.add_argument("--debug", action="store_true", help="debug")
    parser.add_argument("--script", default="train.py", help="script to launch")
    parser.add_argument(
        "--python", default="python", help="path to nonstandard python binary"
    )

    # identify cluster
    hostname = socket.gethostname()
    is_azure = is_aws = is_fair = is_tpu = is_rsc = False
    if os.path.exists("/shared/home"):
        cluster = "azure"
        is_azure = True
    elif "p3dn" in hostname or "p4d" in hostname or os.path.exists("/fsx"):
        cluster = "aws"
        is_aws = True
    elif "fair" in hostname:
        cluster = "fair"
        is_fair = True
    elif "rsc" in hostname:
        cluster = "rsc"
        is_rsc = True
    else:
        try:
            import torch_xla  # noqa

            cluster = "tpu"
            is_tpu = True
        except ImportError:
            raise NotImplementedError(hostname)

    # configure --checkpoints-dir, CPU settings, etc.
    # we also configure cpu-bind options to improve all-to-all performance, especially on A100
    if is_fair or is_aws or is_azure or is_rsc:
        default_backend = "slurm"
        default_partition = None
        default_local_checkpoints_dir = None
        if is_azure:
            prefix = "/shared/home"
            cpus_per_task = 12  # assumes 8 tasks per node
            default_tensorboard_logdir = None  # will default to save_dir/tb
            default_cpu_bind = "mask_cpu:ffffff000000,ffffff000000,ffffff,ffffff,ffffff000000000000000000,ffffff000000000000000000,ffffff000000000000,ffffff000000000000"
            azure_upload_path = os.environ.get("AZURE_BLOB_SAS_URL", "")
            if azure_upload_path != "":
                # write checkpoints to local scratch storage on each node
                default_local_checkpoints_dir = os.path.join(
                    "/mnt/scratch",
                    os.environ["USER"],
                    "checkpoints",
                    str(datetime.date.today()),
                )
                # then copy them to Azure blob storage
                o = urlparse(azure_upload_path)
                o = o._replace(
                    path=os.path.join(
                        o.path, os.environ["USER"], str(datetime.date.today())
                    )
                )
                azure_upload_path = o.geturl()
                parser.add_argument(
                    "--azure-upload-path",
                    default=azure_upload_path,
                    help="Azure blob storage SAS URL",
                )
                # if needed, create a container for this user on the Azure blob account
                cmd = [
                    "/shared/home/shru/bin/azcopy_linux_amd64_10.12.1/azcopy",
                    "make",
                    o._replace(path=os.path.dirname(o.path)).geturl(),
                ]
                res = subprocess.run(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
        elif is_aws:
            prefix = "/fsx"
            cpus_per_task = 12  # assumes 8 tasks per node
            default_tensorboard_logdir = os.path.join(
                prefix,
                os.environ["USER"],
                "tensorboard_logs",
                str(datetime.date.today()),
            )
            default_cpu_bind = "mask_cpu:000000ffffff000000ffffff,000000ffffff000000ffffff,000000ffffff000000ffffff,000000ffffff000000ffffff,ffffff000000ffffff000000,ffffff000000ffffff000000,ffffff000000ffffff000000,ffffff000000ffffff000000"
        elif is_fair:
            default_partition = "learnfair"
            prefix = "/checkpoint"
            cpus_per_task = 10  # assumes 8 tasks per node
            default_cpu_bind = "map_ldom:0,0,0,0,1,1,1,1"
            default_tensorboard_logdir = os.path.join(
                prefix,
                os.environ["USER"],
                "tensorboard_logs",
                str(datetime.date.today()),
            )
        elif is_rsc:
            default_partition = "debug"
            prefix = "/checkpoint/xlmg"
            cpus_per_task = 32  # assumes 8 tasks per node
            default_cpu_bind = "none"
            # default_cpu_bind = "mask_cpu:ffff0000000000000000000000000000ffff000000000000,ffff0000000000000000000000000000ffff000000000000,ffff0000000000000000000000000000ffff0000,ffff0000000000000000000000000000ffff0000,ffff0000000000000000000000000000ffff0000000000000000000000000000,ffff0000000000000000000000000000ffff0000000000000000000000000000,ffff0000000000000000000000000000ffff00000000000000000000,ffff0000000000000000000000000000ffff00000000000000000000"
            default_tensorboard_logdir = os.path.join(
                prefix,
                os.environ["USER"],
                "tensorboard_logs",
                str(datetime.date.today()),
            )
        else:
            raise Exception
        parser.add_argument(
            "--checkpoints-dir",
            default=(
                os.path.join(prefix, os.environ["USER"], str(datetime.date.today()))
                if is_fair or is_rsc
                else os.path.join(
                    prefix,
                    os.environ["USER"],
                    "checkpoints",
                    str(datetime.date.today()),
                )
            ),
            help="save checkpoints and logs in <checkpoints-dir>/<prefix>.<save_dir_key>",
        )
        parser.add_argument(
            "--log-dir",
            default=None,
            help="save logs in <log-dir>/<prefix>.<save_dir_key> if specified; else <checkpoints-dir>",
        )
        parser.add_argument(
            "--skip-create-save-dir",
            action="store_true",
            help="skip creating save dir when launching job; useful if launching jobs on rsccpu",
        )
        parser.add_argument("--cpus-per-task", type=str, default=str(cpus_per_task))
        parser.add_argument("--cpu-bind", default=default_cpu_bind)
        parser.add_argument(
            "--local-checkpoints-dir",
            default=default_local_checkpoints_dir,
            help="node-local directory for saving checkpoints",
        )
        parser.add_argument(
            "--tensorboard-logdir",
            default=default_tensorboard_logdir,
            help="save tensorboard logs in <tensorboard-logdir>/<prefix>.<save_dir_key>",
        )
    elif is_tpu:
        default_backend = "tpu"
        parser.add_argument(
            "--checkpoints-dir",
            default=os.path.join(
                "/mnt/fairseq_data",
                os.environ["USER"],
                "checkpoints",
                str(datetime.date.today()),
            ),
            help="save checkpoints and logs in <checkpoints-dir>/<prefix>.<save_dir_key>",
        )
    else:
        default_backend = "fblearner"
        parser.add_argument(
            "--checkpoints-dir",
            default=os.path.join(
                "/mnt/vol/gfsai-east/ai-group/users",
                os.environ["USER"],
                "checkpoints",
                str(datetime.date.today()),
            ),
            help="save checkpoints and logs in <checkpoints-dir>/<prefix>.<save_dir_key>",
        )
        parser.add_argument(
            "--log-main-dir",
            default=None,
            help="dir to store log in addition to stdout. If this "
            "is not set, it will be set to args.checkpoints_dir",
        )

    parser.add_argument(
        "--backend",
        choices=["fblearner", "chronos", "slurm", "tpu"],
        default=default_backend,
    )
    parser.add_argument("--cluster", default=cluster)

    # FBLearner params
    parser.add_argument("--entitlement", help="entitlement to use", default="gpu_fair")
    parser.add_argument(
        "--run-as-secure-group", help="secure group to use", default="oncall_fairseq"
    )
    parser.add_argument(
        "--capabilities",
        help="capabilities: e.g. GPU_V100_HOST,GPU_V100_32G_HOST",
        type=csv_str_list,
        default=None,
    )
    # for manifold in fblearner
    parser.add_argument(
        "--manifold-max-parallel",
        default=8,
        type=int,
        help="set ManifoldPathHandler max_parallel download number",
    )
    parser.add_argument(
        "--manifold-timeout-sec",
        default=1800,
        type=int,
        help="set ManifoldPathHandler timeout seconds",
    )
    parser.add_argument(
        "--manifold-has-user-data",
        default=None,
        type=lambda x: x.lower() not in ("no", "false", "f", "n", "0")
        if x is not None
        else None,
        help="set ManifoldPathHandler has_user_data option",
    )
    parser.add_argument(
        "--manifold-num-retries",
        default=15,
        type=int,
        help="set ManifoldPathHandler num_retries option",
    )
    parser.add_argument(
        "--manifold-ttl",
        default=None,
        type=int,
        help="A manifold  resource's time-to-live, applied to all manifold written resources. By default, there is no TTL.",
    )
    # for manifold in fblearner

    # Chronos params
    parser.add_argument("--hostgroup", help="hostgroup to use")
    parser.add_argument("--host-filter", help="host filter")
    parser.add_argument("--fbpkg", help="use the given fbpkg")
    parser.add_argument("--build-only", action="store_true")

    # Slurm params
    parser.add_argument(
        "--salloc", action="store_true", help="run agaist current allocation"
    )
    parser.add_argument(
        "--partition",
        help="partition to run on",
        default=default_partition,
    )
    parser.add_argument("--reservation", help="reservation to run on")
    parser.add_argument(
        "--exclusive", action="store_true", help="if set, get exclusive host"
    )
    parser.add_argument(
        "--dep",
        metavar="JOBID",
        type=int,
        help="add JOBID as a dependency (i.e., wait for it to finish)",
    )
    parser.add_argument(
        "--sequential", action="store_true", help="schedule jobs to run sequentially"
    )
    parser.add_argument(
        "--time", default="4320", help="expected job duration in minutes"
    )
    parser.add_argument("--mem", "--mem", help="memory to request")
    parser.add_argument("--container-image")
    parser.add_argument("--container-save")
    parser.add_argument(
        "--constraint",
        metavar="CONSTRAINT",
        help='gpu constraint, if any. e.g. "volta"',
    )
    parser.add_argument("--comment", help="comment string")
    parser.add_argument(
        "--snapshot-code",
        action="store_true",
        default=False,
        help="Flag for creating a snapshot of training code while creating slurm job,"
        ' path is "./slurm_snapshot_code/<TIME_ISO_FORMAT/>:", '
        "can find time from comment of slurm job.",
    )
    parser.add_argument(
        "--snapshot-root",
        type=str,
        default=".",
        help="root path for saving the snapshot code.",
    )
    parser.add_argument(
        "--snapshot-recurse-dirs",
        default="fairseq,fairseq_cli",
        help="comma-separated directories from where to recursively copy *.py, *.so and *.yaml files",
    )
    parser.add_argument(
        "--no-tensorboard", action="store_true", help="disable tensorboard logging"
    )
    parser.add_argument("--no-wandb", action="store_true", help="disable WandB logging")
    parser.add_argument(
        "--post-steps",
        nargs="+",
        help="additional steps to execute after the primary job is complete. "
        "this can be a file with the steps, or a string. some placeholders such as "
        "{job_dir} will be replaced",
    )
    parser.add_argument(
        "--use-jobarray", action="store_true", help="Submit sweep as job-array"
    )
    parser.add_argument(
        "--jobarray-name",
        type=str,
        default=None,
        help="Folder name for job-array. Defaults to <jobarray_timestamp>",
    )

    # GCP params
    parser.add_argument("--tpu", help="tpu to use")

    if add_extra_options_func is not None:
        add_extra_options_func(parser)
    args = parser.parse_args(input_args)
    if args.use_jobarray:
        if args.jobarray_name is None:
            ja_hash = datetime.datetime.now().isoformat().replace(":", "_")
            args.jobarray_name = f"jobarray_{ja_hash}"
        assert not args.local, "Job array should not be local"
        assert not args.sequential, "Cannot have both sequential and jobarray"
    return args


class hyperparam(object):
    """Base class for defining hyperparameters."""

    def __init__(
        self,
        name,
        values=None,
        binary_flag=False,
        save_dir_key=None,
        positional_arg=False,
    ):
        """
        Arguments:
        - name : the name of the hyperparameter (e.g., `--dropout`)
        - values : the set of values to sweep over (e.g., `[0.0, 0.1, 0.2]`)
        - binary_flag : whether the hyperparameter uses a boolean flag (e.g., `--no-save`)
        - save_dir_key : function that takes the hyperparameter value and returns the "key"
                         to be appended to the output directory name
        - positional_arg : whether the hyperparameter is a positional argument
        """
        self.name = name
        if values is None:  # syntactic sugar for binary flags
            self.values = [True]
            self.binary_flag = True
        else:
            self.values = values if isinstance(values, list) else [values]
            self.binary_flag = binary_flag
        self.save_dir_key = save_dir_key
        self.positional_arg = positional_arg
        self.current_value = None

        if positional_arg and name.startswith("-"):
            raise ValueError(
                f"positional arguments must not start with a dash ({name})"
            )

        if len(self.values) > 1 and self.save_dir_key is None:
            raise ValueError(
                f"{name} has more than one value but is missing a save_dir_key!"
            )

    def get_cli_args(self):
        if self.binary_flag:
            return [self.name] if self.current_value else []
        elif self.positional_arg:
            return [self.current_value]
        else:
            return [self.name, self.current_value]

    def get_save_dir_key(self):
        if self.save_dir_key is None:
            return None
        if self.binary_flag:
            return self.save_dir_key(1) if self.current_value else None
        return self.save_dir_key(self.current_value)


def main(
    get_grid: Callable[[argparse.Namespace], List[hyperparam]],
    postprocess_hyperparams: Callable[
        [argparse.Namespace, MutableMapping[str, hyperparam]], None
    ],
    add_extra_options_func: Optional[Callable[[argparse.ArgumentParser], None]] = None,
    scheduler_args: Optional[List[str]] = None,
) -> None:
    """Do a grid search.
    Example:
    >>> # a 1-dimensional grid with 2 possible configurations
    >>> def get_1d_grid(args):
    ...   return [hyperparam("--some-arg", values=[1, 10])]
    ...
    >>> # a 2-dimensional grid with 3*3=9 possible configurations
    >>> def get_2d_grid(args):
    ...   return [
    ...     hyperparam("--foo", values=[1, 10, 100]),
    ...     hyperparam("--bar", values=[2, 4, 8]),
    ...   ]
    ...
    >>> def double_hyperparams(args, config):
    ...   for k in config:
    ...     config[k].current_value *= 2
    ...
    >>> def add_extra_options_func(parser):
    ...   parser.add_argument("--some-extra-argument", help="My extra argument")
    ...
    >>> # sweep over 1d grid with 2 possible values; read arguments from sys.argv
    >>> main(
    ...   get_1d_grid, double_hyperparams, add_extra_options_func
    ... )
    >>> # sweep over 2d grid with 9 possible values; read arguments from elsewhere
    >>> main(
    ...   get_2d_grid,
    ...   double_hyperparams,
    ...   add_extra_options_func,
    ...   ["--some-extra-argument"],
    ... )

    Parameters:
        get_grid: A unary callable which returns the grid to search over.
            The callable is passed the parsed sweep arguments including the extra
            arguments defined by `add_extra_options_func`. See also `get_args`.
            The returned list represents the dimensions of the grid. That is, a list of
            length n represents a grid of dimension n. Let v_i denote the number of
            possible values for dimension i. Then the total number of configurations
            is given by v_1 * ... * v_n.
        postprocess_hyperparams: A 2-ary callable to post-process hyperparameter
            configurations before running the job. The first argument is the parsed
            sweep arguments including the extra arguments defined by
            `add_extra_options_func`. The second argument is a realized hyperparameter
            configuration as a mutable mapping of hyperparameter name to `hyperparam`
            instance with a `current_value` set.
        add_extra_options_func: A unary callable which adds extra arguments to the
            sweep CLI. It is passed the parser used to define the sweep script's CLI.
        scheduler_args: A list of unprocessed arguments to parse. If None, then
            `sys.argv[1:]`.
    """
    args = get_args(add_extra_options_func, scheduler_args)
    if args.backend == "fblearner":
        from .fblearner import main as backend_main
    elif args.backend == "chronos":
        from .chronos import main as backend_main
    elif args.backend == "slurm":
        from .slurm import main as backend_main
    elif args.backend == "tpu":
        from .tpu import main as backend_main

    get_grid = get_grid[args.grid] if args.grid is not None else get_grid
    backend_main(get_grid, postprocess_hyperparams, args)
