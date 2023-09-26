import os
import json
import tempfile
teardown_tmp_dirs = []
def get_auto_remove_tmp_dir(tmp_dir=None, before=None, after=None):
    """
    Args:
        tmp_dir (`string`, *optional*):
            if `None`:
               - a unique temporary path will be created
               - sets `before=True` if `before` is `None`
               - sets `after=True` if `after` is `None`
            else:
               - `tmp_dir` will be created
               - sets `before=True` if `before` is `None`
               - sets `after=False` if `after` is `None`
        before (`bool`, *optional*):
            If `True` and the `tmp_dir` already exists, make sure to empty it right away if `False` and the
            `tmp_dir` already exists, any existing files will remain there.
        after (`bool`, *optional*):
            If `True`, delete the `tmp_dir` at the end of the test if `False`, leave the `tmp_dir` and its contents
            intact at the end of the test.
    Returns:
        tmp_dir(`string`): either the same value as passed via *tmp_dir* or the path to the auto-selected tmp dir
    """
    if tmp_dir is not None:
        # defining the most likely desired behavior for when a custom path is provided.
        # this most likely indicates the debug mode where we want an easily locatable dir that:
        # 1. gets cleared out before the test (if it already exists)
        # 2. is left intact after the test
        if before is None:
            before = True
        if after is None:
            after = False

        # using provided path
        path = Path(tmp_dir).resolve()

        # to avoid nuking parts of the filesystem, only relative paths are allowed
        if not tmp_dir.startswith("./"):
            raise ValueError(
                f"`tmp_dir` can only be a relative path, i.e. `./some/path`, but received `{tmp_dir}`"
            )

        # ensure the dir is empty to start with
        if before is True and path.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)

        path.mkdir(parents=True, exist_ok=True)

    else:
        # defining the most likely desired behavior for when a unique tmp path is auto generated
        # (not a debug mode), here we require a unique tmp dir that:
        # 1. is empty before the test (it will be empty in this situation anyway)
        # 2. gets fully removed after the test
        if before is None:
            before = True
        if after is None:
            after = True

        # using unique tmp dir (always empty, regardless of `before`)
        tmp_dir = tempfile.mkdtemp()

    # if after is True:
    #     # register for deletion
    #     teardown_tmp_dirs.append(tmp_dir)

    return tmp_dir

def get_results(output_dir):
    results = {}
    path = os.path.join(output_dir, "all_results.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            results = json.load(f)
    else:
        raise ValueError(f"can't find {path}")
    return results