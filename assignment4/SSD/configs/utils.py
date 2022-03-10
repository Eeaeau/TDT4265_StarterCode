import getpass
import pathlib

def get_dataset_dir(dataset_relpath: str):
    server_dir = pathlib.Path("/work/datasets", dataset_relpath)
    if server_dir.is_dir():
        print("Found dataset directory in:", server_dir)
        return str(server_dir)
    return str(pathlib.Path("data", dataset_relpath))

def get_output_dir():
    work_dir = pathlib.Path("/work", "snotra", getpass.getuser())
    save_in_work = False
    if work_dir.is_dir():
        print(f"It looks like you are currently working on the server, with a 'working directory' in: {work_dir}")
        if save_in_work:
            print("Saving all SSD outputs to:", work_dir.joinpath("ssd_outputs"))
            return work_dir.joinpath("ssd_outputs")
        else:
            print("\tIf you struggle with NTNU home directory becoming full, we recommend you to change the output directory to:", work_dir)
            print(f"\t {work_dir} does not sync with NTNU HOME, and is a directory only located on the server.")
            print("\t To change the output directory of SSD, set save_in_work to True in the file configs/utils.py, in the function get_output_dir.")

    print("Saving SSD outputs to: outputs/")
    return pathlib.Path("outputs")
