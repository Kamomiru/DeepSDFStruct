from DeepSDFStruct.deep_sdf.training_gan import train_deep_sdf_gan
import shutil
from pathlib import Path
import sys

def start_training_cycle(experiment_name, repetitions = 1, device = "cuda"):

    #checking whether experiment is under test experiments or actual experiments
    test_experiment_root_folder = Path("C:/Users/camil/Desktop/Bachelorarbeit/DeepSDFStruct/DeepSDFStruct/trained_models/test_experiments/")
    experiment_root_folder = Path("C:/Users/camil/Desktop/Bachelorarbeit/DeepSDFStruct/DeepSDFStruct/trained_models/experiments/")

    test_experiment_folder = test_experiment_root_folder / experiment_name  # the / operator joins paths in shutil
    experiment_folder = experiment_root_folder / experiment_name

    if test_experiment_folder.is_dir():
        root_path = test_experiment_folder
    elif experiment_folder.is_dir():
        root_path = experiment_folder
    else:
        raise RuntimeError(f"No suitable experiment path found for '{experiment_name}'.")


    if repetitions < 1:
        raise ValueError("repetitions must be at least 1")


    #check if experiment has already been run before
    if (root_path / "ModelParameters").is_dir():
        raise RuntimeError(
            "The path already contains a trained network without scheduling! "
            "Please pass a path containing only specs.json or a path containing scheduled training runs."
        )

    n_runs = get_highest_run_number(root_path)
    if n_runs > 0:
        while True:
            answer = input(
                f"The experiment already has {n_runs} training runs. "
                f"Do you wish to run {repetitions} more? [y/n]: "
            ).strip().lower()

            if answer == "n":
                print("Stopping training...")
                sys.exit(0)

            if answer in ("y", ""):
                print(f"Continuing running {repetitions} training runs")
                break



    # Determine where the first new run starts
    first_run = n_runs + 1

    first_run_folder = root_path / f"{root_path.name}.{first_run}"

    #find specs.json
    if n_runs == 0:
        specs_file = root_path / "specs.json"
    else:
        specs_file = (
            root_path
            / f"{root_path.name}.{n_runs}"
            / "specs.json"
        )

    if not specs_file.is_file():
        raise RuntimeError(
            f"Could not find specs.json at {specs_file}"
        )

    first_run_folder.mkdir()

    # Copy specs.json from the original experiment
    if n_runs == 0:
        shutil.move(
            root_path / "specs.json",
            first_run_folder / "specs.json"
        )
    else:
        shutil.copy(
            root_path / f"{root_path.name}.{n_runs}" / "specs.json",
            first_run_folder / "specs.json"
        )


    # Create remaining runs
    for i in range(first_run + 1, first_run + repetitions):

        new_folder = root_path / f"{root_path.name}.{i}"

        shutil.copytree(
            first_run_folder,
            new_folder
        )

    
    #training repetitions
    for i in range(first_run, first_run + repetitions):
        train_deep_sdf_gan(root_path / f"{root_path.name}.{i}", device=device)

def get_highest_run_number(root_path):
    highest = 0

    for item in root_path.iterdir():

        if not item.is_dir():
            continue

        if not item.name.startswith(root_path.name + "."):
            continue

        try:
            run_number = int(item.name.rsplit(".", 1)[1])
        except ValueError:
            continue

        highest = max(highest, run_number)

    return highest

if __name__ == "__main__":
    experiment_name = "gan_experiment23 chi 2-2"
    start_training_cycle(experiment_name, 5)