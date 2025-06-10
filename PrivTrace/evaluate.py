"""Evaluation Code."""
import datetime
import logging
import multiprocessing
from argparse import ArgumentParser
from pathlib import Path
from timeit import default_timer as timer

import pandas as pd

import config.folder_and_file_names as fname
from config.parameter_carrier import ParameterCarrier
from config.parameter_setter import ParSetter
from data_preparation.data_preparer import DataPreparer
from discretization.get_discretization import DisData
from generator.state_trajectory_generation import StateGeneration
from generator.to_real_translator import RealLocationTranslator
from primarkov.build_markov_model import ModelBuilder
from tools.data_writer import DataWriter
from tools.utils import printc

# CONSTANTS
DEBUG = False
DATASETS = ["PORTO", "GEOLIFE"]
N_FOLDS = 5
N_TRAJS = 3000  # Number of trajectories to generate for our evaluation
EPSILON = 10.0
epsilons = [10.0]

c_parameters = {  # This is the c parameter used in the paper to compute K
    "Brinkhoff": 5000,
    "PORTO": 1200,
    "GEOLIFE": 500,
}
pop = {  # Population density for each dataset
    "PORTO": 1_310_000,  # Metropolitan Area of Porto
    "GEOLIFE": 	22_596_500  # Metropolitan Area of Beijing
}

# Whether we use these parameters or not
LEVEL_1_PARAMETER = False  # This equals K in the paper - we provide c instead and compute K from it
LEVEL_2_PARAMETER = False  # This equals kappa in the paper - we tried providing pop, but the equation in the paper
# is incorrect, at least, the result doesn't make any sense.

# Set up log
logging.basicConfig(
    format="%(asctime)s - [%(levelname)s]: %(message)s",
    level=logging.DEBUG,
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def run(dataset: str, fold: int, epsilon: float = EPSILON,
        level_1_parameter: bool = LEVEL_1_PARAMETER,
        level_2_parameter: bool = LEVEL_2_PARAMETER,
        ) -> None:
    # Copy of main.py
    if level_1_parameter:
        c_param = c_parameters[dataset]
    if level_2_parameter:
        pop_param = pop[dataset]

    # Log to file
    basename= f"{dataset}_e{epsilon:.1f}_{fold:02d}"
    # Append the other parameters if used
    if level_1_parameter:
        basename += f"custom-K"
    if level_2_parameter:
        basename += f"custom-Kappa"
    log_filename = Path("logs") / f"{basename}.log"
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - [%(levelname)s]: %(message)s")
    file_handler.setFormatter(formatter)
    # Add to root logger
    logging.getLogger().addHandler(file_handler)

    input_filename = f"{dataset}_{fold}.dat"
    output_filename = f"outputs/{basename}_output.dat"

    # Start Timer
    start = timer()

    writer = DataWriter()
    print('begin all')
    print(datetime.datetime.now())
    par = ParSetter().set_up_args(
        dataset_file_name=input_filename,  # Input File
        output_file_name=output_filename,  # Output File
        epsilon=epsilon,  # Final Epsilon (e1 + e2 + e3)
        out_size=N_TRAJS,  # Number of trajectories to generate for our evaluation

        # We use the default values as the code and the paper are not consistent
        c_parameter=c_param if level_1_parameter else False,  # c parameter
        pop=pop_param if level_2_parameter else False,  # Population density
    )

    pc = ParameterCarrier(par)

    # Load Data
    data_preparer = DataPreparer(par)
    trajectory_set = data_preparer.get_trajectory_set()
    log.info("Preparing data finished")

    # Step 1: Discretisation
    disdata1 = DisData(pc)
    grid = disdata1.get_discrete_data(trajectory_set)
    log.info("Discretization finished")
    log.info(f"Usable State Number: {grid.usable_state_number}")
    # Print the size of the grid
    log.info(f"Grid Size: {len(grid.x_divide_bins - 1)}X{len(grid.y_divide_bins - 1)}")

    # Step 2: Learn Markov Models
    mb1 = ModelBuilder(pc)
    mo1 = mb1.build_model(grid, trajectory_set)
    log.info("Markov model building finished")
    mb1 = ModelBuilder(pc)
    mo1 = mb1.filter_model(trajectory_set, grid, mo1)
    log.info("Markov model filtering finished")

    # Step 3: Generate Trajectories
    sg1 = StateGeneration(pc)
    st_tra_list = sg1.generate_tra(mo1)
    log.info("State trajectory generation finished")
    rlt1 = RealLocationTranslator(pc)
    real_tra_list = rlt1.translate_trajectories(grid, st_tra_list)
    log.info("Real trajectory generation finished")
    writer.save_trajectory_data_in_list_to_file(real_tra_list, par['output'])
    print('end all')
    print(datetime.datetime.now())

    # Get end time
    end = timer()
    runtime = end - start
    printc("Total Runtime:", f"{runtime // 60 // 60}h {runtime // 60 % 60}m {runtime % 60}s")
    # Write to file
    filepath = Path(fname.OUTPUT_DIR) / f"PrivTrace_{dataset}_e{epsilon:.1f}_runtime.txt"
    df = pd.DataFrame(data={'run_id': [fold], 'runtime': [runtime]})
    if filepath.exists():
        df.to_csv(filepath, mode='a', header=False, index=False)
    else:
        df.to_csv(filepath, mode='w', header=True, index=False)


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="PrivTrace Evaluation")
    # Add arguments for dataset, gold and epsilon
    parser.add_argument('-d', '--dataset', type=str, choices=DATASETS, required=True, help='Dataset to use')
    parser.add_argument('-f', '--fold', type=int, required=True, help='Fold number')
    parser.add_argument('-e', '--epsilon', type=float, default=EPSILON, help='Epsilon value')
    # Use custom level 1 parameter
    parser.add_argument('-c', '--custom-c', action='store_true', help='Use custom c/K parameter')
    # Enable Manual mode
    parser.add_argument('-m', '--manual', action='store_true', help='Enable manual mode')
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    if DEBUG:
        # Set logging to debug level
        log.setLevel(logging.DEBUG)
        # Only run geolife dataset with fold 1 and epsilon 10.0
        run(dataset="GEOLIFE", fold=6, epsilon=1000.0)
    elif args.manual:
        run(dataset=args.dataset, fold=args.fold, epsilon=args.epsilon, level_1_parameter=args.custom_c)
    else:
        processes = []
        for dataset in DATASETS:
            for fold in range(1, N_FOLDS + 1):
                for epsilon in epsilons:
                    p = multiprocessing.Process(
                        target=run,
                        kwargs={
                            "dataset": dataset,
                            "fold": fold,
                            "epsilon": epsilon,
                        },
                    )
                    processes.append(p)
                    print(f"Starting process for {dataset} fold {fold} with epsilon {epsilon}.")
                    p.start()

        for p in processes:
            p.join()
