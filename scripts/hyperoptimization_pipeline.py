import os
import sys

# Add the project root to the system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import click
from util import (
    get_config_path,
    load_config,
    get_dataset,
    construct_results_directory_path,
    get_script_name,
)
from recpack.pipelines import PipelineBuilder
from minipack.pipelines.registries import SCENARIO_REGISTRY
from minipack.logger_config import setup_logging


@click.command()
@click.option(
    "--output-path",
    help="Specifies the directory where output files will be written. If not specified, outputs are saved to the current working directory.",
    default=os.getcwd(),
)
@click.option(
    "--dataset",
    help="Selects the dataset to be used for running the experiment. Options include 'Netflix', 'MovieLens', 'MSD', and 'Dummy'. The default is 'Netflix'.",
    type=click.Choice(["Netflix", "MovieLens", "MSD", "Dummy"], case_sensitive=False),
    default="Dummy",
)
@click.option(
    "--dataset-path",
    help="Defines the path to the location of the dataset files. Defaults to '../data/' for easy access.",
    default="../data/",
)
@click.option(
    "--algorithm",
    help="Chooses the algorithm to train. Options are 'EASE', 'ItemKNN', 'ItemPNN', 'NMFItemToItem', 'SVDItemToItem' and 'SLIM'. 'EASE' is selected by default.",
    type=click.Choice(
        ["EASE", "ItemKNN", "ItemPNN", "NMFItemToItem", "SVDItemToItem", "SLIM"],
        case_sensitive=False,
    ),
    default="EASE",
)
@click.option(
    "--scenario",
    help="Determines the testing scenario for the experiment. Options include 'StrongGeneralization', 'ProportionalStratifiedStrongGeneralization' and 'UniformStratifiedStrongGeneralization'. The default 'ProportionalStratifiedStrongGeneralization' is for extensive testing.",
    type=click.Choice(
        ["StrongGeneralization", "ProportionalStratifiedStrongGeneralization", "UniformStratifiedStrongGeneralization"], case_sensitive=False
    ),
    default="ProportionalStratifiedStrongGeneralization",
)
@click.option(
    "--seed",
    help="Sets the random seed for initializing the experiment, ensuring reproducibility. The default value is 42.",
    default=42,
)
def run(output_path, dataset, dataset_path, algorithm, scenario, seed):
    """Run the hyperoptimization pipeline for the specified algorithm and dataset."""
    # Construct the output directory path
    script_name = get_script_name(__file__)
    config_path = get_config_path()
    results_directory = construct_results_directory_path(
        output_path,
        script_name,
        dataset,
        algorithm,
        scenario=scenario,
    )

    # Define the results and log file paths
    results_file_path = os.path.join(results_directory, "results.csv")
    optim_results_file_path = os.path.join(results_directory, "optim_results.csv")
    log_file_path = os.path.join(results_directory, "run.log")

    # required for each entry-point
    logger = setup_logging(log_file_path)
    logger.info(f"Logs can be found at: {log_file_path}")
    logger.info(f"Results can be found at: {results_directory}")
    logger.info(
        f"Starting hyperoptimization pipeline for {algorithm} on {dataset}, with {scenario} scenario."
    )

    logger.info("Loading dataset")
    interaction_matrix = get_dataset(dataset_path, dataset)["dataset"].load()
    logger.info("Finished loading dataset")

    logger.info("Splitting dataset")
    scenario_cls = SCENARIO_REGISTRY.get(scenario)
    scenario_config = load_config(f"{config_path}/scenario_config.yaml")
    scenario_params = scenario_config[scenario][dataset]
    scenario_params["seed"] = seed
    scenario_instance = scenario_cls(**scenario_params)
    scenario_instance.split(interaction_matrix)
    logger.info("Finished splitting dataset")

    logger.info("Building hyperoptimization pipeline")
    builder = PipelineBuilder()
    builder.set_data_from_scenario(scenario_instance)

    hyperopt_config = load_config(f"{config_path}/hyperopt_config.yaml")
    hyperopt_params = hyperopt_config[algorithm]

    builder.add_algorithm(algorithm, grid=hyperopt_params)
    builder.set_optimisation_metric("NDCGK", K=100)

    # Metrics for test set:
    builder.add_metric("NDCGK", K=[100])
    builder.add_metric("RecallK", K=[20, 50])
    builder.add_metric("CalibratedRecallK", K=[20, 50])

    # Construct pipeline
    pipeline = builder.build()
    logger.info("Finished building hyperoptimization pipeline")

    logger.info("Running hyperoptimization pipeline")
    pipeline.run()
    logger.info("Finished running hyperoptimization pipeline")

    logger.info("Saving metrics")
    # Save the test results to a csv file
    final_result = pipeline.get_metrics()
    final_result.to_csv(results_file_path)

    optim_results = pipeline.optimisation_results
    optim_results.to_csv(optim_results_file_path)
    logger.info("Finished saving metrics")
    logger.info("Finished hyperoptimization pipeline")


if __name__ == "__main__":
    run()
