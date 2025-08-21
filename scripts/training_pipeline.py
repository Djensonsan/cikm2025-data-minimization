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
from minipack.pipelines.training.pipeline_builder import PipelineBuilder
from minipack.pipelines.registries import SCENARIO_REGISTRY
from minipack.logger_config import setup_logging


@click.command()
@click.option(
    "--output-path",
    help="Specifies the directory where output files will be written. If not specified, outputs are saved to the current working directory by default.",
    default=os.getcwd(),
)
@click.option(
    "--dataset",
    help="Selects the dataset to be used for running the experiment. Available options include 'Netflix', 'MovieLens', 'MSD', and 'Dummy'. Defaults to 'Dummy'.",
    type=click.Choice(["Netflix", "MovieLens", "MSD", "Dummy"], case_sensitive=False),
    default="Dummy",
)
@click.option(
    "--dataset-path",
    help="Defines the path to the location of dataset files. Defaults to '../data/'.",
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
    help="Sets the random seed for initializing the algorithm, ensuring reproducibility. The default value is 42.",
    default=42,
)
def run(output_path, dataset, dataset_path, algorithm, scenario, seed):
    """Run the training pipeline for the specified algorithm and dataset."""
    # Construct the output directory path
    script_name = get_script_name(__file__)
    config_path = get_config_path()
    results_directory = construct_results_directory_path(
        output_path, script_name, dataset, algorithm, scenario
    )

    # Define the results and log file paths
    results_file_path = os.path.join(results_directory, "results.csv")
    log_file_path = os.path.join(results_directory, "run.log")

    # required for each entry-point
    logger = setup_logging(log_file_path)
    logger.info(f"Logs can be found at: {log_file_path}")
    logger.info(f"Results can be found at: {results_directory}")
    logger.info(
        f"Starting training pipeline for {algorithm} on {dataset}, with {scenario} scenario."
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

    logger.info("Building training pipeline")
    builder = PipelineBuilder()
    builder.set_data_from_scenario(scenario_instance)

    model_config = load_config(f"{config_path}/model_config.yaml")
    model_params = model_config[algorithm][dataset]
    builder.add_algorithm(algorithm, params=model_params)

    # Metrics for test set:
    builder.add_metric("NDCGK", K=[100])
    builder.add_metric("RecallK", K=[20, 50])
    builder.add_metric("CalibratedRecallK", K=[20, 50])

    builder.results_directory = results_directory

    # Construct pipeline
    pipeline = builder.build()
    logger.info("Finished building training pipeline")

    logger.info("Running training pipeline")
    pipeline.run()
    logger.info("Finished running training pipeline")

    logger.info("Saving metrics")
    # Save the test results to a csv file
    final_result = pipeline.get_metrics()
    final_result.to_csv(results_file_path)
    logger.info("Finished saving metrics")
    logger.info("Finished training pipeline")


if __name__ == "__main__":
    run()
