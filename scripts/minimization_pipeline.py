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
    eliminate_empty_users,
)
from minipack.pipelines.registries import ALGORITHM_REGISTRY, SCENARIO_REGISTRY
from minipack.pipelines.minimization.pipeline_builder import PipelineBuilder
from minipack.logger_config import setup_logging

@click.command()
@click.option(
    "--config_file",
    help="Specifies the configuration file to use for the experiment. The config file should be located in the 'config' directory under 'minimization_config'.",
    default="minimizer_baselines_config.yaml",
)
@click.option(
    "--output-path",
    help="Specifies the directory where output files will be written. If not specified, outputs are saved to the current working directory.",
    default=os.getcwd(),
)
@click.option(
    "--dataset",
    help="Selects the dataset to be used for running the experiment. Choose from 'Netflix', 'MovieLens', 'MSD', or 'Dummy'. This default is only for easy testing purposes.",
    type=click.Choice(["Netflix", "MovieLens", "MSD", "Dummy"], case_sensitive=False),
    default="Dummy",
)
@click.option(
    "--dataset-path",
    help="Defines the path to the location of the dataset files. Defaults to '../data/' for convenience.",
    default="../data/",
)
@click.option(
    "--model",
    help="Specifies the class of the recommendation model. Options are 'EASE', 'ItemKNN', 'ItemPNN', 'NMFItemToItem', 'SVDItemToItem' and 'SLIM'. 'EASE' is selected by default.",
    type=click.Choice(
        ["EASE", "ItemKNN", "ItemPNN", "NMFItemToItem", "SVDItemToItem", "SLIM"],
        case_sensitive=False,
    ),
    default="EASE",
)
@click.option(
    "--model-path",
    help="Provides the file path to the trained model. If not specified, no default path will be used. This default is only for easy testing purposes.",
)
@click.option(
    "--scenario",
    help="Determines the testing scenario for the experiment. Options include 'ProportionalStratifiedStrongGeneralization' and 'UniformStratifiedStrongGeneralization'. The default 'ProportionalStratifiedStrongGeneralization' is for extensive testing.",
    type=click.Choice(
        ["ProportionalStratifiedStrongGeneralization", "UniformStratifiedStrongGeneralization"], case_sensitive=False
    ),
    default="ProportionalStratifiedStrongGeneralization",
)
@click.option(
    "--target-estimator",
    help="Specifies the target estimator to use for the target estimation pipeline. The default value is 'ExponentialDecayEstimator'.",
    type=click.Choice(["ExponentialDecayEstimator"], case_sensitive=False),
    default="ExponentialDecayEstimator",
)
@click.option(
    "--num-cpus",
    help="Specifies the number of CPUs to use for parallel processing. The default value is 1.",
    default=1,
)
@click.option(
    "--object-store-memory",
    help="Specifies the amount of memory to allocate for the shared object store. The default value is 2 (GB).",
    default=2,
)
@click.option(
    "--seed",
    help="Sets the random seed for initializing the experiment, ensuring reproducibility. The default value is 42.",
    default=42,
)
def run(
    config_file,
    output_path,
    dataset,
    dataset_path,
    model,
    model_path,
    scenario,
    target_estimator,
    num_cpus,
    object_store_memory,
    seed,
):
    """
    Run the minimization pipeline for the specified algorithm and dataset.
    """
    # Construct the output directory path
    script_name = get_script_name(__file__)
    config_path = get_config_path()
    results_directory = construct_results_directory_path(
        output_path,
        script_name,
        dataset,
        model,
        scenario=scenario,
    )

    # Define the results and log file paths
    log_file_path = os.path.join(results_directory, "run.log")

    # required for each entry-point
    logger = setup_logging(log_file_path)
    logger.info(f"Logs can be found at: {log_file_path}")
    logger.info(f"Results can be found at: {results_directory}")
    logger.info(
        f"Starting minimization pipeline for {model} on {dataset}, with {scenario} scenario."
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

    logger.info("Loading model")
    model_cls = ALGORITHM_REGISTRY.get(model)
    model_instance = model_cls.load(model_path)
    logger.info("Finished loading model")

    logger.info("Building minimization pipeline")
    builder = PipelineBuilder()
    builder.save_results = True
    builder.results_directory = results_directory
    builder.log_file_path = log_file_path
    builder.model = model_instance

    target_estimator_config = load_config(
        f"{config_path}/target_estimation_config/target_estimator_config.yaml"
    )
    target_estimator_params = target_estimator_config[target_estimator][model][dataset]
    builder.add_target_estimator(target_estimator, params=target_estimator_params)

    interaction_matrices = [test_data_bin.binary_values for test_data_bin in scenario_instance.test_data_in_bins]
    test_matrices = [test_data_bin.binary_values for test_data_bin in scenario_instance.test_data_out_bins]
    interaction_matrices, test_matrices = eliminate_empty_users(interaction_matrices, test_matrices)
    builder.interaction_matrices = interaction_matrices
    builder.test_matrices = test_matrices

    minimizers = load_config(f"{config_path}/minimization_config/{config_file}")
    builder.add_minimizers(minimizers)

    # Metrics for test set:
    builder.add_metric("NDCGK", K=[100])
    builder.add_metric("RecallK", K=[20, 50])
    builder.add_metric("CalibratedRecallK", K=[20, 50])

    # Build the pipeline & run
    pipeline = builder.build()
    logger.info("Finished building minimization pipeline")

    logger.info("Running minimization pipeline")
    # TODO: other parameters?
    pipeline.run(num_cpus=num_cpus, object_store_memory=object_store_memory)
    logger.info("Finished running minimization pipeline")

if __name__ == "__main__":
    run()