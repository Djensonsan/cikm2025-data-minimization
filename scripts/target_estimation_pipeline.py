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
from minipack.pipelines.target_estimation.pipeline_builder import PipelineBuilder
from minipack.pipelines.registries import (
    ALGORITHM_REGISTRY,
    SCENARIO_REGISTRY,
)
from minipack.logger_config import setup_logging


@click.command()
@click.option(
    "--output-path",
    help="Specifies the directory where output files will be written. If not specified, outputs are saved to the current working directory.",
    default=os.getcwd(),
)
@click.option(
    "--dataset",
    help="Selects the dataset to be used for running the experiment. Choose from 'Netflix', 'MovieLens', 'MSD', or 'Dummy'.",
    type=click.Choice(["Netflix", "MovieLens", "MSD", "Dummy"], case_sensitive=False),
    default="Dummy",
)
@click.option(
    "--dataset-path",
    help="Defines the path to the location of the dataset files.",
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
    help="Provides the file path to the trained model. This option requires a user-defined path.",
)
@click.option(
    "--scenario",
    help="Determines the testing scenario for the experiment. Options include 'ProportionalStratifiedStrongGeneralization' and 'UniformStratifiedStrongGeneralization'. The default 'ProportionalStratifiedStrongGeneralization' is for extensive testing.",
    type=click.Choice(
        [
            "ProportionalStratifiedStrongGeneralization",
            "UniformStratifiedStrongGeneralization",
        ],
        case_sensitive=False,
    ),
    default="ProportionalStratifiedStrongGeneralization",
)
@click.option(
    "--minimizer",
    help="Specifies the minimizer to use for the target estimation pipeline. The default value is 'GreedyForwardMinimizer'.",
    type=click.Choice(["GreedyForwardMinimizer"], case_sensitive=False),
    default="GreedyForwardMinimizer",
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
    output_path,
    dataset,
    dataset_path,
    model,
    model_path,
    scenario,
    minimizer,
    target_estimator,
    num_cpus,
    object_store_memory,
    seed,
):
    # TODO: What about the seed?
    """Run the target estimation pipeline for the specified model and dataset."""
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

    # Define the log file path
    log_file_path = os.path.join(results_directory, "run.log")

    # Required for each entry-point:
    logger = setup_logging(log_file_path)
    logger.info(f"Logs can be found at: {log_file_path}")
    logger.info(f"Results can be found at: {results_directory}")
    logger.info(
        f"Starting target estimation pipeline for {model} on {dataset}, with {scenario} scenario."
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
    model = model_cls.load(model_path)
    logger.info("Finished loading model")

    logger.info("Building target estimation pipeline")
    builder = PipelineBuilder()
    builder.save_results = True
    builder.results_directory = results_directory
    builder.log_file_path = log_file_path
    builder.model = model

    target_estimator_hyperopt_config = load_config(
        f"{config_path}/target_estimation_config/target_estimator_hyperopt_config.yaml"
    )
    target_estimator_hyperopt_params = target_estimator_hyperopt_config["estimator"][
        target_estimator
    ]

    builder.add_target_estimator(
        target_estimator, grid=target_estimator_hyperopt_params
    )

    # RecPack uses InteractionMatrices, we exclusively use CSR matrices
    interaction_matrix = scenario_instance.estimation_data_in.binary_values
    test_matrix = scenario_instance.estimation_data_out.binary_values

    # Precaution: some datasets might have empty users in the test data (probably better to add a filter to the scenario method)
    interaction_matrix, test_matrix = eliminate_empty_users(
        interaction_matrix, test_matrix
    )

    builder.interaction_matrix = interaction_matrix
    builder.test_matrix = test_matrix

    # Metrics for test set:
    builder.add_metric("NDCGK", K=[100])
    builder.add_metric("RecallK", K=[20, 50])
    builder.add_metric("CalibratedRecallK", K=[20, 50])

    minimizer_params = target_estimator_hyperopt_config["minimizer"][minimizer]["params"]
    builder.add_minimizer(minimizer, params=minimizer_params)

    # Build the pipeline & run
    pipeline = builder.build()
    logger.info("Finished building target estimation pipeline")

    logger.info("Running target estimation pipeline")
    # TODO: other parameters?
    pipeline.run(num_cpus=num_cpus, object_store_memory=object_store_memory)
    logger.info("Finished running target estimation pipeline")


if __name__ == "__main__":
    run()
