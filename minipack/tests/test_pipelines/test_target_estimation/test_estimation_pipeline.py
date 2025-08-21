import os
import shutil

def test_pipeline_get_results(mock_pipeline_builder):
    pipeline = mock_pipeline_builder.build()

    pipeline.run(local_mode=True)

    results_columns = pipeline.batch_results.columns
    assert "minimizer_identifier" in results_columns
    assert "estimator_identifier" in results_columns
    assert "batch_number_of_users" in results_columns
    assert "batch_minimization_ratio" in results_columns
    assert "batch_runtime" in results_columns
    assert "batch_sample_count" in results_columns
    assert "average_runtime" in results_columns
    assert "average_sample_count" in results_columns


def test_pipeline_save_results(mock_pipeline_builder):
    mock_pipeline_builder.save_results = True
    pipeline = mock_pipeline_builder.build()

    try:
        assert not os.path.isdir(pipeline.results_directory)

        pipeline.run(local_mode=True)

        assert os.path.isfile(pipeline.batch_results_file_path)
        assert os.path.isfile(pipeline.user_results_file_path)

    finally:
        # Ensure cleanup happens regardless of test outcome
        if os.path.isdir(pipeline.results_directory):
            shutil.rmtree(pipeline.results_directory)