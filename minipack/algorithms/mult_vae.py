import torch
from torch.nn import Tanh, Dropout, Linear, ModuleList
from recpack.algorithms.mult_vae import MultiVAETorch
from recpack.algorithms.mult_vae import MultVAE as RecPackMultVAE
from minipack.algorithms.stopping_criterion import StoppingCriterion
from minipack.algorithms.util.serializable import SerializableTorchModel

# Allowlist globally, Torch throws an error when loading otherwise, (TODO) needs to be fixed in RecPack.
torch.serialization.add_safe_globals([MultiVAETorch, Tanh, Linear, Dropout, ModuleList])

class MultVAE(SerializableTorchModel, RecPackMultVAE):
    """
    A simple wrapper around a RecPack MultVAE model with save and load functionality.
    """

    def __init__(
        self,
        batch_size: int = 500,
        max_epochs: int = 200,
        learning_rate: float = 1e-4,
        seed: int = None,
        dim_bottleneck_layer: int = 200,
        dim_hidden_layer: int = 600,
        max_beta: float = 0.2,
        anneal_steps: int = 200000,
        dropout: float = 0.5,
        stopping_criterion: str = "ndcg",
        stop_early: bool = False,
        max_iter_no_change: int = 5,
        min_improvement: int = 0.01,
        save_best_to_file=False,
        keep_last: bool = False,
        predict_topK: int = None,
        validation_sample_size: int = None,
    ):
        super().__init__(
            batch_size=batch_size,
            max_epochs=max_epochs,
            learning_rate=learning_rate,
            dim_hidden_layer=dim_hidden_layer,
            dim_bottleneck_layer=dim_bottleneck_layer,
            max_beta=max_beta,
            anneal_steps=anneal_steps,
            dropout=dropout,
            stopping_criterion=stopping_criterion,
            stop_early=stop_early,
            max_iter_no_change=max_iter_no_change,
            min_improvement=min_improvement,
            seed=seed,
            save_best_to_file=save_best_to_file,
            keep_last=keep_last,
            predict_topK=predict_topK,
            validation_sample_size=validation_sample_size,
        )
        # Note: We use a custom StoppingCriterion because RecPack hardcodes the parameters.
        # TODO: Probably want to make StoppingCriterion more flexible.
        self.stopping_criterion = StoppingCriterion.create(
            stopping_criterion,
            stop_early=stop_early,
            max_iter_no_change=max_iter_no_change,
            min_improvement=min_improvement,
        )