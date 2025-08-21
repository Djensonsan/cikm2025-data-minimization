import torch
from typing import Optional
from torch.nn import Linear, Dropout, LayerNorm
from recpack.algorithms.rec_vae import RecVAETorch
from recpack.algorithms.rec_vae import Encoder, CompositePrior
from recpack.algorithms.rec_vae import RecVAE as RecPackRecVAE
from minipack.algorithms.stopping_criterion import StoppingCriterion
from minipack.algorithms.util.serializable import SerializableTorchModel

# Allowlist globally, Torch throws an error when loading otherwise, (TODO) needs to be fixed in RecPack.
torch.serialization.add_safe_globals([RecVAETorch, Linear, LayerNorm, Dropout, Encoder, CompositePrior])

class RecVAE(SerializableTorchModel, RecPackRecVAE):
    """
    A simple wrapper around a RecPack RecVAE model with save and load functionality.
    """

    def __init__(
        self,
        batch_size: int = 500,
        max_epochs: int = 200,
        learning_rate: float = 5e-4,
        n_enc_epochs: int = 3,
        n_dec_epochs: int = 1,
        seed: Optional[int] = None,
        dim_bottleneck_layer: int = 200,
        dim_hidden_layer: int = 600,
        gamma: Optional[float] = 0.005,
        beta: Optional[float] = None,
        dropout: float = 0.5,
        stopping_criterion: str = "ndcg",
        stop_early: bool = False,
        max_iter_no_change: int = 5,
        min_improvement: float = 0.0,
        save_best_to_file: bool = False,
        keep_last: bool = False,
        predict_topK: int = None,
        validation_sample_size: int = None,
    ):
        super().__init__(
            batch_size=batch_size,
            max_epochs=max_epochs,
            learning_rate=learning_rate,
            n_enc_epochs=n_enc_epochs,
            n_dec_epochs=n_dec_epochs,
            dim_hidden_layer=dim_hidden_layer,
            dim_bottleneck_layer=dim_bottleneck_layer,
            gamma=gamma,
            beta=beta,
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