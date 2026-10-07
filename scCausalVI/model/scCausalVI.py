import logging
from typing import List, Literal, Optional, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from anndata import AnnData
from numpy import ndarray
from scvi.data import AnnDataManager
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
)
from scvi.dataloaders import AnnDataLoader
from scvi.distributions import ZeroInflatedNegativeBinomial
from scvi.model._utils import _init_library_size
from scvi.model.base import BaseModelClass
from scipy.stats import ttest_1samp
from sklearn.decomposition import PCA
from statsmodels.stats.multitest import multipletests
from torch.distributions import Binomial, Gamma, Poisson

from scCausalVI.model.base._utils import _invert_dict
from scCausalVI.model.base.training_mixin import scCausalVITrainingMixin
from scCausalVI.module.scCausalVI import scCausalVIModule
from .base import SCCAUSALVI_REGISTRY_KEYS

logger = logging.getLogger(__name__)


def _coupled_zinb_sample(
        mu_a: torch.Tensor,
        mu_b: torch.Tensor,
        zi_logits_a: torch.Tensor,
        zi_logits_b: torch.Tensor,
        theta: torch.Tensor,
):
    """
    Draw a pair of count matrices (x_a, x_b), each exactly ZINB(mu, theta, zi_logits),
    sharing as much randomness as possible so that x_a - x_b reflects only the difference
    of the two distributions.

    NB(mu, theta) is Poisson(mu * G) with G ~ Gamma(theta, theta); G is shared. The Poisson
    parts are coupled by binomial thinning (lower rate) or by adding an independent Poisson
    increment (higher rate). Zero inflation uses one shared uniform per entry.
    """
    theta = theta.expand_as(mu_a)
    g = Gamma(theta, theta).sample()
    lam_a, lam_b = mu_a * g, mu_b * g
    x_a = Poisson(lam_a).sample()
    keep = torch.where(lam_a > 0, torch.minimum(lam_a, lam_b) / lam_a, torch.zeros_like(lam_a))
    thinned = Binomial(total_count=x_a, probs=keep.clamp(0, 1)).sample()
    extra = Poisson((lam_b - lam_a).clamp(min=0)).sample()
    x_b = torch.where(lam_b <= lam_a, thinned, x_a + extra)
    u = torch.rand_like(mu_a)
    x_a = torch.where(u < torch.sigmoid(zi_logits_a), torch.zeros_like(x_a), x_a)
    x_b = torch.where(u < torch.sigmoid(zi_logits_b), torch.zeros_like(x_b), x_b)
    return x_a, x_b


def _log_normalize(x: torch.Tensor, target_sum: float) -> np.ndarray:
    """normalize_total(target_sum) + log1p on a dense count tensor."""
    size = x.sum(dim=1, keepdim=True).clamp(min=1.0)
    return torch.log1p(x / size * target_sum).cpu().numpy()


class scCausalVIModel(scCausalVITrainingMixin, BaseModelClass):
    """
    Model class for scCausalVI.
    Args:
    -----
        adata: AnnData object with count data.
        condition2int: Dict mapping condition name (str) -> index (int)
        control: Control condition in case-control study, containing cells in unperturbed states
        n_background_latent: Dimensionality of background latent space.
        n_te_latent: Dimensionality of treatment effect latent space.
        n_layers: Number of hidden layers of each sub-networks.
        n_hidden: Number of hidden nodes in each layer of neural network.
        dropout_rate: Dropout rate for the network.
        use_observed_lib_size: Whether to use the observed library size.
        use_mmd: Whether to use Maximum Mean Discrepancy (MMD) to align background latent representations
        across conditions.
        mmd_weight: Weight of MMD in loss function.
        norm_weight: Normalization weight in loss function.
        gammas: Kernel bandwidths for calculating MMD.
    """

    def __init__(
            self,
            adata: AnnData,
            condition2int: dict,
            control: str,
            n_background_latent: int = 10,
            n_te_latent: int = 10,
            n_layers: int = 2,
            n_hidden: int = 128,
            dropout_rate: float = 0.1,
            use_observed_lib_size: bool = True,
            use_mmd: bool = True,
            mmd_weight: float = 1.0,
            norm_weight: float = 0.3,
            gammas: Optional[np.ndarray] = None,
    ) -> None:

        super(scCausalVIModel, self).__init__(adata)

        # Determine number of batches from summary stats
        n_batch = self.summary_stats.n_batch

        # Initialize library size parameters if not using observed library size
        if use_observed_lib_size:
            library_log_means, library_log_vars = None, None
        else:
            library_log_means, library_log_vars = _init_library_size(self.adata_manager, n_batch)

        # Set default gamma values for MMD if not provided
        if use_mmd and gammas is None:
            gammas = torch.FloatTensor([10 ** x for x in range(-6, 7, 1)])

        # Initialize the module with the specified parameters
        self.module = scCausalVIModule(
            n_input=self.summary_stats["n_vars"],
            control=control,
            condition2int=condition2int,
            n_batch=n_batch,
            n_hidden=n_hidden,
            n_background_latent=n_background_latent,
            n_te_latent=n_te_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            use_observed_lib_size=use_observed_lib_size,
            library_log_means=library_log_means,
            library_log_vars=library_log_vars,
            use_mmd=use_mmd,
            mmd_weight=mmd_weight,
            norm_weight=norm_weight,
            gammas=gammas,
        )

        # Summary string for the model
        self._model_summary_string = "scCausalVI"

        # Capture initialization parameters for saving/loading
        self.init_params_ = self._get_init_params(locals())

        logger.info("The model has been initialized.")

    @classmethod
    # @setup_anndata_dsp.dedent
    def setup_anndata(
            cls,
            adata: AnnData,
            layer: Optional[str] = None,
            batch_key: Optional[str] = None,
            condition_key: Optional[str] = None,
            size_factor_key: Optional[str] = None,
            categorical_covariate_keys: Optional[List[str]] = None,
            continuous_covariate_keys: Optional[List[str]] = None,
            **kwargs,
    ):
        """
        Set up AnnData instance for scCausalVI model

        Args:
            adata: AnnData object with .layers[layer] attribute containing count data.
            layer: Key for `.layers` or `.raw` where count data are stored.
            batch_key: Key for batch information in `adata.obs`.
            condition_key: Key for condition information in `adata.obs`.
            size_factor_key: Key for size factor information in `adata.obs`.
            categorical_covariate_keys: Keys for categorical covariates in `adata.obs`.
            continuous_covariate_keys: Keys for continuous covariates in `adata.obs`.
        """

        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(SCCAUSALVI_REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(SCCAUSALVI_REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(SCCAUSALVI_REGISTRY_KEYS.CONDITION_KEY, condition_key),
            NumericalObsField(
                SCCAUSALVI_REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False
            ),
            CategoricalJointObsField(
                SCCAUSALVI_REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys
            ),
            NumericalJointObsField(
                SCCAUSALVI_REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys
            ),
        ]
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    @torch.no_grad()
    def get_latent_representation(
            self,
            adata: Optional[AnnData] = None,
            indices: Optional[Sequence[int]] = None,
            give_mean: bool = True,
            batch_size: Optional[int] = None,
    ) -> tuple[ndarray, ndarray]:
        """
        Compute background and treatment effect latent representations for each cell based on their condition labels.

        Args:
        ----
        adata: AnnData object. If `None`, defaults to the AnnData object used to initialize the model.
        indices: Indices of cells in adata to use. If `None`, all cells are used.
        give_mean: Bool. If True, give mean of distribution insteading of sampling from it.
        batch_size: Mini-batch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        Returns
        -------
            A tuple of two numpy arrays with shape `(n_cells, n_latent)` for background and
            treatment effect latent representations.
        """
        adata = self._validate_anndata(adata)
        data_loader = self._make_data_loader(
            adata=adata,
            indices=indices,
            batch_size=batch_size,
            shuffle=False,
            data_loader_class=AnnDataLoader,
        )

        latent_bg = []
        latent_t = []

        # Integer label for control condition
        control_label_idx = self.module.condition2int[self.module.control]

        label_to_name = _invert_dict(self.module.condition2int)

        for tensors in data_loader:
            x = tensors[SCCAUSALVI_REGISTRY_KEYS.X_KEY]
            batch_index = tensors[SCCAUSALVI_REGISTRY_KEYS.BATCH_KEY]
            label_index = tensors[SCCAUSALVI_REGISTRY_KEYS.CONDITION_KEY]

            # Placeholders for this mini-batch
            bg_holder = torch.zeros([x.shape[0], self.module.n_background_latent])
            t_holder = torch.zeros([x.shape[0], self.module.n_te_latent])

            unique_labels = label_index.unique()

            for lbl in unique_labels:
                mask = (label_index == lbl).squeeze(dim=-1).cpu()
                x_sub = x[mask]
                batch_sub = batch_index[mask]

                if lbl.item() == control_label_idx:
                    # Control path to get background latent factors
                    src = "control"
                    outputs = self.module._generic_inference(
                        x=x_sub, batch_index=batch_sub, src=src, condition_label=label_index[mask],
                    )
                    z_bg_label = outputs["z_bg"]
                    chosen_bg = outputs["qbg_m"] if give_mean else z_bg_label
                    bg_holder[mask] = chosen_bg.detach().cpu()
                    # Treatment effect latent factors remain zero in control condition
                else:
                    # Treatment path to get background latent factors
                    treat_name = label_to_name.get(lbl.item(), None)
                    if treat_name is None:
                        raise ValueError(f"Unknown condition label: {lbl.item()}")
                    src = "treatment"
                    outputs = self.module._generic_inference(
                        x=x_sub, batch_index=batch_sub, src=src, condition_label=label_index[mask]
                    )
                    z_bg_label = outputs["z_bg"]
                    chosen_bg = outputs["qbg_m"] if give_mean else z_bg_label

                    z_t_label = outputs["z_t"]
                    chosen_t = outputs["qt_m"] if give_mean else z_t_label

                    bg_holder[mask] = chosen_bg.detach().cpu()
                    t_holder[mask] = chosen_t.detach().cpu()

            latent_bg.append(bg_holder)
            latent_t.append(t_holder)

        latent_bg = torch.cat(latent_bg, dim=0).numpy()
        latent_t = torch.cat(latent_t, dim=0).numpy()
        return latent_bg, latent_t

    @torch.no_grad()
    def get_latent_representation_cross_condition(
            self,
            source_condition: str,
            target_condition: str,
            adata: Optional[AnnData] = None,
            indices: Optional[Sequence[int]] = None,
            give_mean: bool = True,
            batch_size: Optional[int] = None,
    ) -> tuple[ndarray, ndarray]:
        """
        Compute latent representations for cross-condition prediction.
        For cells from source_condition, computes their latent representations as if they were
        under target_condition. For all other cells, computes latent representations under
        their original conditions.

        Args:
        ----
        source_condition: Name of source condition from which to select cells for cross-condition prediction.
        target_condition: Name of target condition under which to predict expression profiles.
        adata: AnnData object to use. If None, uses the model's AnnData object.
        indices: Indices of cells in adata to use. If None, uses all cells.
        give_mean: If True, returns the mean of the distribution instead of sampling
        batch_size: Minibatch size for data loading. If None, uses scvi.settings.batch_size.

        Returns
        -------
            A tuple of ndarrays of background latent factors and treatment effect latent factors with
            shape `(n_cells, n_latent)`.
        """

        if source_condition == target_condition:
            raise ValueError(f"source condition and target condition should be different.")

        adata = self._validate_anndata(adata)

        data_loader = self._make_data_loader(
            adata=adata,
            indices=indices,
            batch_size=batch_size,
            shuffle=False,
            data_loader_class=AnnDataLoader,
        )

        latent_bg = []
        latent_t = []

        control_label_idx = self.module.condition2int[self.module.control]

        label_to_name = _invert_dict(self.module.condition2int)

        source_label_idx = self.module.condition2int[source_condition]
        target_label_idx = self.module.condition2int[target_condition]

        for tensors in data_loader:
            x = tensors[SCCAUSALVI_REGISTRY_KEYS.X_KEY]
            batch_index = tensors[SCCAUSALVI_REGISTRY_KEYS.BATCH_KEY]
            label_index = tensors[SCCAUSALVI_REGISTRY_KEYS.CONDITION_KEY]

            bg_holder = torch.zeros([x.shape[0], self.module.n_background_latent])
            t_holder = torch.zeros([x.shape[0], self.module.n_te_latent])

            unique_labels = label_index.unique()

            for lbl in unique_labels:
                mask = (label_index == lbl).squeeze(dim=-1).cpu()
                x_sub = x[mask]
                batch_sub = batch_index[mask]

                if lbl.item() == control_label_idx:
                    # Use control_background_encoder for control data
                    outputs = self.module._generic_inference(
                        x=x_sub, batch_index=batch_sub, src="control", condition_label=label_index[mask]
                    )
                    z_bg_label = outputs["z_bg"]
                    chosen_bg = outputs["qbg_m"] if give_mean else z_bg_label
                    bg_holder[mask] = chosen_bg.detach().cpu()

                    # If the source_condition is control, forcibly apply target_conditions's treatment
                    # effect encoder
                    if source_label_idx == control_label_idx:
                        if target_label_idx != control_label_idx:
                            # From control -> real treatment
                            target_name = label_to_name.get(target_label_idx, None)
                            if target_name is not None:
                                s_enc = self.module.treatment_te_encoders[target_name]
                                tm, tv, zt = s_enc(z_bg_label)
                                chosen_t = tm if give_mean else zt
                                t_holder[mask] = chosen_t.detach().cpu()
                            else:
                                raise ValueError(f"Unknown treatment: {target_name}")
                        else:
                            raise ValueError(f'target_condition should be different from source_condition.')
                    else:
                        # Normal control => no treatment effects
                        pass
                else:
                    # Use corresponding treatment_background_encoder for each treated data
                    tname = label_to_name.get(lbl.item(), None)
                    if tname is None:
                        raise ValueError(f"Unknown treatment: {tname} in model.")
                    outputs = self.module._generic_inference(
                        x=x_sub, batch_index=batch_sub, src="treatment", condition_label=label_index[mask]
                    )
                    z_bg_label = outputs["z_bg"]
                    chosen_bg = outputs["qbg_m"] if give_mean else z_bg_label
                    bg_holder[mask] = chosen_bg.detach().cpu()

                    # If this batch label == source_condition, forcibly apply target_condition's
                    # treatment effect encoder
                    if lbl.item() == source_label_idx:
                        if target_label_idx == control_label_idx:
                            # From treat -> control => No treatment effect
                            pass
                        else:
                            target_name = label_to_name.get(target_label_idx, None)
                            if target_name is not None:
                                s_enc = self.module.treatment_te_encoders[target_name]
                                tm, tv, zt = s_enc(z_bg_label)
                                chosen_t = tm if give_mean else zt
                                t_holder[mask] = chosen_t.detach().cpu()
                            else:
                                raise ValueError(f"Unknown treatment: {target_name}")
                    else:
                        # Normal path => keep same treatment's treatment effect latent representation
                        chosen_t = outputs["qt_m"] if give_mean else outputs["z_t"]
                        t_holder[mask] = chosen_t.detach().cpu()

            latent_bg.append(bg_holder)
            latent_t.append(t_holder)

        latent_bg = torch.cat(latent_bg, dim=0).numpy()
        latent_t = torch.cat(latent_t, dim=0).numpy()
        return latent_bg, latent_t

    @torch.no_grad()
    def get_count_expression(
            self,
            adata: Optional[AnnData] = None,
            indices: Optional[Sequence[int]] = None,
            target_batch: Optional[int] = None,
            batch_size: Optional[int] = None,
    ) -> AnnData:
        """
        Predict count expression of input data with corresponding condition labels and provided target
        batch index (optional).

        Args:
            adata: AnnData to predict. If `None`, use AnnData object in model.
            indices: Indices of cells in adata to use. If `None`, use all cells.
            target_batch: Target batch index. If `None`, defaults to keep original batch index of each cell.
            batch_size: Minibatch size for data loading. If None, uses scvi.settings.batch_size.

        Returns:
            AnnData object of count expression prediction.

        """

        adata = self._validate_anndata(adata)
        data_loader = self._make_data_loader(
            adata=adata,
            indices=indices,
            batch_size=batch_size,
            shuffle=False,
            data_loader_class=AnnDataLoader,
        )

        exprs = []
        predicted_batch = []
        control_label_idx = self.module.condition2int[self.module.control]

        label_to_name = _invert_dict(self.module.condition2int)

        for tensors in data_loader:
            x = tensors[SCCAUSALVI_REGISTRY_KEYS.X_KEY]
            batch_index = tensors[SCCAUSALVI_REGISTRY_KEYS.BATCH_KEY]
            label_index = tensors[SCCAUSALVI_REGISTRY_KEYS.CONDITION_KEY]
            unique_labels = label_index.unique()

            latent_bg_tensor = torch.zeros([x.shape[0], self.module.n_background_latent], device=self.module.device)
            latent_t_tensor = torch.zeros([x.shape[0], self.module.n_te_latent], device=self.module.device)
            latent_library_tensor = torch.zeros([x.shape[0], 1], device=self.module.device)

            for lbl in unique_labels:
                mask = (label_index == lbl).squeeze(dim=-1)
                x_sub = x[mask]
                batch_sub = batch_index[mask]

                if lbl.item() == control_label_idx:
                    # Compute prediction of control data
                    outputs = self.module._generic_inference(
                        x=x_sub,
                        batch_index=batch_sub,
                        src="control",
                        condition_label=label_index[mask],
                    )
                    latent_bg_tensor[mask] = outputs["z_bg"]
                    latent_library_tensor[mask] = outputs["library"]
                else:
                    # Compute prediction of treatment data
                    treat_name = label_to_name.get(lbl.item(), None)
                    if treat_name is None:
                        raise ValueError(f"Unknown treatment: {treat_name} in model.")
                    outputs = self.module._generic_inference(
                        x=x_sub,
                        batch_index=batch_sub,
                        src="treatment",
                        condition_label=label_index[mask],
                    )
                    latent_bg_tensor[mask] = outputs["z_bg"]
                    latent_t_tensor[mask] = outputs["z_t"]
                    latent_library_tensor[mask] = outputs["library"]

            # Merge background & treatement latent representations into one tensor
            latent_tensor = torch.cat([latent_bg_tensor, latent_t_tensor], dim=-1)

            if target_batch is None:
                # Prediction under the same batch
                target_batch_index = batch_index
            else:
                # Prediction under given target batch
                target_batch_index = torch.full_like(batch_index, fill_value=target_batch)

            px_scale_tensor, px_r_tensor, px_rate_tensor, px_dropout_tensor = self.module.decoder(
                self.module.dispersion,
                latent_tensor,
                latent_library_tensor,
                target_batch_index,
            )
            if px_r_tensor is None:
                px_r_tensor = torch.exp(self.module.px_r)

            count_tensor = ZeroInflatedNegativeBinomial(
                mu=px_rate_tensor,
                theta=px_r_tensor,
                zi_logits=px_dropout_tensor,
            ).sample()

            exprs.append(count_tensor.detach().cpu())
            predicted_batch.append(target_batch_index.detach().cpu())

        expression = torch.cat(exprs, dim=0).numpy()
        predicted_batch_all = torch.cat(predicted_batch, dim=0).numpy()
        adata_out = AnnData(X=expression, obs=adata.obs.copy(), var=adata.var.copy())
        adata_out.obs['predicted_batch'] = predicted_batch_all
        return adata_out

    @torch.no_grad()
    def get_count_expression_cross_condition(
            self,
            source_condition: str,
            target_condition: str,
            target_batch: Optional[int] = None,
            adata: Optional[AnnData] = None,
            indices: Optional[List[int]] = None,
            batch_size: Optional[int] = None,
    ):
        """
        Compute count expression for cross-condition prediction.
        For cells from source_condition, computes their count data as if they were
        under target_condition. For all other cells, computes count expression under
        their original conditions.

        Args:
            source_condition: Name of source condition from which to select cells for cross-condition prediction.
            target_condition: Name of target condition under which to predict expression profiles.
            target_batch: Index of target batch under which to predict expression profiles.
            If `None`, defaults to keep the same batch index.
            adata: AnnData object to use. If `None`, use the model's AnnData object.
            indices: Indices of cells in adata to use. If `None`, uses all cells.
            batch_size: Minibatch size for data loading. If None, uses scvi.settings.batch_size.

        Returns
        -------
            AnnData with predicted cross-condition count expression as well as metadata of targeted settings.
        """
        if source_condition == target_condition:
            raise ValueError("source condition and target condition should be different.")

        adata = self._validate_anndata(adata)

        data_loader = self._make_data_loader(
            adata=adata,
            indices=indices,
            batch_size=batch_size,
            shuffle=False,
            data_loader_class=AnnDataLoader,
        )

        exprs = []
        px_rate_list = []
        px_r_list = []
        px_dropout_list = []
        predicted_batch = []
        predicted_labels = []

        control_label_idx = self.module.condition2int[self.module.control]
        source_label_idx = self.module.condition2int[source_condition]
        target_label_idx = self.module.condition2int[target_condition]

        label_to_name = _invert_dict(self.module.condition2int)

        for tensors in data_loader:
            x = tensors[SCCAUSALVI_REGISTRY_KEYS.X_KEY]
            batch_index = tensors[SCCAUSALVI_REGISTRY_KEYS.BATCH_KEY]
            label_index = tensors[SCCAUSALVI_REGISTRY_KEYS.CONDITION_KEY]
            unique_labels = label_index.unique()

            latent_bg_tensor = torch.zeros(
                [x.shape[0], self.module.n_background_latent], device=self.module.device
            )
            latent_t_tensor = torch.zeros(
                [x.shape[0], self.module.n_te_latent], device=self.module.device
            )
            latent_library_tensor = torch.zeros([x.shape[0], 1], device=self.module.device)
            predicted_label = torch.zeros([x.shape[0], 1], device=self.module.device)

            for lbl in unique_labels:
                mask = (label_index == lbl).squeeze(dim=-1)
                x_sub = x[mask]
                batch_sub = batch_index[mask]

                if lbl.item() == control_label_idx:
                    # Compute background latent factors of control data
                    outputs = self.module._generic_inference(
                        x=x_sub,
                        batch_index=batch_sub,
                        src="control",
                        condition_label=label_index[mask],
                    )
                    latent_bg_tensor[mask] = outputs["z_bg"]
                    latent_library_tensor[mask] = outputs["library"]

                    # If source_condition is control => forcibly apply target_condition's
                    # treatment effect encoder
                    if source_label_idx == control_label_idx:
                        if target_label_idx != control_label_idx:
                            target_name = label_to_name.get(target_label_idx, None)
                            if target_name is not None:
                                tm, tv, zt = self.module.treatment_te_encoders[target_name](
                                    outputs["z_bg"]
                                )
                                attn = torch.softmax(self.module.attention(zt), dim=-1)
                                zt = attn * zt
                                latent_t_tensor[mask] = zt
                                predicted_label[mask] = target_label_idx
                            else:
                                raise ValueError(f"Unknown treatment: {target_name}")
                        else:
                            raise ValueError(f"target condition should be different from source condition.")
                    else:
                        predicted_label[mask] = label_index[mask]  # Keep same condition label index
                else:
                    # Compute background latent factors of treated data
                    tname = label_to_name.get(lbl.item(), None)
                    if tname is None:
                        raise ValueError(f"Unknown treatment: {tname} in model.")
                    outputs = self.module._generic_inference(
                        x=x_sub,
                        batch_index=batch_sub,
                        src="treatment",
                        condition_label=label_index[mask],
                    )
                    latent_bg_tensor[mask] = outputs["z_bg"]

                    # If this label == source_label_idx => forcibly apply target_condition's
                    # treatment effect encoder
                    if lbl.item() == source_label_idx:
                        if target_label_idx == control_label_idx:
                            # Treated -> control => no treatment effect
                            pass
                        else:
                            target_name = label_to_name.get(target_label_idx, None)
                            if target_name is not None:
                                tm, tv, zt = self.module.treatment_te_encoders[target_name](
                                    outputs["z_bg"]
                                )
                                attn = torch.softmax(self.module.attention(zt), dim=-1)
                                zt = attn * zt
                                latent_t_tensor[mask] = zt
                        predicted_label[mask] = target_label_idx
                    else:
                        # Normal path => existing outputs
                        latent_t_tensor[mask] = outputs["z_t"]
                        predicted_label[mask] = label_index[mask]

                    latent_library_tensor[mask] = outputs["library"]

            # Decode to expression
            full_latent = torch.cat([latent_bg_tensor, latent_t_tensor], dim=-1)

            if target_batch is None:
                target_batch_index = batch_index
            else:
                if self.module.n_batch == 1:
                    print('Only one batch found in adata. Predicting under the same batch. target_batch is ignored.')
                target_batch_index = torch.full_like(batch_index, fill_value=target_batch)

            px_scale_tensor, px_r_tensor, px_rate_tensor, px_dropout_tensor = self.module.decoder(
                self.module.dispersion,
                full_latent,
                latent_library_tensor,
                target_batch_index,
            )
            if px_r_tensor is None:
                px_r_tensor = torch.exp(self.module.px_r)

            px_rate_list.append(px_rate_tensor.detach().cpu())
            px_r_list.append(px_r_tensor.detach().cpu())
            px_dropout_list.append(px_dropout_tensor.detach().cpu())

            predicted_labels.append(predicted_label.detach().cpu())
            predicted_batch.append(target_batch_index.detach().cpu())

        # Sample count data from distribution
        torch.manual_seed(0)
        expression = ZeroInflatedNegativeBinomial(
            mu=torch.cat(px_rate_list, dim=0),
            theta=px_r_tensor,
            zi_logits=torch.cat(px_dropout_list, dim=0),
        ).sample()

        predicted_condition = torch.cat(predicted_labels, dim=0).numpy()
        predicted_condition_name = [label_to_name.get(int(c), None) for c in predicted_condition]
        predicted_batch_all = torch.cat(predicted_batch, dim=0).numpy()

        adata_out = AnnData(X=expression.numpy(), obs=adata.obs.copy(), var=adata.var.copy())
        adata_out.obs['predicted_condition'] = predicted_condition_name
        adata_out.obs['predicted_batch'] = predicted_batch_all
        return adata_out

    @torch.no_grad()
    def responsive_cells(
            self,
            adata: AnnData,
            treatment_condition: str,
            control_condition: str,
            responsive_label: Optional[str] = 'if_responsive',
            multi_test_correction: Optional[bool] = False,
            target_sum: Optional[float] = 1e4,
            method: Literal['coupled', 'legacy'] = 'coupled',
            n_draws: int = 20,
            n_comps: int = 20,
            alpha: float = 0.05,
            batch_size: Optional[int] = None,
            seed: Optional[int] = 0,
    ):
        """
        Identify responsive cells of specified condition in contrast to control condition.

        For each treated cell, two distances are compared in PCA space of normalized,
        log-transformed expression:
        diff_cf   = ||x_real - \\hat x_cross_condition(control_condition)|| (treatment-induced difference) and
        diff_null = ||x_real - \\hat x_reconstructed||                     (uncertainty of the generative model).

        method='coupled' (default):
            Each of `n_draws` rounds draws one posterior sample (z_bg, z_te) per cell. The reconstruction
            is decoded from (z_bg, z_te) and the cross-condition prediction from the same z_bg (with
            z_te = 0 for the control condition), the same library size and batch. The two count matrices
            are drawn from their ZINB distributions with shared random numbers, so their difference only
            reflects the treatment effect, not independent sampling noise. PCA is fitted on observed
            expression of treated cells only and kept fixed across draws. Per cell,
            delta = diff_cf - diff_null over the `n_draws` rounds is tested with a one-sided one-sample
            t-test (H1: E[delta] > 0); cells with p < `alpha` are responsive.
            Note: the p value reflects model-sampling uncertainty over the draws, not biological
            replication. Larger `n_draws` gives smaller p values and more responsive cells, so compare
            responsive fractions only at the same `n_draws`.

        method='legacy':
            Behaviour of scCausalVI <= 0.0.11. Reconstruction and cross-condition prediction are sampled
            independently (different z_bg and ZINB draws), and diff_cf of each cell is compared against
            diff_null of all treated cells (empirical p value). The difference is dominated by sampling
            noise; kept only for reproducing earlier results.

        Args:
            adata: AnnData to predict.
            treatment_condition: Specified condition to identify responsive cells.
            control_condition: Control group to compare against.
            responsive_label: Column names in adata.obs to store labels for responsive cells.
            multi_test_correction: Whether to apply Benjamini-Hochberg correction. Default is False.
                For method='coupled' the p values depend on `n_draws`, so FDR control is not meaningful.
            target_sum: Target sum of count expression for normalization in scanpy.pp.normalize_total.
                For method='legacy' it should be consistent with the normalization of adata.X, which is
                used as observed expression. method='coupled' normalizes the registered count layer itself.
            method: 'coupled' (default) or 'legacy'. See above.
            n_draws: Number of coupled posterior draws per cell. Only used for method='coupled'.
            n_comps: Number of principal components.
            alpha: Significance level for calling responsive cells.
            batch_size: Minibatch size for data loading. Only used for method='coupled'.
            seed: Random seed for posterior and count sampling. Only used for method='coupled'.
                The global torch random state is left unchanged. If None, the current random state is used.

        Returns:
            AnnData of cells of treatment_condition with labels in .obs[responsive_label] ('True' / 'False'),
            .obs['-log p values'], and, for method='coupled', .obs['p_value'], .obs['diff_null'],
            .obs['diff_cf'] (means over draws) and .obs['delta'] (mean of diff_cf - diff_null).
        """
        if treatment_condition == control_condition:
            raise ValueError("treatment_condition and control_condition should be different.")

        if treatment_condition not in self.module.condition2int.keys() or control_condition not in self.module.condition2int.keys():
            raise ValueError("treatment_condition or control_condition is not valid conditions.")

        if treatment_condition == self.module.control:
            raise ValueError("treatment_condition should not be the control condition of the model.")

        if method not in ('coupled', 'legacy'):
            raise ValueError("method should be 'coupled' or 'legacy'.")

        tm_mask = (adata.obs['_scvi_condition'] == self.module.condition2int[treatment_condition]).values
        adata_tm = adata[tm_mask].copy()
        adata_tm.obs['is_real'] = 'real tm'

        if method == 'legacy':
            p_values = self._responsive_pvalues_legacy(
                adata_tm, treatment_condition, control_condition, target_sum, n_comps,
            )
        else:
            p_values, diff_null, diff_cf, delta = self._responsive_pvalues_coupled(
                adata, np.where(tm_mask)[0], treatment_condition, control_condition,
                target_sum, n_draws, n_comps, batch_size, seed,
            )
            adata_tm.obs['p_value'] = p_values
            adata_tm.obs['diff_null'] = diff_null
            adata_tm.obs['diff_cf'] = diff_cf
            adata_tm.obs['delta'] = delta

        # Apply multi-testing correction (e.g., Benjamini-Hochberg)
        if multi_test_correction:
            reject, pvals_corrected, _, _ = multipletests(p_values, method='fdr_bh', alpha=alpha)
            significant_cells = np.where(reject)[0]
        else:
            pvals_corrected = np.asarray(p_values)
            significant_cells = np.where(pvals_corrected < alpha)[0]

        adata_tm.obs['-log p values'] = -np.log(pvals_corrected + 1)
        labels = np.full(adata_tm.n_obs, 'False', dtype=object)
        labels[significant_cells] = 'True'
        adata_tm.obs[responsive_label] = labels

        return adata_tm

    def _responsive_pvalues_legacy(
            self,
            adata_tm: AnnData,
            treatment_condition: str,
            control_condition: str,
            target_sum: float,
            n_comps: int,
    ) -> np.ndarray:
        """Empirical p values of scCausalVI <= 0.0.11 (independent sampling, pooled null)."""
        adata_cross = self.get_count_expression_cross_condition(
            adata=adata_tm,
            source_condition=treatment_condition,
            target_condition=control_condition,
        )

        sc.pp.normalize_total(adata_cross, target_sum=target_sum)
        sc.pp.log1p(adata_cross)
        adata_cross.obs['is_real'] = 'tm -> ctrl'

        adata_recon = self.get_count_expression(adata_tm)

        sc.pp.normalize_total(adata_recon, target_sum=target_sum)
        sc.pp.log1p(adata_recon)
        adata_recon.obs['is_real'] = 'tm -> tm'

        stim_real_pred = ad.concat([adata_tm, adata_recon, adata_cross])

        sc.pp.pca(stim_real_pred, n_comps=n_comps)

        diff_null = stim_real_pred[stim_real_pred.obs['is_real'] == "real tm"].obsm['X_pca'] - \
                    stim_real_pred[stim_real_pred.obs['is_real'] == "tm -> tm"].obsm['X_pca']

        l2_norm_null = np.linalg.norm(diff_null, axis=1)

        diff_cf = stim_real_pred[stim_real_pred.obs['is_real'] == "real tm"].obsm['X_pca'] - \
                  stim_real_pred[stim_real_pred.obs['is_real'] == "tm -> ctrl"].obsm['X_pca']
        l2_norm_cf = np.linalg.norm(diff_cf, axis=1)

        # Use the distribution of null hypothesis (l2_norm_null) to test the significance of l2_norm_cf
        n = len(diff_null)
        p_values = []
        for l in l2_norm_cf:
            extreme_count = np.sum(l2_norm_null >= l)
            p_values.append((extreme_count + 1) / (n + 1))
        return np.array(p_values)

    def _responsive_pvalues_coupled(
            self,
            adata: AnnData,
            indices: np.ndarray,
            treatment_condition: str,
            control_condition: str,
            target_sum: float,
            n_draws: int,
            n_comps: int,
            batch_size: Optional[int],
            seed: Optional[int],
    ):
        """Per-cell paired test on coupled reconstruction / cross-condition draws."""
        if n_draws < 2:
            raise ValueError("n_draws should be at least 2 for the per-cell t-test.")

        adata = self._validate_anndata(adata)
        data_loader = self._make_data_loader(
            adata=adata,
            indices=indices,
            batch_size=batch_size,
            shuffle=False,
            data_loader_class=AnnDataLoader,
        )
        self.module.eval()
        device = self.module.device

        # Fixed projection: PCA fitted on observed expression of treated cells only
        x_obs = np.concatenate(
            [_log_normalize(t[SCCAUSALVI_REGISTRY_KEYS.X_KEY].float(), target_sum) for t in data_loader]
        )
        pca = PCA(n_components=n_comps, random_state=0 if seed is None else seed)
        pc_obs = pca.fit_transform(x_obs)
        del x_obs

        control_idx = self.module.condition2int[self.module.control]
        target_idx = self.module.condition2int[control_condition]
        target_te_encoder = None
        if target_idx != control_idx:
            target_te_encoder = self.module.treatment_te_encoders[control_condition]
        theta = torch.exp(self.module.px_r)

        def te_latent(z_t):
            return torch.softmax(self.module.attention(z_t), dim=-1) * z_t

        d_null = np.zeros((len(indices), n_draws), dtype=np.float32)
        d_cf = np.zeros((len(indices), n_draws), dtype=np.float32)

        fork_devices = [device] if device.type == 'cuda' else []
        with torch.random.fork_rng(devices=fork_devices, enabled=seed is not None):
            if seed is not None:
                torch.manual_seed(seed)
            start = 0
            for tensors in data_loader:
                x = tensors[SCCAUSALVI_REGISTRY_KEYS.X_KEY].to(device)
                batch_index = tensors[SCCAUSALVI_REGISTRY_KEYS.BATCH_KEY].to(device)
                label_index = tensors[SCCAUSALVI_REGISTRY_KEYS.CONDITION_KEY].to(device)
                stop = start + x.shape[0]
                pc_x = pc_obs[start:stop]

                for k in range(n_draws):
                    # One posterior sample (z_bg, z_te) shared by reconstruction and cross-condition prediction
                    outputs = self.module._generic_inference(
                        x=x, batch_index=batch_index, condition_label=label_index, src='treatment',
                    )
                    z_bg = outputs['z_bg']
                    if target_te_encoder is None:
                        z_t_cf = torch.zeros_like(outputs['z_t'])
                    else:
                        z_t_cf = te_latent(target_te_encoder(z_bg)[2])

                    latent_rc = torch.cat([z_bg, te_latent(outputs['z_t'])], dim=-1)
                    latent_cf = torch.cat([z_bg, z_t_cf], dim=-1)
                    _, _, rate_rc, drop_rc = self.module.decoder(
                        self.module.dispersion, latent_rc, outputs['library'], batch_index,
                    )
                    _, _, rate_cf, drop_cf = self.module.decoder(
                        self.module.dispersion, latent_cf, outputs['library'], batch_index,
                    )
                    x_rc, x_cf = _coupled_zinb_sample(rate_rc, rate_cf, drop_rc, drop_cf, theta)

                    pc_rc = pca.transform(_log_normalize(x_rc, target_sum))
                    pc_cf = pca.transform(_log_normalize(x_cf, target_sum))
                    d_null[start:stop, k] = np.linalg.norm(pc_x - pc_rc, axis=1)
                    d_cf[start:stop, k] = np.linalg.norm(pc_x - pc_cf, axis=1)
                start = stop

        delta = d_cf - d_null
        p_values = ttest_1samp(delta, 0.0, axis=1, alternative='greater').pvalue
        # Zero variance across draws (e.g. no counts): no evidence for a response
        p_values = np.where(np.isnan(p_values), 1.0, p_values)
        return p_values, d_null.mean(1), d_cf.mean(1), delta.mean(1)
