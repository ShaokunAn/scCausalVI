from typing import Dict, Optional, Tuple, Union, List, Any

import numpy as np
import torch
import torch.nn.functional as F
from scCausalVI.model.base import SCCAUSALVI_REGISTRY_KEYS
from scvi.distributions import ZeroInflatedNegativeBinomial
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from scvi.nn import DecoderSCVI, Encoder, one_hot
from torch import Tensor
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl

from scCausalVI.data.utils import gram_matrix


class scCausalVIModule(BaseModuleClass):
    """
    PyTorch module for scCausalVI (Variational Inference).

    Args:
    ----
        n_input: Number of input genes.
        condition2int: Dict mapping condition name (str) -> index (int)
        control: Control condition in case-control study, containing cells in unperturbed states
        n_batch: Number of batches. If 1, no batch information incorporated into model/
        n_hidden: Number of hidden nodes in each layer of neural network.
        n_background_latent: Dimensionality of the background latent space.
        n_te_latent: Dimensionality of the treatment effect latent space.
        n_layers: Number of hidden layers of each sub-networks.
        dropout_rate: Dropout rate for neural networks.
        use_observed_lib_size: Use observed library size for RNA as scaling factor in
            mean of conditional distribution.
        library_log_means: 1 x n_batch array of means of the log library sizes.
            Parameterize prior on library size if not using observed library size.
        library_log_vars: 1 x n_batch array of variances of the log library sizes.
            Parameterize prior on library size if not using observed library size.
        use_mmd: Whether to use the maximum mean discrepancy to force background latent
            variables of the control and treatment dataset to follow the same
            distribution.
        mmd_weight: Weight of MMD in loss function.
        norm_weight: Normalization weight in loss function.
        gammas: Kernel bandwidths for calculating MMD.
    """

    def sample(self, *args, **kwargs):
        pass

    def __init__(
            self,
            n_input: int,
            condition2int: dict,
            control: str,
            n_batch: int,
            n_hidden: int = 128,
            n_background_latent: int = 10,
            n_te_latent: int = 10,
            n_layers: int = 1,
            dropout_rate: float = 0.1,
            use_observed_lib_size: bool = True,
            library_log_means: Optional[np.ndarray] = None,
            library_log_vars: Optional[np.ndarray] = None,
            use_mmd: bool = True,
            mmd_weight: float = 1.0,
            norm_weight: float = 1.0,
            gammas: Optional[np.ndarray] = None,
    ) -> None:
        super(scCausalVIModule, self).__init__()
        self.n_input = n_input
        self.control = control
        self.condition2int = condition2int
        self.n_conditions = len(condition2int)
        self.treat_ind = [i for i in range(self.n_conditions) if i != self.condition2int[self.control]]
        self.n_batch = n_batch
        self.n_hidden = n_hidden
        self.n_background_latent = n_background_latent
        self.n_te_latent = n_te_latent
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.latent_distribution = "normal"
        self.dispersion = "gene"
        self.px_r = torch.nn.Parameter(torch.randn(n_input))
        self.use_observed_lib_size = use_observed_lib_size
        self.use_mmd = use_mmd
        self.mmd_weight = mmd_weight
        self.norm_weight = norm_weight
        self.gammas = gammas

        if not self.use_observed_lib_size:
            if library_log_means is None or library_log_vars is None:
                raise ValueError(
                    "If not using observed_lib_size, "
                    "must provide library_log_means and library_log_vars."
                )
            self.register_buffer(
                "library_log_means", torch.from_numpy(library_log_means).float()
            )
            self.register_buffer(
                "library_log_vars", torch.from_numpy(library_log_vars).float()
            )

        if use_mmd:
            if gammas is None:
                raise ValueError("If using mmd, must provide gammas.")
            # self.register_buffer("gammas", torch.from_numpy(gammas).float())

        cat_list = [n_batch]

        # Background encoder encodes cellular baseline states
        # of control data. Input dim equals to the number of genes.
        self.control_background_encoder = Encoder(
            n_input,
            n_background_latent,
            n_cat_list=cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=self.latent_distribution,
            inject_covariates=True,
            use_batch_norm=True,
            use_layer_norm=False,
            var_activation=None,
        )

        treatment_names = [k for k in self.condition2int.keys() if k != control]
        self.treatment_names = treatment_names

        # Each treatment_background_encoder encodes baseline states of treated cells without treatment effect
        self.treatment_background_encoders = torch.nn.ModuleDict()
        for tname in self.treatment_names:
            enc = Encoder(
                n_input,
                n_background_latent,
                n_cat_list=cat_list,
                n_layers=n_layers,
                n_hidden=n_hidden,
                dropout_rate=dropout_rate,
                distribution=self.latent_distribution,
                inject_covariates=True,
                use_batch_norm=True,
                use_layer_norm=False,
                var_activation=None,
            )
            # Copy weights from control
            enc.load_state_dict(self.control_background_encoder.state_dict())
            self.treatment_background_encoders[tname] = enc

        self.treatment_te_encoders = torch.nn.ModuleDict()

        if len(self.treatment_names) > 0:
            # Create a reference treatment effect encoder
            ref_te = Encoder(
                n_input=n_background_latent,
                n_output=n_te_latent,
                n_cat_list=None,
                n_layers=n_layers,
                n_hidden=n_hidden,
                dropout_rate=dropout_rate,
                distribution=self.latent_distribution,
                inject_covariates=False,
                use_batch_norm=True,
                use_layer_norm=False,
                var_activation=None,
            )

            # First one is the reference
            self.treatment_te_encoders[self.treatment_names[0]] = ref_te
            # Others copy from reference
            for tname in self.treatment_names[1:]:
                enc = Encoder(
                    n_input=n_background_latent,
                    n_output=n_te_latent,
                    n_cat_list=None,
                    n_layers=n_layers,
                    n_hidden=n_hidden,
                    dropout_rate=dropout_rate,
                    distribution=self.latent_distribution,
                    inject_covariates=False,
                    use_batch_norm=True,
                    use_layer_norm=False,
                    var_activation=None,
                )
                enc.load_state_dict(ref_te.state_dict())
                self.treatment_te_encoders[tname] = enc

        # Attention layer to capture differential treatment effect patterns
        self.attention = torch.nn.Linear(self.n_te_latent, 1)

        # Library size encoder.
        self.l_encoder = Encoder(
            n_input,
            n_output=1,
            n_layers=1,
            n_cat_list=cat_list,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=True,
            use_batch_norm=True,
            use_layer_norm=False,
            var_activation=None,
        )
        # Decoder from latent variable to distribution parameters in data space.
        n_total_latent = n_background_latent + n_te_latent
        self.decoder = DecoderSCVI(
            n_total_latent,
            n_input,
            n_cat_list=cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            inject_covariates=True,
            use_batch_norm=True,
            use_layer_norm=True,
        )

    @auto_move_data
    def _compute_local_library_params(
            self, batch_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_batch = self.library_log_means.shape[1]
        local_library_log_means = F.linear(
            one_hot(batch_index, n_batch), self.library_log_means
        )
        local_library_log_vars = F.linear(
            one_hot(batch_index, n_batch), self.library_log_vars
        )
        return local_library_log_means, local_library_log_vars

    def _get_inference_input(
            self,
            tensors: Dict[str, torch.Tensor],
            **kwargs
    ) -> Union[Dict[str, list[str]], Dict[str, Any]]:

        x = tensors[SCCAUSALVI_REGISTRY_KEYS.X_KEY]
        batch_index = tensors[SCCAUSALVI_REGISTRY_KEYS.BATCH_KEY]
        condition_label = tensors[SCCAUSALVI_REGISTRY_KEYS.CONDITION_KEY]

        out = dict(x=x, condition_label=condition_label, batch_index=batch_index,)
        return out

    @auto_move_data
    def _generic_inference(
            self,
            x: torch.Tensor,
            batch_index: torch.Tensor,
            condition_label: torch.Tensor,
            src: str,
    ) -> Dict[str, torch.Tensor]:
        """
        If src = 'control', use self.control_background_encoder.
        If src = 'treatment', for each unique treat label in `condition_label`,
        use that label's background & treatment_effect encoders.
        """

        n_cells = x.shape[0]
        x_ = torch.log(x + 1)

        z_bg = torch.zeros((n_cells, self.n_background_latent), device=x.device)
        qbg_m = torch.zeros((n_cells, self.n_background_latent), device=x.device)
        qbg_v = torch.zeros((n_cells, self.n_background_latent), device=x.device)

        z_t = torch.zeros((n_cells, self.n_te_latent), device=x.device)
        qt_m = torch.zeros((n_cells, self.n_te_latent), device=x.device)
        qt_v = torch.zeros((n_cells, self.n_te_latent), device=x.device)

        library = torch.zeros((n_cells, 1), device=x.device)
        ql_m = None
        ql_v = None

        if not self.use_observed_lib_size:
            ql_m = torch.zeros((n_cells, 1), device=x.device)
            ql_v = torch.zeros((n_cells, 1), device=x.device)

        if src == 'control':
            if self.use_observed_lib_size:
                lib_ = torch.log(x.sum(dim=1, keepdim=True) + 1e-8)
                library[:] = lib_
            else:
                qlm, qlv, lib_ = self.l_encoder(x_, batch_index)
                ql_m[:] = qlm
                ql_v[:] = qlv
                library[:] = lib_

            # Background latent factors
            bg_m, bg_v_, bg_z = self.control_background_encoder(x_, batch_index)
            qbg_m[:] = bg_m
            qbg_v[:] = bg_v_
            z_bg[:] = bg_z
            # Treatment effect latent factors remain 0 of control data.
        else:
            # Multiple distinct treatment labels
            unique_treats = condition_label.unique()
            for t_lbl in unique_treats:
                if t_lbl.item() == self.condition2int[self.control]:
                    raise ValueError('Control label found in treatment labels')  # skip control label if present by chance.

                # Invert condition2int to find the name for t_lbl
                treat_name = None
                for k, v in self.condition2int.items():
                    if v == t_lbl.item():
                        treat_name = k
                        break
                if treat_name is None:
                    raise ValueError(f'Treatment label {t_lbl} not found in condition2int')

                mask = (condition_label == t_lbl).squeeze(dim=-1)
                x_sub = x_[mask]
                batch_sub = batch_index[mask]

                if self.use_observed_lib_size:
                    lib_ = torch.log(x_sub.sum(dim=1, keepdim=True) + 1e-8)
                    library[mask] = lib_
                else:
                    qlm, qlv, lib_ = self.l_encoder(x_sub, batch_sub)
                    library[mask] = lib_
                    ql_m[mask] = qlm
                    ql_v[mask] = qlv

                bg_enc = self.treatment_background_encoders[treat_name]
                bg_m, bg_v_, bg_z = bg_enc(x_sub, batch_sub)
                qbg_m[mask] = bg_m
                qbg_v[mask] = bg_v_
                z_bg[mask] = bg_z

                # Treatment-specific module
                s_enc = self.treatment_te_encoders[treat_name]
                tm, tv, zt = s_enc(bg_z)
                qt_m[mask] = tm
                qt_v[mask] = tv
                z_t[mask] = zt

        return {
            "z_bg": z_bg,
            "qbg_m": qbg_m,
            "qbg_v": qbg_v,
            "z_t": z_t,
            "qt_m": qt_m,
            "qt_v": qt_v,
            "library": library,
            "ql_m": ql_m,
            "ql_v": ql_v,
        }

    @auto_move_data
    def inference(
            self,
            x: torch.Tensor,
            condition_label: torch.Tensor,
            batch_index: torch.Tensor,
            n_samples: int = 1,
    ) -> Dict[str, Dict[str, torch.Tensor]]:

        # Inference of data
        ctrl_mask = (condition_label == self.condition2int[self.control]).squeeze(dim=-1)

        x_control = x[ctrl_mask]
        condition_control = condition_label[ctrl_mask]
        batch_control = batch_index[ctrl_mask]

        x_treatment = x[~ctrl_mask]
        condition_treatment = condition_label[~ctrl_mask]
        batch_treatment = batch_index[~ctrl_mask]

        inference_control = self._generic_inference(
            x_control, batch_control, src='control', condition_label=condition_control
        )
        inference_treatment = self._generic_inference(
            x_treatment, batch_treatment, src='treatment', condition_label=condition_treatment
        )

        return {"control": inference_control, "treatment": inference_treatment}

    def _get_generative_input(
            self,
            tensors: torch.Tensor,
            inference_outputs: Dict[str, Dict[str, torch.Tensor]],
            **kwargs,
    ):
        """
        Merges the control/treatment in original order.
        """
        x = tensors[SCCAUSALVI_REGISTRY_KEYS.X_KEY]
        batch_index = tensors[SCCAUSALVI_REGISTRY_KEYS. BATCH_KEY]
        condition_label = tensors[SCCAUSALVI_REGISTRY_KEYS.CONDITION_KEY]

        ctrl_mask = (condition_label == self.condition2int[self.control]).squeeze(dim=-1)
        n_cells = x.shape[0]

        z_bg_merged = torch.zeros((n_cells, self.n_background_latent), device=x.device)
        z_t_merged = torch.zeros((n_cells, self.n_te_latent), device=x.device)
        library_merged = torch.zeros((n_cells, 1), device=x.device)

        ctrl_inference = inference_outputs['control']
        treatment_inference = inference_outputs['treatment']

        # Fill control portion
        z_bg_merged[ctrl_mask] = ctrl_inference['z_bg']
        z_t_merged[ctrl_mask] = ctrl_inference['z_t']
        library_merged[ctrl_mask] = ctrl_inference['library']

        # Fill treatment portion
        z_bg_merged[~ctrl_mask] = treatment_inference['z_bg']
        z_t_merged[~ctrl_mask] = treatment_inference['z_t']
        library_merged[~ctrl_mask] = treatment_inference['library']

        return {
            'z_bg': z_bg_merged,
            'z_t': z_t_merged,
            'library': library_merged,
            'batch_index': batch_index,
        }

    @auto_move_data
    def generative(
            self,
            z_bg: torch.Tensor,
            z_t: torch.Tensor,
            library: torch.Tensor,
            batch_index: List[int],
    ) -> Dict[str, Dict[str, torch.Tensor]]:

        attention_weights = torch.softmax(self.attention(z_t), dim=-1)
        z_t = attention_weights * z_t
        latent = torch.cat([z_bg, z_t], dim=-1)
        px_scale, px_r, px_rate, px_dropout = self.decoder(
            self.dispersion,
            latent,
            library,
            batch_index,
        )
        px_r = torch.exp(self.px_r)
        return {
            'px_scale': px_scale,
            'px_r': px_r,
            'px_rate': px_rate,
            'px_dropout': px_dropout,
        }

    @staticmethod
    def reconstruction_loss(
            x: torch.Tensor,
            px_rate: torch.Tensor,
            px_r: torch.Tensor,
            px_dropout: torch.Tensor,
    ) -> torch.Tensor:
        if x.shape[0] != px_rate.shape[0]:
            print(f"x.shape[0]= {x.shape[0]} and px_rate.shape[0]= {px_rate.shape[0]}.")
        recon_loss = (
            -ZeroInflatedNegativeBinomial(mu=px_rate, theta=px_r, zi_logits=px_dropout)
            .log_prob(x)
            .sum(dim=-1)
        )
        return recon_loss

    @staticmethod
    def latent_kl_divergence(
            variational_mean: torch.Tensor,
            variational_var: torch.Tensor,
            prior_mean: torch.Tensor,
            prior_var: torch.Tensor,
    ) -> torch.Tensor:
        return kl(
            Normal(variational_mean, variational_var.sqrt()),
            Normal(prior_mean, prior_var.sqrt()),
        ).sum(dim=-1)

    def library_kl_divergence(
            self,
            batch_index: torch.Tensor,
            variational_library_mean: torch.Tensor,
            variational_library_var: torch.Tensor,
            library: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_observed_lib_size:
            (
                local_library_log_means,
                local_library_log_vars,
            ) = self._compute_local_library_params(batch_index)

            kl_library = kl(
                Normal(variational_library_mean, variational_library_var.sqrt()),
                Normal(local_library_log_means, local_library_log_vars.sqrt()),
            )
        else:
            kl_library = torch.zeros_like(library)
        return kl_library.sum(dim=-1)

    def mmd_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cost = torch.mean(gram_matrix(x, x, gammas=self.gammas.to(self.device)))
        cost += torch.mean(gram_matrix(y, y, gammas=self.gammas.to(self.device)))
        cost -= 2 * torch.mean(gram_matrix(x, y, gammas=self.gammas.to(self.device)))
        if cost < 0:  # Handle numerical instability.
            return torch.tensor(0)
        return cost

    def _generic_loss(
            self,
            x: torch.Tensor,
            batch_index: torch.Tensor,
            qbg_m: torch.Tensor,
            qbg_v: torch.Tensor,
            library: torch.Tensor,
            ql_m: Optional[torch.Tensor],
            ql_v: Optional[torch.Tensor],
            px_rate: torch.Tensor,
            px_r: torch.Tensor,
            px_dropout: torch.Tensor,
    ) -> dict[str, Union[Tensor, list[Tensor]]]:
        prior_bg_m = torch.zeros_like(qbg_m)
        prior_bg_v = torch.ones_like(qbg_v)

        recon_loss = self.reconstruction_loss(x, px_rate, px_r, px_dropout)
        kl_bg = self.latent_kl_divergence(qbg_m, qbg_v, prior_bg_m, prior_bg_v)

        if not self.use_observed_lib_size:
            kl_library = self.library_kl_divergence(batch_index, ql_m, ql_v, library)
        else:
            kl_library = torch.zeros_like(recon_loss)

        return {
            'recon_loss': recon_loss,
            'kl_bg': kl_bg,
            'kl_library': kl_library,
        }

    def loss(
            self,
            tensors: Dict[str, torch.Tensor],
            inference_outputs: Dict[str, Dict[str, torch.Tensor]],
            generative_outputs: Dict[str, torch.Tensor],
            **loss_args,
    ) -> LossOutput:
        """
        The entire batch is in `tensors`.
        We separate  control vs. treat, compute the relevant losses, and combine.
        """
        x = tensors[SCCAUSALVI_REGISTRY_KEYS.X_KEY]
        batch_index = tensors[SCCAUSALVI_REGISTRY_KEYS.BATCH_KEY]
        condition_label = tensors[SCCAUSALVI_REGISTRY_KEYS.CONDITION_KEY]

        ctrl_mask = (condition_label == self.condition2int[self.control]).squeeze(dim=-1)
        x_ctrl = x[ctrl_mask]
        batch_ctrl = batch_index[ctrl_mask]
        ctrl_inference = inference_outputs['control']

        x_trt = x[~ctrl_mask]
        batch_trt = batch_index[~ctrl_mask]
        trt_inference = inference_outputs['treatment']

        # generative outputs are for all cells in order.
        px_rate = generative_outputs['px_rate']
        px_r = generative_outputs['px_r']
        px_dropout = generative_outputs['px_dropout']

        # separate them
        px_rate_ctrl = px_rate[ctrl_mask]
        px_dropout_ctrl = px_dropout[ctrl_mask]
        px_rate_trt = px_rate[~ctrl_mask]
        px_dropout_trt = px_dropout[~ctrl_mask]

        # ELBO loss of control data
        elbo_ctrl = self._generic_loss(
            x_ctrl,
            batch_ctrl,
            ctrl_inference['qbg_m'],
            ctrl_inference['qbg_v'],
            ctrl_inference['library'],
            ctrl_inference['ql_m'],
            ctrl_inference['ql_v'],
            px_rate_ctrl,
            px_r,
            px_dropout_ctrl,
        )

        # ELBO loss of treatment data
        elbo_trt = self._generic_loss(
            x_trt,
            batch_trt,
            trt_inference['qbg_m'],
            trt_inference['qbg_v'],
            trt_inference['library'],
            trt_inference['ql_m'],
            trt_inference['ql_v'],
            px_rate_trt,
            px_r,
            px_dropout_trt,
        )

        recon_loss = torch.cat([elbo_ctrl['recon_loss'], elbo_trt['recon_loss']], dim=0)
        kl_bg = torch.cat([elbo_ctrl['kl_bg'], elbo_trt['kl_bg']], dim=0)
        kl_library = torch.cat([elbo_ctrl['kl_library'], elbo_trt['kl_library']], dim=0)

        # MMD loss
        loss_mmd = torch.tensor(0.0, device=x.device)
        if self.use_mmd:
            z_bg_control = ctrl_inference["z_bg"]
            z_bg_treatment_all = trt_inference["z_bg"]
            cond_treat = condition_label[~ctrl_mask]

            # Compute MMD loss between distributions of background latent space for control and
            # each treatment data, to align each baseline states of treated samples with the control population
            unique_treats = cond_treat.unique()
            for t_lbl in unique_treats:
                treat_submask = (cond_treat == t_lbl).squeeze(dim=-1)
                z_bg_t_sub = z_bg_treatment_all[treat_submask]
                # MMD between all control cells and the subset of treatment cells of this label
                loss_mmd += self.mmd_loss(z_bg_control, z_bg_t_sub)

            loss_mmd *= self.mmd_weight

        # Norm cost, e.g. L2 on z_t in treatments.
        loss_norm = torch.tensor(0.0, device=x.device)
        if self.norm_weight > 0:
            z_t_trt = trt_inference['z_t']
            norm_val = (z_t_trt ** 2).sum(dim=1)
            loss_norm = self.norm_weight * norm_val

        # Summation
        total_loss = recon_loss.mean() + kl_bg.mean() + kl_library.mean() + loss_mmd + loss_norm.mean()

        kl_local = {
            'loss_kl_bg': kl_bg,
            'loss_kl_l': kl_library,
            'loss_mmd': loss_mmd,
        }

        return LossOutput(
            loss=total_loss,
            reconstruction_loss=recon_loss,
            kl_local=kl_local,
        )