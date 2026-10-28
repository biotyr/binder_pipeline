# /// script
# requires-python = "<=3.13"
# dependencies = [
#     "abnumber",
#     "esm@git+https://github.com/Biohub/esm.git@main",
#     "modal",
# ]
# ///
"""
Code for binder design with ESMFold2 and ESMC.

As described in [Language Modeling Materializes a World Model of Protein Biology](https://www.biorxiv.org/content/10.64898/2026.06.03.729735).
"""

import logging
import math
import os
import random
import string
import time
from dataclasses import dataclass
from functools import cache
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import biotite.structure
import modal
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from transformers.models.esmc.modeling_esmc import ESMCForMaskedLM
from transformers.models.esmc.tokenization_esmc import ESMCTokenizer
from transformers.models.esmfold2.modeling_esmfold2_common import (
    BACKEND_CUEQ,
    BACKEND_FUSED,
    CUE_AVAILABLE,
    TRITON_KERNELS_AVAILABLE,
    PairUpdateBlock,
)
from transformers.models.esmfold2.modeling_esmfold2_common import (
    _seed_context as seed_context,
)
from transformers.models.esmfold2.modeling_esmfold2_experimental import (
    ESMFold2ExperimentalModel,
)
from transformers.models.esmfold2.modeling_esmfold2_experimental import (
    MSAEncoder as ESMFold2MSAEncoder,
)

from esm.models.esmfold2 import (
    ELEMENT_NUMBER_TO_SYMBOL,
    ProteinInput,
    StructurePredictionInput,
    load_ccd,
    prepare_esmfold2_input,
)
from esm.models.esmfold2.constants import (
    MOL_TYPE_NONPOLYMER,
    PROTEIN_1TO3,
    PROTEIN_3TO1,
    RES_TYPE_TO_CCD,
)
from esm.utils.structure.mmcif_parsing import PLDDT_B_FACTOR_SCALE
from esm.utils.structure.protein_chain import ProteinChain
from esm.utils.structure.protein_complex import ProteinComplex

os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

TrajectoryStep = dict[str, torch.Tensor | float]
Trajectory = dict[int, TrajectoryStep]


# ---- Constants ----


# General
TOKENS = ["<pad>", "-"] + [RES_TYPE_TO_CCD[i] for i in range(2, 33)]
ELEMENTS = ["X"] * (max(ELEMENT_NUMBER_TO_SYMBOL) + 1)
ELEMENTS[0] = "<pad>"
for _atomic_num, _symbol in ELEMENT_NUMBER_TO_SYMBOL.items():
    ELEMENTS[_atomic_num] = _symbol[:1] + _symbol[1:].lower()
TOKEN_IDS = {token: idx for idx, token in enumerate(TOKENS)}
AA_DIMS = 20
# Cysteine index in the 20-dim AA space (TOKEN_IDS are offset by 2 for <pad> and -)
CYS_IDX = TOKEN_IDS[PROTEIN_1TO3["C"]] - 2
MUTABLE_TOKEN = "#"
# Contains AA chars at fixed positions and MUTABLE_TOKEN at mutable positions.
BinderPromptStr = str

# Design
LOSS_WEIGHTS = {"intra_contact": 0.5, "inter_contact": 0.5, "glob": 0.2, "epitope": 0.5}
STEPS = 150
# Hotspot-neighborhood inter-loss mode: a parallel implementation of the
# inter contact loss that averages over surface-accessible target residues
# near the hotspots instead of every target residue. Opt-in per run via
# ``inter_loss_mode="hotspot_neighborhood"``; the default full-coverage
# behavior is unchanged. Thresholds are provisional defaults — calibration
# pending.
INTER_NEIGHBORHOOD_RADIUS = 15.0      # Å: expected distogram distance to any hotspot
INTER_BURIAL_CONTACT_DISTANCE = 10.0  # Å: distogram cutoff of the contact-number burial proxy
INTER_BURIAL_QUANTILE = 0.5           # burial quantile below which a residue counts as accessible
INTER_BURIAL_MIN_SEP = 2              # sequence pairs closer than this are excluded from burial
INTER_NEIGHBORHOOD_MIN_SIZE = 10      # fewer selected residues: fall back to full coverage
# Fold-rescue phase schedule (hotspot designs only; opt-in via
# ``fold_rescue_schedule=True``). Phase 1 runs the native LOSS_WEIGHTS; if
# binder folding stalls (intra contact-loss plateau at the checkpoint steps),
# phase 2 hands the optimization to folding with residual hotspot anchoring;
# once folding completes (intra mean below the phase-3 threshold), phase 3
# restores full hotspot weights for rigid-body interface refinement. Phase
# transitions only shift loss weights — everything else is untouched.
# Calibrated on 4ZQK junction pairs (2026-10): P2 rescue 3/3, low P2
# anchoring avoids the drift fuel of higher anchor weights.
FOLD_RESCUE_CHECKPOINT_STEPS = (25, 35)   # 0-indexed steps of the P1 plateau check
FOLD_RESCUE_INTRA_TRIGGER = 3.8           # last-5 mean above this = folding plateau
FOLD_RESCUE_P2_WEIGHTS = {"inter_contact": 0.1, "epitope": 0.1}
FOLD_RESCUE_P3_MIN_STEP = 50              # earliest phase-3 evaluation step
FOLD_RESCUE_P3_WINDOW = 10                # last-N steps averaged for the phase-3 test
FOLD_RESCUE_P3_INTRA = 3.0                # last-N mean below this = folding complete
FOLD_RESCUE_P3_WEIGHTS = {"inter_contact": 0.5, "epitope": 0.5}
LOG_INTERVAL = 5
LEARNING_RATE = 0.1
TEMPERATURE_MIN = 1e-2
# PDB REMARK number for predicted quality metrics (AlphaFold convention: 250)
PDB_QUALITY_REMARK_NUM = "250"
ESMC_MASK_FRACTION = 0.15
LM_LOSS_BATCH_SIZE = 128
LM_MASK_PASSES = 4
BYTES_PER_GIB = 1024**3
COMPILE = True
# NOTE - This significantly reduces VRAM usage.
# On config (target_name="cd45", binder_name="trastuzumab_framework_vhvl", batch_size=1)
# this reduces VRAM from 51GB -> 27GB.  And enables increasing batch size up to 6.
# We are testing this setting in silico, and may change the default to True, in the future.
REUSE_ESMC = True


# ---- Prompts ----


@dataclass(frozen=True)
class PromptFactory:
    """A simple factory for making binder prompt strings."""

    name: str
    template: str  # string with format fields
    length_ranges: dict[str, tuple[int, int]]  # map from field name tp length range
    is_antibody: bool  # Used to set LM loss weight for antibodies.

    def sample(self, seed: int) -> BinderPromptStr:
        random.seed(seed)
        return self.template.format(
            **{
                key: MUTABLE_TOKEN * random.randint(low, high)
                for key, (low, high) in self.length_ranges.items()
            }
        )


# fmt: off
BINDER_PROMPT_FACTORIES = {
    "minibinder": PromptFactory(name="minibinder", template="{seq}", length_ranges={"seq": (60, 200)}, is_antibody=False),
    "trastuzumab_framework_vhvl": PromptFactory(
        name="trastuzumab_framework_vhvl",
        template="EVQLVESGGGLVQPGGSLRLSCAAS{hcdr1}YIHWVRQAPGKGLEWVARI{hcdr2}TRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSR{hcdr3}WGQGTLVTVSSGGGSGGGSGGGSGGGSDIQMTQSPSSLSASVGDRVTITC{lcdr1}WYQQKPGKAPKLLIY{lcdr2}GVPSRFSGSRSGTDFTLTISSLQPEDFATYYC{lcdr3}FGQGTKVEIK",
        length_ranges = {"hcdr1": (7, 9), "hcdr2": (5, 6), "hcdr3": (9, 15), "lcdr1": (11, 16), "lcdr2": (7, 7), "lcdr3": (9, 9)},
        is_antibody=True,
    ),
    "atezolizumab_framework_vhvl": PromptFactory(
        name="atezolizumab_framework_vhvl",
        template="EVQLVESGGGLVQPGGSLRLSCAAS{hcdr1}WIHWVRQAPGKGLEWVAWI{hcdr2}TYYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCAR{hcdr3}WGQGTLVTVSSGGGSGGGSGGGSGGGSDIQMTQSPSSLSASVGDRVTITC{lcdr1}WYQQKPGKAPKLLIY{lcdr2}GVPSRFSGSGSGTDFTLTISSLQPEDFATYYC{lcdr3}FGQGTKVEIK",
        length_ranges = {"hcdr1": (7, 9), "hcdr2": (5, 6), "hcdr3": (9, 15), "lcdr1": (11, 16), "lcdr2": (7, 7), "lcdr3": (9, 9)},
        is_antibody=True,
    ),
    "ocankitug_framework_vhvl": PromptFactory(
        name="ocankitug_framework_vhvl",
        template="QVQLVQSGAEVKKPGSSVKVSCKAS{hcdr1}WMHWVRQAPGQGLEWMGII{hcdr2}TSLNQKFQGRVTITADTSTSTAYMELSSLRSEDTAVYYCAR{hcdr3}WGQGTLVTVSSGGGSGGGSGGGSGGGSDIQMTQSPSSLSASVGDRVTITC{lcdr1}WYQQKPGKAPKLLIY{lcdr2}GVPSRFSGSGSGTDFTLTISSLQPEDFATYYC{lcdr3}FGQGTKVEIK",
        length_ranges = {"hcdr1": (7, 9), "hcdr2": (5, 6), "hcdr3": (8, 14), "lcdr1": (11, 16), "lcdr2": (7, 7), "lcdr3": (9, 9)},
        is_antibody=True,
    )
}


TARGET_SEQUENCES = {
    # https://www.uniprot.org/uniprotkb/P08575  389-574
    "cd45": "GSPGEPQIIFCRSEAAHQGVITWNPPQRSFHNFTLCYIKETEKDCLNLDKNLIKYDLQNLKPYTKYVLSLHAYIIAKVQRNGSAAMCHFTTKSAPPSQVWNMTVSMTSDNSMHVKCRPPRDRNGPHERYHLEVEAGNTLVRNESHKNCDFRVKDLQYSTDYTFKAYFHNGDYPGEPFILHHSTSY",
    # https://www.uniprot.org/uniprotkb/P16410  37-155
    "ctla4": "MHVAQPAVVLASSRGIASFVCEYASPGKATEVRVTVLRQADSQVTEVCAATYMMGNELTFLDDSICTGTSSGNQVNLTIQGLRAMDTGLYICKVELMYPPPYYLGIGNGTQIYVIDPE",
    # https://www.uniprot.org/uniprotkb/P00533  333-524
    "egfr": "RKVCNGIGIGEFKDSLSINATNIKHFKNCTSISGDLHILPVAFRGDSFTHTPPLDPQELDILKTVKEITGFLLIQAWPENRTDLHAFENLEIIRGRTKQHGQFSLAVVSLNITSLGLRSLKEISDGDVIISGNKNLCYANTINWKKLFGTSGQKTKIISNRGENSCKATGQVCHALCSPEGCWGPEPRDCV",
    # https://www.uniprot.org/uniprotkb/Q9NZQ7  17-132
    "pd-l1": "AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNA",
    # https://www.uniprot.org/uniprotkb/P09619  125-312
    "pdgfr": "GFLPNDAEELFIFLTEITEITIPCRVTDPQLVVTLHEKKGDVALPVPYDHQRGFSGIFEDRSYICKTTIGDREVDSDAYYVYRLQVSSINVSVNAVQTVVRQGENITLMCIVIGNEVVNFEWTYPRKESGRLVEPVTDFLLDMPYHIRSILHIPSAELEDSGTYTCNVTESVNDHQDEKAINITVVE",
}
# fmt: on


# ---- Helper functions ----


def build_initial_soft_sequence_logits(sequence: str, batch_size: int) -> torch.Tensor:
    """
    Initialize logits with:
    - High confidence (10.0) for fixed positions
    - Random (~0) for mutable positions
    - -1e6 for cysteines
    """
    if all(aa == MUTABLE_TOKEN for aa in sequence):
        logits = 0.01 * torch.randn([batch_size, len(sequence), AA_DIMS])
        logits[:, :, CYS_IDX] = -1e6  # remove cysteines
    else:
        logits = torch.zeros([batch_size, len(sequence), AA_DIMS])
        for i, aa in enumerate(sequence):
            if aa == MUTABLE_TOKEN:  # mutable position - random
                logits[:, i, :] = 0.01 * torch.randn(batch_size, AA_DIMS)
                logits[:, i, CYS_IDX] = -1e6
            else:  # fixed position
                assert aa in PROTEIN_1TO3, aa
                token_id = TOKEN_IDS[PROTEIN_1TO3[aa]]
                logits[:, i, token_id - 2] = 10.0

    return logits.requires_grad_(True)


def build_gradient_mask(sequence: str, batch_size: int) -> torch.Tensor:
    """
    Build gradient mask [B, L, V]:
    - 0 for fixed (all amino acids)
    - 0 for cysteine at all positions
    - 1 for non-cysteine amino acids at mutable positions
    """
    mask = torch.ones([batch_size, len(sequence), AA_DIMS])
    fixed_positions = [i for i, aa in enumerate(sequence) if aa != MUTABLE_TOKEN]
    mask[:, fixed_positions, :] = 0.0
    mask[:, :, CYS_IDX] = 0.0
    return mask


def sequence_to_one_hot(sequence: str, device="cuda") -> torch.Tensor:
    """Convert target string to one-hot tensor [1, L_target, num_tokens]."""

    const_dict = {token: i for i, token in enumerate(TOKENS)}
    target_index = [const_dict[PROTEIN_1TO3[letter]] for letter in sequence]
    one_hot = F.one_hot(torch.tensor(target_index), num_classes=len(TOKENS))
    return one_hot.to(device).unsqueeze(0).float()


##### claude 2026/8/13 start #####
def build_target_one_hot(target_sequence: str, device="cuda") -> torch.Tensor:
    """Build per-token target one-hot for a possibly multi-chain target.

    Each ``|``-separated chain is encoded independently by
    :func:`sequence_to_one_hot` and concatenated along the token axis,
    mirroring the token order produced by the input pipeline
    (``prepare_esmfold2_tensors`` tokenizes chains in ``split("|")`` order).

    Returns
    -------
    torch.Tensor
        Shape ``[1, sum(chain lengths), 33]`` float32 one-hot.
    """
    chains = target_sequence.split("|")
    assert all(len(c) > 0 for c in chains), (
        f"Empty chain in target_sequence {target_sequence!r}."
    )
    return torch.cat(
        [sequence_to_one_hot(chain, device=device) for chain in chains], dim=1
    )
##### claude 2026/8/13 end #####


def get_mid_points() -> torch.Tensor:
    """128 distance bin midpoints (2p-52 Angstrom range)."""

    boundaries = torch.linspace(2, 52.0, 127)
    lower = torch.tensor([1.0])
    upper = torch.tensor([52.0 + 5.0])
    exp_boundaries = torch.cat((lower, boundaries, upper))
    return (exp_boundaries[:-1] + exp_boundaries[1:]) / 2


def binned_entropy(
    dgram: torch.Tensor, bin_distance: torch.Tensor, cutoff: float
) -> torch.Tensor:
    """Entropy of distance distribution within cutoff (design losses only)."""

    bin_mask = ~(bin_distance < cutoff)
    masked_dgram = dgram - (1e7 * bin_mask)
    px = torch.softmax(masked_dgram, dim=-1)
    log_px = torch.log_softmax(dgram, dim=-1)
    return -(px * log_px).sum(-1)


def masked_min_k(x: torch.Tensor, mask: torch.Tensor, k: int) -> torch.Tensor:
    """Mean of the smallest k values in x under mask along the last dimension."""

    mask = mask.bool()
    y = torch.sort(torch.where(mask, x, float("nan")))[0]
    k_mask = (torch.arange(y.shape[-1]).to(y.device) < k) & (~torch.isnan(y))
    return torch.where(k_mask, y, 0).sum(-1) / (k_mask.sum(-1) + 1e-8)


def masked_average(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked mean along last axis."""

    mask = mask.bool()
    return torch.where(mask, x, 0).sum(-1) / (torch.where(mask, 1, 0).sum(-1) + 1e-8)


# ---- Loss functions ----


def compute_contact_loss(
    distogram_logits: torch.Tensor,
    bin_distance: torch.Tensor,
    num_contacts: int,
    min_sep: int,
    cutoff: float,
    chain_mask: torch.Tensor,
    binder_mask: torch.Tensor,
) -> torch.Tensor:
    """Algorithm 12 Contact Losses.

    Entropy-based contact loss with sequence separation constraint."""

    con_loss = binned_entropy(distogram_logits, bin_distance, cutoff)
    position = torch.arange(distogram_logits.shape[1])
    p_dist = position[:, None] - position[None, :]
    if min_sep > 0:
        separation_mask = (torch.abs(p_dist) >= min_sep).to(distogram_logits.device)
        binder_mask = torch.logical_and(separation_mask, binder_mask)
    per_residue = masked_min_k(con_loss, mask=binder_mask, k=num_contacts).to(
        distogram_logits.device
    )
    return masked_average(per_residue, mask=chain_mask).to(distogram_logits.device)


def compute_intra_contact_loss(
    distogram_logits: torch.Tensor, binder_length: int, bin_distance: torch.Tensor
) -> torch.Tensor:
    """Binder internal contacts (k=2, min_sep=9, cutoff=14A)."""

    full_len = distogram_logits.shape[1]
    is_binder = torch.ones(full_len, device=distogram_logits.device)
    is_binder[:-binder_length] *= 0.0
    return compute_contact_loss(
        distogram_logits,
        bin_distance,
        num_contacts=2,
        min_sep=9,
        cutoff=14.0,
        chain_mask=is_binder,
        binder_mask=is_binder,
    )


def compute_inter_contact_loss(
    distogram_logits: torch.Tensor, binder_length: int, bin_distance: torch.Tensor
) -> torch.Tensor:
    """Binder-target interface (k=1, min_sep=0, cutoff=22A)."""

    full_len = distogram_logits.shape[1]
    is_binder = torch.ones(full_len, device=distogram_logits.device)
    is_binder[:-binder_length] *= 0.0
    return compute_contact_loss(
        distogram_logits,
        bin_distance,
        num_contacts=1,
        min_sep=0,
        cutoff=22.0,
        chain_mask=1 - is_binder,
        binder_mask=is_binder,
    )


def compute_inter_contact_loss_neighborhood(
    distogram_logits: torch.Tensor,
    binder_length: int,
    bin_distance: torch.Tensor,
    neighborhood_mask: torch.Tensor,
) -> torch.Tensor:
    """Binder-target interface restricted to hotspot-neighborhood residues.

    Parallel mode to :func:`compute_inter_contact_loss`: identical engine and
    cutoffs (k=1, min_sep=0, cutoff=22A), but the averaging domain is the
    precomputed surface-accessible hotspot neighborhood instead of every
    target residue, so the loss no longer demands binder contacts over the
    whole target surface.

    ``neighborhood_mask`` is a boolean ``(1, T)`` tensor over the target
    residues (computed once from a pre-optimization fold by
    :func:`derive_hotspot_neighborhood_mask`); it is shared across the batch,
    matching the 1-D masks used by the other contact losses.
    """

    full_len = distogram_logits.shape[1]
    is_binder = torch.ones(full_len, device=distogram_logits.device)
    is_binder[:-binder_length] *= 0.0
    chain_mask = torch.zeros(full_len, device=distogram_logits.device)
    chain_mask[:-binder_length] = neighborhood_mask[0].to(torch.float32)
    return compute_contact_loss(
        distogram_logits,
        bin_distance,
        num_contacts=1,
        min_sep=0,
        cutoff=22.0,
        chain_mask=chain_mask,
        binder_mask=is_binder,
    )


def compute_globularity_loss(
    distogram_logits: torch.Tensor, binder_length: int, bin_distance: torch.Tensor
) -> torch.Tensor:
    """Algorithm 13 Globularity Loss.

    Radius of gyration vs theoretical packed protein."""

    binder_disto = distogram_logits[:, -binder_length:, -binder_length:, :]
    n = binder_disto.shape[1]
    disto_probs = torch.softmax(binder_disto, dim=-1)
    bin_distance = bin_distance.clamp(max=27)
    e_sq_dist = torch.sum(disto_probs * torch.square(bin_distance), dim=-1)
    sum_sq_dist = torch.sum(torch.tril(e_sq_dist, diagonal=-1), dim=(1, 2))
    rg_term = torch.sqrt(sum_sq_dist / (n * n))
    rg_th = 2.38 * (n**0.365)
    return F.elu(rg_term - rg_th)


def get_epitope_mask(
    target_chains: list[str],
    binder_length: int,
    target_hotspot_ids: list[str],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    ##### claude 2026/8/13 start #####
    """Full-complex mask for chain-qualified target hotspot residue IDs.

    Hotspot IDs must have the form ``<chain>:<residue><idx>`` (e.g. ``A:L150``):
    the chain label is a single uppercase letter A-Z naming one of the target
    chains, and the residue index is 1-indexed within that chain. Chain labels
    follow the complex's asym_id assignment (target chains first, binder last).
    """
    chain_labels = [_asym_id_to_chain_label(i) for i in range(len(target_chains))]
    chain_starts: list[int] = []
    pos = 0
    for chain in target_chains:
        chain_starts.append(pos)
        pos += len(chain)
    total_target = pos

    epitope_mask = torch.zeros(
        total_target + binder_length, dtype=torch.bool, device=device
    )
    for hotspot_id in target_hotspot_ids:
        assert ":" in hotspot_id, (
            f"Hotspot ID {hotspot_id!r} must be chain-qualified as "
            "<chain>:<residue><idx>, for example A:L150."
        )
        chain_label, residue_id = hotspot_id.split(":", 1)
        assert len(chain_label) == 1 and "A" <= chain_label <= "Z", (
            f"Hotspot chain label {chain_label!r} must be a single "
            "uppercase letter A-Z."
        )
        assert chain_label in chain_labels, (
            f"Hotspot chain {chain_label!r} not found among target chains "
            f"{chain_labels}."
        )
        chain_idx = chain_labels.index(chain_label)
        chain_seq = target_chains[chain_idx]
        offset = chain_starts[chain_idx]

        assert len(residue_id) >= 2, (
            f"Hotspot residue ID {residue_id!r} must have form <residue><idx>, "
            "for example L150."
        )
        residue = residue_id[0]
        hotspot_idx_text = residue_id[1:]
        assert residue in PROTEIN_1TO3, (
            f"Hotspot residue ID {residue_id!r} starts with unknown residue "
            f"{residue!r}."
        )
        assert hotspot_idx_text.isdecimal(), (
            f"Hotspot residue ID {residue_id!r} must have form <residue><idx>, "
            "for example L150."
        )
        hotspot_idx = int(hotspot_idx_text)
        assert 1 <= hotspot_idx <= len(chain_seq), (
            f"Hotspot residue {residue_id!r} is out of range for chain "
            f"{chain_label} of length {len(chain_seq)}. Hotspots are "
            "1-indexed within the chain."
        )
        chain_residue = chain_seq[hotspot_idx - 1]
        assert chain_residue == residue, (
            f"Hotspot residue ID {hotspot_id!r} does not match chain "
            f"{chain_label} sequence: residue at 1-indexed position "
            f"{hotspot_idx} is {chain_residue!r}."
        )
        epitope_mask[offset + hotspot_idx - 1] = True
    return epitope_mask[None].repeat(batch_size, 1)
    ##### claude 2026/8/13 end #####


def compute_epitope_loss(
    distogram_logits: torch.Tensor,
    epitope_mask: torch.Tensor,
    target_length: int,
    binder_length: int,
    bin_distance: torch.Tensor,
    epitope_contact_distance: float = 12.0,
) -> torch.Tensor:
    """Encourage binder contacts to the requested target hotspot residues.

    Converts full-complex distogram logits into contact probabilities below
    ``epitope_contact_distance``, then penalizes each hotspot by the best few
    binder contacts. The target is assumed to occupy the first ``target_length``
    positions of the complex and the binder the final ``binder_length`` positions.

    Parameters
    ----------
    distogram_logits
        Full-complex distogram logits with shape ``(B, L, L, num_bins)``.
    epitope_mask
        Boolean full-complex mask with shape ``(B, L)`` and true values at
        target hotspot positions.
    target_length
        Number of target residues at the start of the complex sequence.
    binder_length
        Number of binder residues at the end of the complex sequence.
    bin_distance
        Distance represented by each distogram bin.
    epitope_contact_distance
        Distance threshold used to define a hotspot-binder contact.

    Returns
    -------
    torch.Tensor
        Per-example epitope losses with shape ``(B,)``.
    """

    B, L, L2, _ = distogram_logits.shape
    assert L == L2, (L, L2)
    assert L == target_length + binder_length, (L, target_length, binder_length)

    binder_mask = torch.zeros((B, L), dtype=torch.bool, device=distogram_logits.device)
    binder_mask[:, target_length:] = True

    positive_mask = bin_distance < epitope_contact_distance
    assert binder_mask.shape == (B, L), f"Binder mask: {binder_mask.shape} != {B, L}"
    assert epitope_mask.shape == (B, L), f"Epitope mask: {epitope_mask.shape} != {B, L}"
    # NOTE - this could be made bidirectional.
    epitope_binder_mask = epitope_mask[:, :, None] & binder_mask[:, None, :]

    probabilities = torch.softmax(distogram_logits, dim=-1)
    positive_probs = torch.sum(probabilities * positive_mask, dim=-1)
    contact_measure = -torch.log(positive_probs + 1e-8)

    epitope_loss = masked_min_k(contact_measure, mask=epitope_binder_mask, k=5)
    return masked_average(epitope_loss, mask=epitope_mask)


def derive_hotspot_neighborhood_mask(
    target_block_logits: torch.Tensor,
    bin_distance: torch.Tensor,
    hotspot_positions: torch.Tensor,
    neighborhood_radius: float = INTER_NEIGHBORHOOD_RADIUS,
    burial_contact_distance: float = INTER_BURIAL_CONTACT_DISTANCE,
    burial_quantile: float = INTER_BURIAL_QUANTILE,
    burial_min_sep: int = INTER_BURIAL_MIN_SEP,
    min_size: int = INTER_NEIGHBORHOOD_MIN_SIZE,
) -> torch.Tensor | None:
    """Surface-accessible target residues near the hotspots, from a distogram.

    Parallel-mode helper for the inter contact loss: instead of demanding
    binder contacts over every target residue ("full coverage"), restrict the
    inter-loss averaging domain to target residues that are both

    - near a hotspot: expected distogram distance to any hotspot below
      ``neighborhood_radius``;
    - surface accessible: contact number of the target self-block — a classic
      burial proxy computable from pairwise distance predictions alone, no
      atomic coordinates or solvent model required — below the
      ``burial_quantile`` quantile of the target.

    The target sequence is fixed during gradient optimization, so this is
    computed ONCE from a single pre-optimization fold and used as a constant
    mask thereafter (same non-gradient status as ``binder_mask`` and
    ``epitope_mask``).

    Parameters
    ----------
    target_block_logits
        Target self-block of the full-complex distogram, ``(B, T, T, num_bins)``.
        For a multi-chain target this spans all target chains in
        ``split("|")`` order, so interface-buried residues (including
        target-target chain interfaces) are correctly counted as buried.
    bin_distance
        Distance represented by each distogram bin, ``(num_bins,)``.
    hotspot_positions
        0-indexed absolute positions of the hotspots within the target block,
        1-D long tensor.
    min_size
        Returns ``None`` when fewer residues survive (caller falls back to
        the full-coverage inter loss with a warning).

    Returns
    -------
    torch.Tensor | None
        Boolean mask ``(B, T)`` with True at usable inter-loss residues, or
        ``None`` when the neighborhood is degenerately small.
    """

    T = target_block_logits.shape[1]
    # Design-loop distogram logits are bf16 (autocast); the mask math runs
    # once on CPU/GPU-safe float32 — quantile() rejects bf16 and the contact
    # sums keep full precision.
    probs = torch.softmax(target_block_logits.to(torch.float32), dim=-1)
    # (B, T): expected distance from every residue to its nearest hotspot.
    exp_dist = (probs * bin_distance).sum(-1)
    near = (exp_dist[:, hotspot_positions, :] < neighborhood_radius).any(dim=1)
    # (B, T): contact number of the target self-block (burial proxy).
    # Sequence-adjacent pairs are excluded: they contact trivially (constant
    # offset for every residue) and, for multi-chain targets, a chain
    # junction is sequence-adjacent while spatially unrelated.
    position = torch.arange(T, device=target_block_logits.device)
    sep_mask = (position[:, None] - position[None, :]).abs() >= burial_min_sep
    contact_prob = (probs * (bin_distance < burial_contact_distance)).sum(-1)
    burial = (contact_prob * sep_mask).sum(-1)
    threshold = torch.quantile(burial, burial_quantile, dim=-1, keepdim=True)
    accessible = burial < threshold
    mask = near & accessible
    mask[:, hotspot_positions] = True  # hotspots anchor the neighborhood
    if int(mask.sum()) < min_size:
        return None
    return mask


def precompute_inter_neighborhood_mask(
    inversion_model: ESMFold2ExperimentalModel,
    target_sequence: str,
    target_one_hot: torch.Tensor,
    binder_length: int,
    target_hotspot_ids: list[str],
    batch_size: int = 1,
    device: str = "cuda",
    seed: int = 0,
    neighborhood_radius: float = INTER_NEIGHBORHOOD_RADIUS,
    burial_contact_distance: float = INTER_BURIAL_CONTACT_DISTANCE,
    burial_quantile: float = INTER_BURIAL_QUANTILE,
    burial_min_sep: int = INTER_BURIAL_MIN_SEP,
    min_size: int = INTER_NEIGHBORHOOD_MIN_SIZE,
) -> torch.Tensor | None:
    """One-shot fold of the fixed target; returns the neighborhood mask.

    Parallel-mode companion to ``inter_loss_mode="hotspot_neighborhood"``:
    wraps :func:`derive_hotspot_neighborhood_mask` (the mask semantics) with
    the model I/O — a single no-grad TARGET-ONLY fold and chain-qualified
    hotspot resolution. 2026-10-14: the placeholder-binder fold was replaced
    by a target-only fold — the mask must be a pure function of the target;
    a placeholder binder perturbed the target self-block and made the mask
    vary with the unknown binder's length/composition (observed 23-28
    residues for the same target across binder lengths). The target
    sequence does not change during optimization, so the mask is a constant
    thereafter.

    ``binder_length`` is only used for hotspot-position bookkeeping (the
    mask layout mirrors the complex) — the fold itself is binder-free.

    Returns a boolean ``(1, T)`` mask over the target residues, or ``None``
    when the neighborhood is degenerately small (caller falls back to the
    full-coverage inter loss).
    """

    with torch.no_grad():
        pre_fold = fold_and_get_distogram(
            inversion_model,
            target_sequence,
            target_one_hot,
            None,
            num_loops=1,
            num_sampling_steps=1,
            calculate_confidence=False,
            seed=seed,
            include_binder=False,
        )
    total_target = pre_fold["distogram_logits"].shape[1]
    assert sum(len(c) for c in target_sequence.split("|")) == total_target
    hotspot_positions = get_epitope_mask(
        target_chains=target_sequence.split("|"),
        binder_length=binder_length,
        target_hotspot_ids=target_hotspot_ids,
        batch_size=1,
        device=device,
    )[0].nonzero(as_tuple=True)[0]
    target_block = pre_fold["distogram_logits"][:, :total_target, :total_target]
    return derive_hotspot_neighborhood_mask(
        target_block,
        get_mid_points().to(target_block.device),
        hotspot_positions,
        neighborhood_radius=neighborhood_radius,
        burial_contact_distance=burial_contact_distance,
        burial_quantile=burial_quantile,
        burial_min_sep=burial_min_sep,
        min_size=min_size,
    )


def update_fold_rescue_schedule(
    state: dict, weights: dict, step: int, intra_mean: float
) -> None:
    """Advance the fold-rescue phase schedule by one optimization step.

    ``state`` is a per-run dict (``call``/``phase``/``intra``/``events``),
    updated in place together with ``weights`` — the per-run loss-weight
    overlay consumed by :func:`compute_structure_losses`. Phase semantics:
    1 = native weights, 2 = fold rescue (FOLD_RESCUE_P2_WEIGHTS), 3 =
    interface refinement (FOLD_RESCUE_P3_WEIGHTS).

    Call AFTER computing the step's losses, so a phase transition takes
    effect on the following step (same ordering as the calibrated test
    scripts). No-op when the schedule already reached phase 3.
    """
    state["call"] += 1
    state["intra"].append(intra_mean)
    step_idx = state["call"] - 1
    if state["phase"] == 1 and step_idx in FOLD_RESCUE_CHECKPOINT_STEPS:
        last5 = state["intra"][-5:]
        mean5 = sum(last5) / len(last5)
        state["events"].append({"step": step_idx, "event": "p1_checkpoint",
                                "intra_last5_mean": round(mean5, 3)})
        if mean5 > FOLD_RESCUE_INTRA_TRIGGER:
            state["phase"] = 2
            weights.update(FOLD_RESCUE_P2_WEIGHTS)
            state["events"].append({"step": step_idx, "event": "phase2",
                                    "weights": dict(weights)})
            logger.info(
                "fold_rescue: step %d plateau (intra_last5=%.2f>%.1f) -> "
                "phase 2 weights %s.",
                step_idx, mean5, FOLD_RESCUE_INTRA_TRIGGER, FOLD_RESCUE_P2_WEIGHTS,
            )
        elif step_idx == FOLD_RESCUE_CHECKPOINT_STEPS[-1]:
            state["events"].append({"step": step_idx, "event": "stay_phase1"})
            logger.info(
                "fold_rescue: step %d folding progressing, staying in "
                "phase 1 for the remaining steps.", step_idx,
            )
    elif state["phase"] == 2 and step_idx >= FOLD_RESCUE_P3_MIN_STEP \
            and step_idx % 5 == 0:
        last_n = state["intra"][-FOLD_RESCUE_P3_WINDOW:]
        mean_n = sum(last_n) / len(last_n)
        state["events"].append({"step": step_idx, "event": "p2_checkpoint",
                                "intra_last10_mean": round(mean_n, 3)})
        if mean_n < FOLD_RESCUE_P3_INTRA:
            state["phase"] = 3
            weights.update(FOLD_RESCUE_P3_WEIGHTS)
            state["events"].append({"step": step_idx, "event": "phase3",
                                    "weights": dict(weights)})
            logger.info(
                "fold_rescue: step %d folding complete (intra_last%d=%.2f<%.1f)"
                " -> phase 3 weights %s.",
                step_idx, FOLD_RESCUE_P3_WINDOW, mean_n, FOLD_RESCUE_P3_INTRA,
                FOLD_RESCUE_P3_WEIGHTS,
            )


def compute_structure_losses(
    distogram_logits: torch.Tensor,
    binder_length: int,
    target_sequence: str | None = None,
    target_hotspot_ids: list[str] | None = None,
    epitope_contact_distance: float = 12.0,
    inter_neighborhood_mask: torch.Tensor | None = None,
    loss_weights: dict | None = None,
) -> dict[str, torch.Tensor]:
    """Compute structural losses and a weighted total.

    ``inter_neighborhood_mask`` switches the inter contact loss to the
    hotspot-neighborhood parallel mode (see
    :func:`compute_inter_contact_loss_neighborhood`); ``None`` keeps the
    default full-coverage inter loss.

    ``loss_weights`` is an optional per-run overlay for the loss weights
    (fold-rescue phase schedule); ``None`` uses the module-level
    ``LOSS_WEIGHTS``.
    """

    weights = loss_weights if loss_weights is not None else LOSS_WEIGHTS
    bin_distance = get_mid_points().to(distogram_logits.device)
    losses: dict[str, torch.Tensor] = {}
    losses["intra_contact_loss"] = compute_intra_contact_loss(
        distogram_logits, binder_length, bin_distance
    )
    if inter_neighborhood_mask is not None:
        losses["inter_contact_loss"] = compute_inter_contact_loss_neighborhood(
            distogram_logits, binder_length, bin_distance, inter_neighborhood_mask
        )
    else:
        losses["inter_contact_loss"] = compute_inter_contact_loss(
            distogram_logits, binder_length, bin_distance
        )
    losses["glob_loss"] = compute_globularity_loss(
        distogram_logits, binder_length, bin_distance
    )
    B = distogram_logits.size(0)
    total = torch.tensor([0.0] * B, device=distogram_logits.device, requires_grad=True)
    total = total + weights["intra_contact"] * losses["intra_contact_loss"]
    total = total + weights["inter_contact"] * losses["inter_contact_loss"]
    total = total + weights["glob"] * losses["glob_loss"]
    if target_hotspot_ids is not None:
        assert (
            target_sequence is not None
        ), "target_sequence is required when target_hotspot_ids is provided."
        ##### claude 2026/8/13 start #####
        # Multi-chain targets: hotspots are chain-qualified ("A:L150") and
        # resolved against the per-chain sequences.
        target_chains = target_sequence.split("|")
        total_target = distogram_logits.shape[1] - binder_length
        assert sum(len(c) for c in target_chains) == total_target, (
            f"Sum of target chain lengths {sum(len(c) for c in target_chains)} "
            f"does not match distogram target length {total_target}."
        )
        epitope_mask = get_epitope_mask(
            target_chains=target_chains,
            binder_length=binder_length,
            target_hotspot_ids=target_hotspot_ids,
            batch_size=B,
            device=distogram_logits.device,
        )
        ##### claude 2026/8/13 end #####
        losses["epitope_loss"] = compute_epitope_loss(
            distogram_logits=distogram_logits,
            epitope_mask=epitope_mask,
            ##### claude 2026/9/9 fix #####
            # 重构残留: 本函数作用域内从未定义 target_length (NameError,
            # 仅热点分支触发)。total_target (上方 assert 已验证其等于
            # 各 target 链长度之和) 即 compute_epitope_loss 所需的
            # target token 跨度。
            target_length=total_target,
            ##### claude 2026/9/9 fix end #####
            binder_length=binder_length,
            bin_distance=bin_distance,
            epitope_contact_distance=epitope_contact_distance,
        )
        total = total + weights["epitope"] * losses["epitope_loss"]
    losses["total_loss"] = total
    return losses


# ---- Distogram iptm proxy ----


def _binding_confidence_entropy(
    dgram: torch.Tensor, bin_distance: torch.Tensor, cutoff: float
) -> torch.Tensor:
    """Pair entropy within cutoff."""

    probs = torch.softmax(dgram, dim=-1)
    cutoff_mask = bin_distance < cutoff
    p_cut = probs[..., cutoff_mask]
    p_cut = p_cut / (p_cut.sum(-1, keepdim=True) + 1e-8)
    return -(p_cut * torch.log(p_cut + 1e-10)).sum(-1)


def _entropy_to_confidence(mean_entropy: float) -> float:
    """Map mean pair entropy to [0, 1]; lower entropy → higher score."""
    return float(max(0.0, min(1.0, 1.0 - mean_entropy / math.log(51))))


def _is_antibody_sequence(binder_sequence: str) -> bool:
    """Return True if ANARCI recognizes binder_sequence as antibody variable domain(s)."""
    from abnumber.common import _anarci_align

    sequence = binder_sequence.replace(MUTABLE_TOKEN, "A")
    result = _anarci_align(
        sequences=[sequence], scheme="chothia", allowed_species=None
    )[0]
    if not result:
        return False
    valid_chain_types = {"H", "K", "L"}
    return all(chain_type in valid_chain_types for _, chain_type, *_ in result)


def _cdr_indices(binder_sequence: str) -> list[int]:
    """0-based binder indices for all Chothia CDRs."""
    from abnumber import Chain
    from abnumber.common import _anarci_align

    result = _anarci_align(
        sequences=[binder_sequence], scheme="chothia", allowed_species=None
    )[0]
    chains = [
        Chain("".join(result[i][0].values()), scheme="chothia")
        for i in range(len(result))
    ]
    if len(chains) == 2 and not chains[0].is_heavy_chain():
        chains.reverse()
    indices: list[int] = []
    for chain in chains:
        for cdr in (chain.cdr1_seq, chain.cdr2_seq, chain.cdr3_seq):
            start = binder_sequence.find(cdr)
            assert start >= 0
            indices.extend(range(start, start + len(cdr)))
    return indices


def compute_distogram_iptm_proxy(
    distogram_logits: torch.Tensor,
    target_length: int,
    binder_sequence: str,
    is_antibody: bool,
) -> dict[str, float]:
    """Algorithm 15 Distogram ipTM Proxy.

    Distogram iptm proxy for a target|binder complex (binder at suffix).

    Returns distogram_iptm_proxy for all designs and
    cdr_distogram_iptm_proxy when the binder can be annotated as an
    antibody; otherwise the CDR score is NaN.
    """
    if distogram_logits.ndim == 4:
        distogram_logits = distogram_logits[0]

    binder_length = len(binder_sequence)
    assert distogram_logits.shape[0] == target_length + binder_length

    bin_distance = get_mid_points().to(distogram_logits.device)
    binder_start = target_length

    def _mean_lowest_k(entropies: torch.Tensor, k: int) -> float:
        sorted_entropies, _ = torch.sort(entropies.reshape(-1))
        k = min(k, sorted_entropies.numel())
        return float(sorted_entropies[:k].mean())

    binder_to_target_entropy = _binding_confidence_entropy(
        distogram_logits[binder_start:, :target_length, :], bin_distance, cutoff=22.0
    )
    distogram_iptm_proxy = _entropy_to_confidence(
        _mean_lowest_k(binder_to_target_entropy, k=binder_length)
    )

    if not is_antibody:
        cdr_distogram_iptm_proxy = float("nan")
    else:
        cdr_indices = _cdr_indices(binder_sequence)
        cdr_rows = [binder_start + i for i in cdr_indices]
        cdr_to_target_entropy = _binding_confidence_entropy(
            distogram_logits[cdr_rows, :target_length, :], bin_distance, cutoff=22.0
        )
        cdr_distogram_iptm_proxy = _entropy_to_confidence(
            _mean_lowest_k(cdr_to_target_entropy, k=len(cdr_indices))
        )

    return {
        "distogram_iptm_proxy": distogram_iptm_proxy,
        "cdr_distogram_iptm_proxy": cdr_distogram_iptm_proxy,
    }


# ---- Folding ----


def _resize_tensor(tensor: torch.Tensor, *, dim: int, size: int) -> torch.Tensor:
    current = tensor.shape[dim]
    if current >= size:
        return tensor.narrow(dim, 0, size)

    pad_shape = list(tensor.shape)
    pad_shape[dim] = size - current
    pad = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
    return torch.cat((tensor, pad), dim=dim)


_ATOM_FEATURE_DIMS = {
    "ref_pos": 0,
    "ref_element": 0,
    "ref_charge": 0,
    "ref_atom_name_chars": 0,
    "ref_space_uid": 0,
    "atom_attention_mask": 0,
    "atom_to_token": 0,
    "is_resolved": 0,
    "gt_coords": 1,
}


@cache
def _ensure_ccd_loaded() -> None:
    load_ccd()


def prepare_esmfold2_tensors(
    input: StructurePredictionInput,
    max_tokens: int | None = None,
    max_atoms: int | None = None,
    max_seqs: int = 16384,
    pad_to_max_seqs: bool = False,
    seed: int | None = None,
    use_vectorized_msa_assembly: bool = True,
) -> dict[str, torch.Tensor]:
    del max_tokens, max_seqs, pad_to_max_seqs, use_vectorized_msa_assembly
    _ensure_ccd_loaded()
    features, _ = prepare_esmfold2_input(input, seed=seed)
    if max_atoms is not None:
        for key, dim in _ATOM_FEATURE_DIMS.items():
            if key in features:
                features[key] = _resize_tensor(features[key], dim=dim, size=max_atoms)
    return features


def fold_and_get_distogram(
    model: ESMFold2ExperimentalModel,
    target_seq: str,
    target_one_hot: torch.Tensor,
    design: torch.Tensor | None,
    num_loops: int = 0,
    num_sampling_steps: int = 1,
    calculate_confidence: bool = False,
    seed: int | None = None,
    include_binder: bool = True,
) -> dict:
    """Prepare inputs, run model forward, return distogram_logits + raw output.

    ``include_binder=False`` folds the target chains only (``design`` must be
    ``None``): used by the neighborhood-mask precompute, where the mask must
    be a pure function of the target — a placeholder binder would perturb the
    target self-block and make the mask vary with the unknown binder's
    length/composition.
    """
    if include_binder:
        assert design is not None, "design is required when include_binder=True"
        padding = (2, 11)
        padded_design = F.pad(design, padding, mode="constant", value=0)

        # Argmax to get the designed sequence string.
        token_lists = torch.argmax(padded_design, dim=-1)
        designed_seq = [
            [PROTEIN_3TO1[TOKENS[int(tkn.item())]] for tkn in token_list]
            for token_list in token_lists
        ]
        seq_list = [target_seq + "|" + "".join(seq) for seq in designed_seq]
    else:
        seq_list = [target_seq]
    max_atoms = None if len(seq_list) == 1 else ((len(seq_list[0]) - 1) * 14) // 32 * 32

    inputs_list = []
    for seq in seq_list:
        ##### claude 2026/8/13 start #####
        # One ProteinInput per chain, in split("|") order. Do NOT deduplicate
        # by sequence string: identical chains (homomeric targets) must each
        # get their own ProteinInput and asym_id — a dict comprehension here
        # silently drops all but the last occurrence of a duplicated chain.
        inputs_raw = StructurePredictionInput(
            sequences=[
                ProteinInput(id=str(idx), sequence=chain, msa=None)
                for idx, chain in enumerate(seq.split("|"))
            ]
        )
        ##### claude 2026/8/13 end #####
        inputs_list.append(prepare_esmfold2_tensors(inputs_raw, max_atoms=max_atoms))

    inputs = {
        key: torch.stack([inp[key] for inp in inputs_list], dim=0).cuda()
        for key in inputs_list[0]
    }
    if include_binder:
        inputs["res_type_soft"] = torch.cat(
            (target_one_hot.repeat(design.size(0), 1, 1), padded_design), dim=1
        )

    with seed_context(seed):
        output = model(
            **inputs,
            num_diffusion_samples=1,
            num_sampling_steps=num_sampling_steps,
            num_loops=num_loops,
            calculate_confidence=calculate_confidence,
            seed=seed,
        )

    result: dict = {
        "distogram_logits": output["distogram_logits"],
        "inputs": inputs,
        "inputs_list": inputs_list,
        "output": output,
        "seq_list": seq_list,
    }
    if calculate_confidence:
        result.update(
            {
                "ptm": output.get("ptm"),
                "iptm": output.get("iptm"),
                "plddt": output.get("plddt"),
            }
        )
    return result


_CHAIN_ID_ALPHABET = string.ascii_uppercase + string.ascii_lowercase + string.digits


def _asym_id_to_chain_label(asym_id: int) -> str:
    if asym_id < 0:
        raise ValueError(f"asym_id must be >= 0, got {asym_id}")
    label = ""
    n = len(_CHAIN_ID_ALPHABET)
    while True:
        label = _CHAIN_ID_ALPHABET[asym_id % n] + label
        asym_id = asym_id // n - 1
        if asym_id < 0:
            return label


def to_atom_array(
    coords: np.ndarray,
    atom_to_token: np.ndarray,
    res_type: np.ndarray,
    residue_index: np.ndarray,
    asym_id: np.ndarray,
    mol_type: np.ndarray,
    ref_atom_name_chars: np.ndarray,
    ref_element: np.ndarray,
    atom_attention_mask: np.ndarray,
    plddt_per_atom: np.ndarray | None = None,
) -> biotite.structure.AtomArray:
    atoms = []
    for atom_i, (
        atom_coord,
        token_idx,
        atom_name_chars,
        element_idx,
        is_not_pad,
    ) in enumerate(
        zip(
            coords, atom_to_token, ref_atom_name_chars, ref_element, atom_attention_mask
        )
    ):
        if not is_not_pad:
            continue
        atoms.append(
            biotite.structure.Atom(
                coord=atom_coord,
                chain_id=_asym_id_to_chain_label(int(asym_id[token_idx])),
                res_id=residue_index[token_idx] + 1,
                res_name=TOKENS[res_type[token_idx]],
                atom_name="".join(chr(c + 32) for c in atom_name_chars if c != 0),
                element=ELEMENTS[element_idx],
                ins_code=" ",
                hetero=mol_type[token_idx] == MOL_TYPE_NONPOLYMER,
                b_factor=float(plddt_per_atom[atom_i])
                if plddt_per_atom is not None
                else 0.0,
            )
        )
    return biotite.structure.array(atoms)


def build_complex(
    inputs: dict[str, torch.Tensor], output: dict[str, Any]
) -> ProteinComplex:
    """Build ProteinComplex from model output."""
    plddt_per_atom = output.get("plddt_per_atom")
    if plddt_per_atom is not None:
        plddt_per_atom = plddt_per_atom[0].cpu().numpy() * PLDDT_B_FACTOR_SCALE
    atom_arr = to_atom_array(
        coords=output["sample_atom_coords"][0].cpu().numpy(),
        atom_to_token=inputs["atom_to_token"][0].cpu().numpy(),
        res_type=inputs["res_type"][0].cpu().numpy(),
        residue_index=inputs["token_index"][0].cpu().numpy(),
        asym_id=inputs["asym_id"][0].cpu().numpy(),
        mol_type=inputs["mol_type"][0].cpu().numpy(),
        ref_atom_name_chars=inputs["ref_atom_name_chars"][0].cpu().numpy(),
        ref_element=inputs["ref_element"][0].cpu().numpy(),
        atom_attention_mask=inputs["atom_attention_mask"][0].cpu().numpy(),
        plddt_per_atom=plddt_per_atom,
    )
    return ProteinComplex.from_chains(
        [
            ProteinChain.from_atomarray(a, is_predicted=True)
            for a in biotite.structure.chain_iter(atom_arr)
        ]
    )


##### claude 2026/10/19 start #####
def merge_pair_chains_iptm(
    pair_chains_iptm: torch.Tensor,
    target_chain_lengths: list[int],
) -> dict:
    """Merge per-chain-pair ipTM into a binder-quality score (t2b direction).

    ``pair_chains_iptm`` is the ConfidenceHead output (fork
    ``modeling_esmfold2_experimental.py``), shape ``[B, n_chains, n_chains]``:
    entry ``[c1, c2]`` is the mean of ``tm_expected`` over the c1×c2 block —
    "placement confidence of c2 residues given c1 is the alignment reference"
    (PAE rows = reference, cols = error subject). The consumed direction is
    therefore **[target][binder] (t2b)**: fix the target, evaluate the binder
    placement. ``[binder][target]`` (b2t) is returned for diagnostics only.

    Multi-chain targets merge by residue count, which is algebraically the
    plain mean over the binder×(all targets) block — i.e. the global ipTM
    block mean restricted to binder-target pairs, removing the target-target
    pairs that inflate multi-chain iPTM. Weights are the fixed target chain
    lengths (non-circular: independent of what the binder actually engages).

    Parameters
    ----------
    pair_chains_iptm
        ``[n_chains, n_chains]`` or ``[B, n_chains, n_chains]`` tensor (the
        first batch element is used in the latter case).
    target_chain_lengths
        Residue count per target chain, in ``split("|")`` order.

    Returns
    -------
    dict
        ``{"merged_t2b": float, "merged_b2t": float,
        "chains": {chain_label: {"t2b": float, "b2t": float}}}``.
    """
    n_target = len(target_chain_lengths)
    binder_idx = n_target  # asym_id convention: targets 0..n-1, binder = n
    matrix = pair_chains_iptm[0] if pair_chains_iptm.ndim == 3 else pair_chains_iptm
    assert matrix.shape[0] > binder_idx, (
        f"pair_chains_iptm has {matrix.shape[0]} chains, expected more than "
        f"{binder_idx} (target chains + binder)."
    )
    chain_labels = [_asym_id_to_chain_label(i) for i in range(matrix.shape[0])]
    lengths = [float(n) for n in target_chain_lengths]
    total = float(sum(lengths))
    merged_t2b = sum(
        length * float(matrix[t, binder_idx]) for t, length in enumerate(lengths)
    ) / total
    merged_b2t = sum(
        length * float(matrix[binder_idx, t]) for t, length in enumerate(lengths)
    ) / total
    chains = {
        chain_labels[t]: {
            "t2b": float(matrix[t, binder_idx]),
            "b2t": float(matrix[binder_idx, t]),
        }
        for t in range(n_target)
    }
    return {
        "merged_t2b": merged_t2b,
        "merged_b2t": merged_b2t,
        "chains": chains,
    }
##### claude 2026/10/19 end #####


def pdb_with_quality_remarks(
    pdb_string: str,
    iptm: float | None = None,
    ptm: float | None = None,
    distogram_binding_confidence: float | None = None,
    pair_chains_iptm: float | None = None,
    pair_chains_iptm_b2t: float | None = None,
    pair_chains_iptm_chains: str | None = None,
    ipsae: float | None = None,
    ipsae_b2t: float | None = None,
    ipsae_anchor: int | None = None,
) -> str:
    """Insert REMARK records with predicted quality metrics into a PDB string.

    AlphaFold convention uses REMARK 250 for prediction confidence metadata.
    Metrics are only included when not None, so partial annotation is safe.

    Parameters
    ----------
    pdb_string : str
        Raw PDB string from ``ProteinComplex.to_pdb_string()``.
    iptm : float | None
        Interface predicted TM-score (Hero Critic).
    ptm : float | None
        Global predicted TM-score.
    distogram_binding_confidence : float | None
        Distogram-based binding confidence proxy (all Critics).
    pair_chains_iptm : float | None
        Binder-target pair ipTM, residue-weighted merge of
        ``pair_chains_iptm[target][binder]`` (t2b: target as alignment
        reference, binder placement confidence). Consumed by quality gates.
    pair_chains_iptm_b2t : float | None
        Reverse-direction merge (``[binder][target]``) — diagnostics only,
        never consumed by gates.
    pair_chains_iptm_chains : str | None
        Human-readable per-target-chain t2b values, e.g. ``"A:0.5512,B:0.6734"``.
        Informational; not parsed by ``quality_filter``.
    ipsae : float | None
        ipSAE (d0res 主分数, t2b 方向; ``ipsae.py`` 自实现)。可解析、
        被 Stage-2 ``min_ipsae`` 补充门消费。
    ipsae_b2t : float | None
        反方向 ipSAE — 诊断 only, 不进消费路径。
    ipsae_anchor : int | None
        t2b 方向锚残基 flat 0-idx (结合 target 链长换算链内序号可定位
        界面锚点)。诊断 only。

    Returns
    -------
    str
        PDB string with ``REMARK {PDB_QUALITY_REMARK_NUM}`` lines inserted
        before the first ATOM/HETATM record.
    """
    remarks: list[str] = []
    remarks.append(f"REMARK {PDB_QUALITY_REMARK_NUM} PREDICTED QUALITY METRICS")

    if iptm is not None:
        remarks.append(f"REMARK {PDB_QUALITY_REMARK_NUM}  iPTM={iptm:.4f}")
    if ptm is not None:
        remarks.append(f"REMARK {PDB_QUALITY_REMARK_NUM}  pTM={ptm:.4f}")
    if distogram_binding_confidence is not None:
        remarks.append(
            f"REMARK {PDB_QUALITY_REMARK_NUM}  "
            f"distogram_binding_confidence={distogram_binding_confidence:.4f}"
        )
    ##### claude 2026/10/19 start #####
    # Binder-target pair ipTM (ConfidenceHead pair_chains_iptm): t2b 合并值
    # 是 Stage-2/6 置信度门的消费对象; b2t 与逐链值仅诊断记录。
    if pair_chains_iptm is not None:
        remarks.append(
            f"REMARK {PDB_QUALITY_REMARK_NUM}  "
            f"pair_chains_iptm={pair_chains_iptm:.4f}"
        )
    if pair_chains_iptm_b2t is not None:
        remarks.append(
            f"REMARK {PDB_QUALITY_REMARK_NUM}  "
            f"pair_chains_iptm_b2t={pair_chains_iptm_b2t:.4f}"
        )
    if pair_chains_iptm_chains is not None:
        remarks.append(
            f"REMARK {PDB_QUALITY_REMARK_NUM}  "
            f"pair_chains_iptm_chains={pair_chains_iptm_chains}"
        )
    ##### claude 2026/10/19 end #####
    ##### claude 2026/10/19 start #####
    # ipSAE (d0res 主分数, ipsae.py 自实现): t2b 可解析、被补充门消费;
    # b2t 与锚残基 (flat 0-idx, 结合 target 链长可换算链内序号) 仅诊断。
    if ipsae is not None:
        remarks.append(
            f"REMARK {PDB_QUALITY_REMARK_NUM}  ipsae={ipsae:.4f}"
        )
    if ipsae_b2t is not None:
        remarks.append(
            f"REMARK {PDB_QUALITY_REMARK_NUM}  ipsae_b2t={ipsae_b2t:.4f}"
        )
    if ipsae_anchor is not None:
        remarks.append(
            f"REMARK {PDB_QUALITY_REMARK_NUM}  ipsae_anchor={int(ipsae_anchor)}"
        )
    ##### claude 2026/10/19 end #####

    remark_block = "\n".join(remarks) + "\n"

    # Insert before the first ATOM/HETATM record.
    # For predicted structures without HEADER, this is near the file start.
    lines = pdb_string.split("\n")
    insert_idx = len(lines)  # fallback: append at end
    for i, line in enumerate(lines):
        if line.startswith(("ATOM  ", "HETATM")):
            insert_idx = i
            break

    annotated = (
        "\n".join(lines[:insert_idx])
        + "\n"
        + remark_block
        + "\n".join(lines[insert_idx:])
    )
    return annotated


# ---- LM loss ----


@cache
def _folding_trunk_to_lm_aa_vocab_matrix(device: torch.device) -> torch.Tensor:
    """Build a matrix of shape [ft_aas=20, lm_aas=20]."""
    three_to_one_map = {v: k for k, v in PROTEIN_1TO3.items()}
    ft_aas = [three_to_one_map[tok_3letter] for tok_3letter in TOKENS[2:22]]

    lm_vocab = sorted(ESMCTokenizer().vocab.items(), key=lambda x: x[1])
    lm_aas = [lm_vocab[i][0] for i in range(4, 24)]

    ft_to_lm_aa_matrix = torch.zeros(20, 20)
    for ft_idx, ft_aa in enumerate(ft_aas):
        lm_idx = lm_aas.index(ft_aa)
        ft_to_lm_aa_matrix[ft_idx, lm_idx] = 1

    return ft_to_lm_aa_matrix.to(device=device)


def _one_hot_from_probs(probs: torch.Tensor) -> torch.Tensor:
    return F.one_hot(torch.argmax(probs, dim=-1), num_classes=probs.size(-1)).to(
        probs.dtype
    )


def _straight_through(discrete: torch.Tensor, continuous: torch.Tensor) -> torch.Tensor:
    return continuous + (discrete - continuous).detach()


def compute_esmc_pseudoperplexity_nll(
    esmc_model: ESMCForMaskedLM,
    binder_design: torch.Tensor,
    score_mask: torch.Tensor,
    batch_size: int = 4,
    n_passes: int = 4,
) -> torch.Tensor:
    """Algorithm 14 ESMC Pseudo-perplexity Sequence Regularization.

    Approximate pseudoperplexity NLL via multiple sampled masks."""
    device = binder_design.device
    lm_vocab_size = esmc_model.config.vocab_size
    model_dtype = esmc_model.esmc.embed.weight.dtype

    target_esm = binder_design @ _folding_trunk_to_lm_aa_vocab_matrix(device)
    input_esm = _straight_through(_one_hot_from_probs(target_esm), target_esm)
    input_ids = torch.zeros(
        (binder_design.size(0), binder_design.size(1) + 2, lm_vocab_size),
        dtype=model_dtype,
        device=device,
    )
    tokenizer = ESMCTokenizer()
    input_ids[:, 0, tokenizer.cls_token_id] = 1  # pyright: ignore
    input_ids[:, -1, tokenizer.eos_token_id] = 1  # pyright: ignore
    input_ids[:, 1:-1, 4:24] = input_esm.to(model_dtype)

    if score_mask.ndim == 1:
        score_mask = score_mask.unsqueeze(0).expand(binder_design.size(0), -1)
    elif score_mask.shape != binder_design.shape[:2]:
        raise ValueError(
            f"Expected score_mask with shape {(binder_design.size(0), binder_design.size(1))}, "
            f"got {tuple(score_mask.shape)}"
        )
    score_mask = score_mask.to(device=device, dtype=torch.bool)

    mask_token = torch.zeros(lm_vocab_size, dtype=model_dtype, device=device)
    mask_token[esmc_model.config.mask_token_id] = 1
    esmc = esmc_model.esmc

    all_masked_sequences = []
    all_pass_masks = []
    for batch_idx in range(binder_design.size(0)):
        position_indices = score_mask[batch_idx].nonzero(as_tuple=False).flatten()
        num_positions = int(position_indices.numel())
        if num_positions == 0:
            raise ValueError(
                "ESMC pseudoperplexity score mask selected zero positions."
            )

        num_masked = max(1, math.ceil(ESMC_MASK_FRACTION * num_positions))
        random_scores = torch.rand((n_passes, num_positions), device=device)
        masked_offsets = random_scores.topk(num_masked, dim=-1, largest=False).indices
        pass_masks = torch.zeros(
            (n_passes, binder_design.size(1)), dtype=torch.bool, device=device
        )
        pass_masks[
            torch.arange(n_passes, device=device)[:, None],
            position_indices[masked_offsets],
        ] = True

        masked_sequences = input_ids[batch_idx : batch_idx + 1].repeat(n_passes, 1, 1)
        mask_rows, mask_cols = pass_masks.nonzero(as_tuple=True)
        masked_sequences[mask_rows, mask_cols + 1] = mask_token

        all_masked_sequences.append(masked_sequences)
        all_pass_masks.append(pass_masks)

    logit_chunks = []
    masked_sequence_rows = torch.cat(all_masked_sequences, dim=0)
    for start in range(0, masked_sequence_rows.size(0), batch_size):
        chunk = masked_sequence_rows[start : start + batch_size]
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            hidden, *_ = esmc.transformer(
                chunk @ esmc.embed.weight.to(chunk.dtype),
                sequence_id=None,
                layers_to_collect=[],
                output_attentions=False,
            )
            logit_chunks.append(esmc_model.lm_head(hidden))
    logits = torch.cat(logit_chunks, dim=0)

    losses = []
    for batch_idx, pass_masks in enumerate(all_pass_masks):
        start = batch_idx * n_passes
        stop = start + n_passes
        target_weights = target_esm[batch_idx]
        log_probs = logits[start:stop].log_softmax(dim=-1)[:, 1:-1, 4:24]
        nlls = -(log_probs * target_weights.to(log_probs.dtype).unsqueeze(0)).sum(
            dim=-1
        )
        losses.append(nlls[pass_masks].mean())

    return torch.stack(losses, dim=0)


# ---- Design ----


def normalized_gradient_tensor(
    grad: torch.Tensor, gradient_mask: torch.Tensor
) -> torch.Tensor:
    masked_grad = grad * gradient_mask
    index_has_nonzero_grad = torch.square(masked_grad).sum(-1) > 0  # (B, L)
    eff_L = index_has_nonzero_grad.sum(-1)  # (B,)
    grad_norm = torch.linalg.norm(masked_grad, axis=(-1, -2))  # (B,)
    normalized_grad = (masked_grad / (grad_norm[:, None, None] + 1e-7)) * torch.sqrt(
        eff_L[:, None, None]
    )
    return normalized_grad * gradient_mask


def design_binder(
    inversion_models: dict[str, ESMFold2ExperimentalModel],
    hf_critic_models: dict[str, ESMFold2ExperimentalModel],
    esmc_model: ESMCForMaskedLM,
    scaling_critic_names: list[str] | None,
    target_name: str,
    target_sequence: str | None,
    binder_name: str,
    binder_sequence: str | None,
    is_antibody: bool | None,
    seed: int,
    batch_size: int = 1,
    target_hotspot_ids: list[str] | None = None,
    epitope_contact_distance: float = 12.0,
    inter_loss_mode: str | None = None,
    fold_rescue_schedule: bool | None = None,
) -> tuple[list[str], Trajectory, list[dict]]:
    """
    Algorithm 11 Gradient-Guided Binder Sequence Optimization.

    Run the full optimization loop.
    Returns dict with designed_sequence, complex, and trajectory.

    Every critic is folded once on the best designed sequence via HF ESMFold2.
    Hero critics expose iPTM; scaling critics contribute distogram scores only.
    ``distogram_binding_confidence`` / ``cdr_distogram_binding_confidence`` come
    from the distogram in all cases.

    ``target_hotspot_ids`` enables epitope loss; hotspot IDs are chain-qualified
    as ``<chain>:<residue><idx>`` (e.g. ``["A:L150"]``) with 1-indexed residue
    positions within the named target chain.

    ``inter_loss_mode`` selects the inter contact loss implementation:
    ``None`` (default) auto-resolves — hotspot designs
    (``target_hotspot_ids`` given) use ``"hotspot_neighborhood"`` (a
    distogram-precomputed surface-accessible hotspot neighborhood),
    no-hotspot designs use ``"full_coverage"``; an explicit
    ``"full_coverage"`` overrides the auto default for hotspot designs.
    ``fold_rescue_schedule`` enables the three-phase fold-rescue weight
    schedule: native weights unless folding stalls, then rescue weights,
    then full hotspot weights for interface refinement once folding
    completes (see ``update_fold_rescue_schedule``). ``None`` (default)
    auto-resolves — hotspot designs enable the schedule, no-hotspot
    designs keep it off (2026-10-21 semantics, mirroring
    ``inter_loss_mode``); an explicit ``False`` opts out for hotspot
    designs, an explicit ``True`` without hotspots raises ValueError.
    """
    # Setup
    device = "cuda"
    if target_name in TARGET_SEQUENCES:
        if target_sequence is not None:
            raise ValueError(
                f"{target_name!r} is a preset target; omit target_sequence."
            )
        target_sequence = TARGET_SEQUENCES[target_name]
    elif target_sequence is None:
        raise ValueError(
            f"{target_name!r} is not a preset target; provide target_sequence."
        )
    ##### claude 2026/8/13 start #####
    # Multi-chain targets are supported via "|" chainbreaks in
    # target_sequence. Each chain is encoded independently by
    # build_target_one_hot; the binder remains a single chain.
    target_one_hot = build_target_one_hot(target_sequence, device=device)
    ##### claude 2026/8/13 end #####

    if binder_name in BINDER_PROMPT_FACTORIES:
        if binder_sequence is not None:
            raise ValueError(
                f"{binder_name!r} is a preset binder; omit binder_sequence."
            )
        binder_prompt_factor = BINDER_PROMPT_FACTORIES[binder_name]
        if is_antibody is not None:
            assert (
                binder_prompt_factor.is_antibody == is_antibody
            ), "Conflict in is_antibody settings."
        is_antibody = binder_prompt_factor.is_antibody
        binder_sequence = binder_prompt_factor.sample(seed=seed)
    elif binder_sequence is None:
        raise ValueError(
            f"{binder_name!r} is not a preset binder; provide binder_sequence."
        )
    elif is_antibody is None:
        is_antibody = _is_antibody_sequence(binder_sequence)

    binder_length = len(binder_sequence)

    ##### claude 2026/8/13 start #####
    # Multi-chain targets are supported: target_sequence may contain "|"
    # chainbreaks (each chain becomes an independent ProteinInput with its
    # own asym_id). The binder remains a single chain — its soft-sequence
    # optimization is unchanged.
    assert "|" not in binder_sequence
    ##### claude 2026/8/13 end #####

    ##### claude 2026/10/12 start #####
    # Parallel inter-loss mode: precompute (ONCE) the surface-accessible
    # hotspot neighborhood from a distogram of the fixed target and restrict
    # the inter contact loss averaging domain to it. The target sequence does
    # not change during optimization, so the mask is a constant thereafter.
    # (Placed after binder_sequence resolution: needs binder_length.)
    # 2026-10-14: ``inter_loss_mode=None`` auto-resolves — hotspot designs
    # default to the neighborhood mask, no-hotspot designs to full coverage;
    # an explicit "full_coverage" keeps the legacy hotspot behavior.
    if inter_loss_mode is None:
        inter_loss_mode = (
            "hotspot_neighborhood" if target_hotspot_ids else "full_coverage"
        )
    # Fold-rescue phase schedule: hotspot design mode only (user spec
    # 2026-10-14). Per-run state + weight overlay consumed by run_step;
    # transitions shift loss weights only.
    ##### claude 2026/10/21 start #####
    # None = auto (mirrors inter_loss_mode): hotspot designs enable the
    # schedule by default (semantic change 2026-10-21), no-hotspot designs
    # keep it off. Explicit False opts out; explicit True without hotspots
    # still hits the ValueError below.
    if fold_rescue_schedule is None:
        fold_rescue_schedule = target_hotspot_ids is not None
    ##### claude 2026/10/21 end #####
    if fold_rescue_schedule and target_hotspot_ids is None:
        raise ValueError(
            "fold_rescue_schedule requires target_hotspot_ids "
            "(hotspot design mode only)."
        )
    if fold_rescue_schedule:
        fold_rescue_state: dict = {"call": 0, "phase": 1, "intra": [],
                                   "events": []}
        fold_rescue_weights: dict = dict(LOSS_WEIGHTS)
    else:
        fold_rescue_state = None
        fold_rescue_weights = None
    inter_neighborhood_mask: torch.Tensor | None = None
    if inter_loss_mode == "full_coverage":
        pass
    elif inter_loss_mode == "hotspot_neighborhood":
        if target_hotspot_ids is None:
            raise ValueError(
                'inter_loss_mode="hotspot_neighborhood" requires target_hotspot_ids.'
            )
        inter_neighborhood_mask = precompute_inter_neighborhood_mask(
            next(iter(inversion_models.values())),
            target_sequence,
            target_one_hot,
            binder_length,
            target_hotspot_ids,
            batch_size=batch_size,
            device=device,
            seed=seed,
        )
        if inter_neighborhood_mask is None:
            logger.warning(
                "inter_loss_mode='hotspot_neighborhood': neighborhood mask is "
                "degenerately small (<%d residues); falling back to full "
                "coverage inter loss.",
                INTER_NEIGHBORHOOD_MIN_SIZE,
            )
        else:
            logger.info(
                "inter_loss_mode='hotspot_neighborhood': %d/%d target residues "
                "selected for the inter loss domain (hotspots %s).",
                int(inter_neighborhood_mask.sum()),
                len(target_sequence.replace("|", "")),
                target_hotspot_ids,
            )
    else:
        raise ValueError(
            f"Unknown inter_loss_mode {inter_loss_mode!r}; expected "
            '"full_coverage" or "hotspot_neighborhood".'
        )
    ##### claude 2026/10/12 end #####

    with seed_context(seed), torch.device(device):
        logits = build_initial_soft_sequence_logits(
            binder_sequence, batch_size=batch_size
        )
        gradient_mask = build_gradient_mask(binder_sequence, batch_size=batch_size)

    # step -> {loss_name: [B] tensor on CPU, metric_name: float}
    trajectory: Trajectory = {}
    global_step = 0

    def run_step(
        logits: torch.Tensor,
        optimizer: optim.Optimizer,
        temperature: float,
        calculate_confidence: bool,
    ) -> tuple[torch.Tensor, list[str], list[float] | None]:
        nonlocal global_step
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        optimizer.zero_grad()

        random.seed(seed + global_step)
        replicate_choice = random.randint(0, len(inversion_models) - 1)
        inversion_model = list(inversion_models.values())[replicate_choice]
        design = F.softmax(logits / temperature, dim=-1)

        fold_result = fold_and_get_distogram(
            inversion_model,
            target_sequence,
            target_one_hot,
            design,
            num_loops=1,
            num_sampling_steps=50 if calculate_confidence else 1,
            calculate_confidence=calculate_confidence,
            seed=seed + global_step,
        )
        sequences: list[str] = fold_result["seq_list"]
        losses = compute_structure_losses(
            fold_result["distogram_logits"],
            binder_length,
            target_sequence=target_sequence,
            target_hotspot_ids=target_hotspot_ids,
            epitope_contact_distance=epitope_contact_distance,
            inter_neighborhood_mask=inter_neighborhood_mask,
            loss_weights=fold_rescue_weights,
        )
        if fold_rescue_state is not None:
            update_fold_rescue_schedule(
                fold_rescue_state,
                fold_rescue_weights,
                global_step,
                float(losses["intra_contact_loss"].mean()),
            )
        structure_loss = losses["total_loss"]
        structure_grad = torch.autograd.grad(structure_loss.mean(), logits)[0]

        # Recompute the logits -> design transform for a fresh graph.
        design = F.softmax(logits / temperature, dim=-1)
        score_mask = gradient_mask.sum(dim=-1) > 0
        with seed_context(seed + global_step):
            plm_loss = compute_esmc_pseudoperplexity_nll(
                esmc_model=esmc_model,
                binder_design=design,
                score_mask=score_mask,
                batch_size=LM_LOSS_BATCH_SIZE,
                n_passes=LM_MASK_PASSES,
            )
        plm_grad = torch.autograd.grad(plm_loss.mean(), logits)[0]

        logits.grad = normalized_gradient_tensor(structure_grad, gradient_mask) + (
            0.05 if is_antibody else 0.15
        ) * normalized_gradient_tensor(plm_grad, gradient_mask)

        for g in optimizer.param_groups:
            g["lr"] = LEARNING_RATE * temperature

        optimizer.step()

        step = global_step
        step_losses: TrajectoryStep = {k: v.detach().cpu() for k, v in losses.items()}
        step_losses["plm_loss"] = plm_loss.detach().cpu()
        step_losses["total_loss"] = (structure_loss + plm_loss).detach().cpu()
        step_losses["time"] = time.time() - start
        step_losses["peak_allocated_gib"] = (
            torch.cuda.max_memory_allocated() / BYTES_PER_GIB
        )
        step_losses["peak_reserved_gib"] = (
            torch.cuda.max_memory_reserved() / BYTES_PER_GIB
        )
        trajectory[step] = step_losses
        loss_str = "  ".join(
            f"{k}={v.mean().item() if torch.is_tensor(v) else v:.4f}"
            for k, v in step_losses.items()
        )
        if step % LOG_INTERVAL == 0:
            logger.info(f"  step {step:3d}  |  {loss_str}  T={temperature:.4f}")
        global_step += 1
        return logits, sequences, fold_result.get("iptm", None)

    # Optimize
    optimizer = optim.SGD([logits], lr=LEARNING_RATE)
    best_iptm: list[float] = [-1.0] * batch_size
    best_sequences: list[str] = [""] * batch_size
    for step in range(STEPS):
        # Cosine schedule
        t = (step + 1) / STEPS
        remaining = 0.5 * (1 + math.cos(math.pi * t))
        temperature = TEMPERATURE_MIN + (1 - TEMPERATURE_MIN) * remaining
        logits, sequences, iptm = run_step(
            logits,
            optimizer,
            temperature=temperature,
            calculate_confidence=temperature < 0.05,
        )
        if iptm is not None:
            for b in range(batch_size):
                if iptm[b] is not None and iptm[b] > best_iptm[b]:
                    best_iptm[b] = iptm[b]
                    best_sequences[b] = sequences[b]
    assert all(seq != "" for seq in best_sequences)

    # Score
    critic_results: list[dict] = []
    target_length = len(target_sequence.replace("|", ""))
    final_total_loss = trajectory[global_step - 1]["total_loss"]
    assert isinstance(final_total_loss, torch.Tensor)

    def score_critic(
        critic_name: str,
        critic_model: ESMFold2ExperimentalModel,
        batch_idx: int,
        is_scaling_critic: bool,
    ) -> None:
        best_seq = best_sequences[batch_idx]
        binder_seq = best_seq.split("|")[-1]
        binder_design = sequence_to_one_hot(binder_seq)[..., 2:22]
        final_fold = fold_and_get_distogram(
            critic_model,
            target_sequence,
            target_one_hot,
            binder_design,
            num_loops=3,
            num_sampling_steps=1 if is_scaling_critic else 200,
            calculate_confidence=not is_scaling_critic,
            seed=seed,
        )
        pred_complex = (
            None
            if is_scaling_critic
            else build_complex(final_fold["inputs"], final_fold["output"])
        )
        iptm_proxy_scores = compute_distogram_iptm_proxy(
            final_fold["distogram_logits"], target_length, binder_seq, is_antibody
        )
        iptm_tensor = final_fold.get("iptm")
        iptm = iptm_tensor.item() if iptm_tensor is not None else None
        ptm_tensor = final_fold.get("ptm")
        ptm = ptm_tensor.item() if ptm_tensor is not None else None

        ##### claude 2026/10/19 start #####
        # Binder-target pair ipTM (ConfidenceHead pair_chains_iptm)。消费方向
        # = [target][binder] (t2b: 固定 target 正确, 评估 binder 位置置信度),
        # 多链 target 按残基加权合并 (binder×target 子块均值)。scaling critic
        # 无 confidence head → output 里无此键 → 条目自然缺失该指标 (向后
        # 兼容, Stage-2 门按 None 跳过)。
        pair_detail: dict | None = None
        if not is_scaling_critic:
            pair_matrix = final_fold["output"].get("pair_chains_iptm")
            if pair_matrix is not None:
                pair_detail = merge_pair_chains_iptm(
                    pair_matrix[batch_idx],
                    [len(c) for c in target_sequence.split("|")],
                )
        # ipSAE (d0res 主分数, 与 pair_chains_iptm 同级的补充信号): 同一
        # output["pae"] 直接计算, 与 annotated_pdb 同源同折。局部 import —
        # ipsae.py 在 pipeline 根目录 (共享度量模块), 保持本文件可被 Modal
        # 镜像独立加载 (根目录入 sys.path 后方可解析)。
        ipsae_detail: dict | None = None
        if not is_scaling_critic:
            from ipsae import compute_ipsae

            pae_matrix = final_fold["output"].get("pae")
            if pae_matrix is not None:
                ipsae_detail = compute_ipsae(
                    pae_matrix[batch_idx],
                    [len(c) for c in target_sequence.split("|")],
                )
        ##### claude 2026/10/19 end #####

        # Build annotated PDB with quality metrics embedded as REMARK records.
        annotated_pdb: str | None = None
        if pred_complex is not None:
            ##### claude 2026/8/13 start #####
            # 修复 key 映射: compute_distogram_iptm_proxy 返回的键是
            # "distogram_iptm_proxy" (旧代码用 "distogram_binding_confidence"
            # 导致 .get() 恒为 None, proxy 从未写入 REMARK 250)。
            annotated_pdb = pdb_with_quality_remarks(
                pred_complex.to_pdb_string(),
                iptm=iptm,
                ptm=ptm,
                distogram_binding_confidence=iptm_proxy_scores.get(
                    "distogram_iptm_proxy"
                ),
                ##### claude 2026/10/19 start #####
                pair_chains_iptm=(
                    pair_detail["merged_t2b"] if pair_detail else None
                ),
                pair_chains_iptm_b2t=(
                    pair_detail["merged_b2t"] if pair_detail else None
                ),
                pair_chains_iptm_chains=(
                    ",".join(
                        f"{label}:{vals['t2b']:.4f}"
                        for label, vals in pair_detail["chains"].items()
                    )
                    if pair_detail
                    else None
                ),
                ##### claude 2026/10/19 end #####
                ##### claude 2026/10/19 start #####
                ipsae=ipsae_detail["ipsae_t2b"] if ipsae_detail else None,
                ipsae_b2t=ipsae_detail["ipsae_b2t"] if ipsae_detail else None,
                ipsae_anchor=(
                    ipsae_detail["anchor_t2b"] if ipsae_detail else None
                ),
                ##### claude 2026/10/19 end #####
            )
            ##### claude 2026/8/13 end #####

        critic_results.append(
            {
                "is_antibody": is_antibody,
                "critic_name": critic_name,
                "is_scaling_critic": is_scaling_critic,
                "batch_idx": batch_idx,
                "designed_sequence": best_seq,
                "complex": pred_complex,
                "annotated_pdb": annotated_pdb,
                "final_loss": final_total_loss[batch_idx].item(),
                "iptm": iptm,
                "ptm": ptm,
                ##### claude 2026/10/19 start #####
                # merged t2b 标量 (Stage-2/6 消费) + 完整明细 (b2t/逐链, 仅诊断)
                "pair_chains_iptm": (
                    pair_detail["merged_t2b"] if pair_detail else None
                ),
                "pair_chains_iptm_detail": pair_detail,
                # ipSAE (补充信号): t2b 标量 + 明细 (锚残基/逐链, 诊断)
                "ipsae": ipsae_detail["ipsae_t2b"] if ipsae_detail else None,
                "ipsae_detail": ipsae_detail,
                ##### claude 2026/10/19 end #####
                "logits": logits[batch_idx].detach().cpu(),
                "fold_rescue_events": (
                    list(fold_rescue_state["events"])
                    if fold_rescue_state is not None
                    else None
                ),
                **iptm_proxy_scores,
            }
        )
        del final_fold

    for critic_name, critic_model in hf_critic_models.items():
        for batch_idx in range(batch_size):
            score_critic(critic_name, critic_model, batch_idx, is_scaling_critic=False)

    for critic_name in scaling_critic_names or []:
        critic_model = _load_hf_model(
            critic_name, lm_dropout=0.25, cache_esmc=False, device="cuda"
        )
        for batch_idx in range(batch_size):
            score_critic(critic_name, critic_model, batch_idx, is_scaling_critic=True)
        del critic_model
        torch.cuda.empty_cache()

    if not critic_results:
        for batch_idx in range(batch_size):
            critic_results.append(
                {
                    "is_antibody": is_antibody,
                    "batch_idx": batch_idx,
                    "designed_sequence": best_sequences[batch_idx],
                    "final_loss": final_total_loss[batch_idx].item(),
                    "logits": logits[batch_idx].detach().cpu(),
                }
            )

    return best_sequences, trajectory, critic_results


# ---- Model Loading ----

_ESMC_CACHE: dict[str, torch.nn.Module] = {}


def _load_hf_model(
    critic_name: str, lm_dropout: float, cache_esmc: bool, device: str
) -> Any:
    """Loads ESMFold2 from huggingface. Will cache ESMC by checkpoint ID among
    all non-scaling checkpoints, to save on VRAM and load time."""
    repo_id = f"biohub/{critic_name}"
    model = ESMFold2ExperimentalModel.from_pretrained(repo_id, load_esmc=not cache_esmc)
    if cache_esmc:
        esmc_id = model.config.esmc_id
        if esmc_id not in _ESMC_CACHE:
            model.load_esmc(esmc_id)
            assert model._esmc is not None
            _ESMC_CACHE[esmc_id] = model._esmc
        model._esmc = _ESMC_CACHE[esmc_id]
    model.configure_lm_dropout(lm_dropout, force_lm_dropout_during_inference=True)
    kernel_backend = None
    if TRITON_KERNELS_AVAILABLE:
        kernel_backend = BACKEND_FUSED
    elif CUE_AVAILABLE:
        kernel_backend = BACKEND_CUEQ
    model.set_kernel_backend(kernel_backend)
    return model.to(device=device).eval().requires_grad_(False)


def _apply_torch_compile(model: torch.nn.Module) -> None:
    """A helper for torch compiling the model."""
    torch._dynamo.config.cache_size_limit = 512
    torch._dynamo.config.accumulated_cache_size_limit = 512

    compile_targets = (ESMFold2MSAEncoder, PairUpdateBlock)

    def _maybe_compile_module(module: torch.nn.Module) -> None:
        if not isinstance(module, compile_targets):
            return
        module.forward = torch.compile(module.forward)  # ty:ignore[invalid-assignment]

    model.apply(_maybe_compile_module)


class ESMFold2Design:
    lm_name = "biohub/ESMC-6B"
    inversion_model_names: list[str] = [
        "ESMFold2-Experimental-Fast",
        "ESMFold2-Experimental-Fast-Cutoff2025",
    ]
    hero_critic_hf_paths: list[str] = [
        "ESMFold2-Experimental-Fast",
        "ESMFold2-Experimental-Fast-Cutoff2025",
        "ESMFold2-Experimental",
        "ESMFold2-Experimental-Cutoff2025",
    ]
    scaling_critic_hf_paths: list[str] = []

    def load(self, use_scaling_critics: bool):
        self.scaling_critic_hf_paths = []
        if use_scaling_critics:
            self.scaling_critic_hf_paths = [
                f"ESMFold2-Experimental-Fast-base{size}-step{step}k"
                for size in ("300M", "600M", "6B")
                for step in ("250", "500", "750", "1000", "1500")
            ]

        self.inversion_models = {
            model_name: _load_hf_model(
                model_name, lm_dropout=0.5, cache_esmc=True, device="cuda"
            )
            for model_name in self.inversion_model_names
        }
        if COMPILE:
            for model in self.inversion_models.values():
                _apply_torch_compile(model)

        self.hf_critic_models: dict[str, Any] = {}
        for name in self.hero_critic_hf_paths:
            self.hf_critic_models[name] = _load_hf_model(
                name, lm_dropout=0.25, cache_esmc=True, device="cuda"
            )

        self.esmc_model = ESMCForMaskedLM.from_pretrained(
            self.lm_name, torch_dtype=torch.float32
        )
        if REUSE_ESMC:
            reusable_esmc_model = self.inversion_models["ESMFold2-Experimental-Fast"]
            assert reusable_esmc_model.config.esmc_id == self.lm_name, (
                f"Cannot reuse ESMC trunk from {reusable_esmc_model.config.esmc_id!r} "
                f"with LM head from {self.lm_name!r}."
            )
            assert reusable_esmc_model._esmc is not None
            del self.esmc_model.esmc
            torch.cuda.empty_cache()
            self.esmc_model.esmc = reusable_esmc_model._esmc
        self.esmc_model = self.esmc_model.cuda().eval().requires_grad_(False)

    def design(
        self,
        target_name: str,
        binder_name: str,
        target_sequence: str | None = None,
        binder_sequence: str | None = None,
        is_antibody: bool | None = None,
        seed: int = 0,
        batch_size: int = 1,
        target_hotspot_ids: list[str] | None = None,
        epitope_contact_distance: float = 12.0,
        inter_loss_mode: str | None = None,
        ##### claude 2026/10/21 start #####
        # None = auto: 热点任务默认启用三阶段调度 (语义变更 2026-10-21),
        # 显式 False 可退出; 解析在 design_binder 内 (与 inter_loss_mode 同址)。
        fold_rescue_schedule: bool | None = None,
        ##### claude 2026/10/21 end #####
    ) -> tuple[list[str], Trajectory, list[dict]]:
        return design_binder(
            self.inversion_models,
            self.hf_critic_models,
            self.esmc_model,
            self.scaling_critic_hf_paths,
            target_name=target_name,
            target_sequence=target_sequence,
            binder_name=binder_name,
            binder_sequence=binder_sequence,
            is_antibody=is_antibody,
            seed=seed,
            batch_size=batch_size,
            target_hotspot_ids=target_hotspot_ids,
            epitope_contact_distance=epitope_contact_distance,
            inter_loss_mode=inter_loss_mode,
            fold_rescue_schedule=fold_rescue_schedule,
        )


# ---- Modal ----


def get_base_image():
    return (
        modal.Image.micromamba(python_version="3.12")
        .run_commands("apt update && apt install -y git build-essential")
        .micromamba_install(
            "anarci>=2020.04.03",
            "hmmer=3.4",
            "cuda-version=12.8",
            "cuda-libraries-dev=12.8",
            "cuda-nvcc=12.8",
            "cmake",
            "ninja",
            channels=["conda-forge", "bioconda"],
        )
        .pip_install(
            "torch==2.8.0",
            "triton==3.4.0",
            index_url="https://download.pytorch.org/whl/cu128",
        )
        .pip_install(
            "flash-attn==2.8.3",
            "transformer-engine[core-cu12,pytorch]==2.13.0",
            "xformers==0.0.32.post1",
            extra_options="--no-build-isolation",
            env={
                "CPATH": (
                    "/opt/conda/lib/python3.12/site-packages/nvidia/cudnn/include:"
                    "/opt/conda/lib/python3.12/site-packages/nvidia/nccl/include:"
                    "/opt/conda/lib/python3.12/site-packages/nvidia/nvtx/include"
                ),
                "LIBRARY_PATH": (
                    "/opt/conda/lib/python3.12/site-packages/nvidia/cudnn/lib:"
                    "/opt/conda/lib/python3.12/site-packages/nvidia/nccl/lib:"
                    "/opt/conda/lib/python3.12/site-packages/nvidia/nvtx/lib"
                ),
                "MAX_JOBS": "8",
                "NVTE_FRAMEWORK": "pytorch",
            },
        )
        .pip_install(
            "abnumber", "esm@git+https://github.com/Biohub/esm.git@main", "modal"
        )
        .env(
            {
                "HF_HOME": "/models",
                "HF_XET_HIGH_PERFORMANCE": "1",
                "XFORMERS_IGNORE_FLASH_VERSION_CHECK": "1",
            }
        )
    )


app = modal.App(
    name="esmfold2-design",
    image=get_base_image(),
    volumes={
        "/models": modal.Volume.from_name("esmfold2-models", create_if_missing=True)
    },
)


# NOTE - Currently the memory usage is quite high, inflating costs.
# In an update coming soon, scaling critics will be loaded on demand
# to avoid needing this large amount of RAM.
@app.cls(gpu="H100", timeout=60 * 60, cpu=16, memory=80 * 1024)
class ESMFold2DesignModal(ESMFold2Design):
    """Modal entrypoint. Hero critics are HF experimental exports with
    confidence heads. Set ``use_scaling_critics=True`` to also load the
    15-checkpoint scaling-experiment ensemble (distogram binding confidence only).
    """

    use_scaling_critics: bool = modal.parameter(default=True)
    deterministic: bool = modal.parameter(default=False)

    @modal.enter()
    def load(self):
        if self.deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
        return super().load(self.use_scaling_critics)

    @modal.method()
    def design(self, *args, **kws):
        return super().design(*args, **kws)


@app.local_entrypoint()
def main(
    target_name: str,
    binder_name: str,
    target_sequence: str | None = None,
    binder_sequence: str | None = None,
    use_scaling_critics: bool = True,
    is_antibody: bool | None = None,
    local: bool = False,
    seed: int = 0,
    batch_size: int = 1,
    target_hotspot_ids: list[str] | None = None,
    epitope_contact_distance: float = 12.0,
):
    if local:
        assert not use_scaling_critics, (
            "'abnumber' will fail if running this script with uv run. "
            "It requires conda packages.  To be addressed soon."
        )
        app = ESMFold2Design()
        app.load(use_scaling_critics)
        run_fn = app.design
    else:
        app = ESMFold2DesignModal(
            use_scaling_critics=use_scaling_critics  # ty:ignore[unknown-argument]
        )
        run_fn = app.design.remote

    seq, trajectory, results = run_fn(
        target_name=target_name,
        target_sequence=target_sequence,
        binder_name=binder_name,
        binder_sequence=binder_sequence,
        is_antibody=is_antibody,
        seed=seed,
        batch_size=batch_size,
        target_hotspot_ids=target_hotspot_ids,
        epitope_contact_distance=epitope_contact_distance,
    )

    avg_final_loss = sum(r["final_loss"] for r in results) / len(results)
    logger.info(f"\nDesigned sequence: {seq}")
    logger.info(f"Trajectory length: {len(trajectory)} steps")
    logger.info(f"Average final loss: {avg_final_loss:.4f}")


if __name__ == "__main__":
    # Run a single local design.
    main(
        # Example case 1
        target_name="pd-l1",
        binder_name="minibinder",
        is_antibody=False,
        # Example case 2
        # target_name="cd45",
        # binder_name="trastuzumab_framework_vhvl",
        # is_antibody=True,
        # Common settings
        seed=0,
        batch_size=1,
        local=True,
        use_scaling_critics=True,
    )
