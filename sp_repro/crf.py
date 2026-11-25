import torch
from torch import nn

from .data import CLASSES, MAX_RESIDUES, Example


SIGNAL_REGIONS = ((), ("n", "h", "c"), ("n", "h"), ("n", "RR", "h", "c"), ("n", "RR", "h"), ("signal",))
TOPOLOGY = ("I", "M", "O")
STATES = tuple(regions + TOPOLOGY for regions in SIGNAL_REGIONS)
STATE_COUNT = max(map(len, STATES))
BLOCKED = -10_000.0
HYDROPATHY = dict(zip("ACDEFGHIKLMNPQRSTVWY", (1.8, 2.5, -3.5, -3.5, 2.8, -0.4, -3.2, 4.5, -3.9, 3.8, 1.9, -3.5, -1.6, -3.5, -4.5, -0.8, -0.7, 4.2, -0.9, -1.3)))


def state_constraints() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    start = torch.zeros((len(CLASSES), STATE_COUNT), dtype=torch.bool)
    end = torch.zeros_like(start)
    allowed = torch.zeros((len(CLASSES), STATE_COUNT, STATE_COUNT), dtype=torch.bool)
    for label, regions in enumerate(SIGNAL_REGIONS):
        mature = len(regions)
        end[label, mature:mature + len(TOPOLOGY)] = True
        allowed[label, mature:mature + len(TOPOLOGY), mature:mature + len(TOPOLOGY)] = True
        if not regions:
            start[label] = end[label]
            continue
        start[label, 0] = True
        for state in range(mature):
            allowed[label, state, state] = True
            if state + 1 < mature:
                allowed[label, state, state + 1] = True
        if "RR" in regions:
            allowed[label, 0, 2] = True
        allowed[label, mature - 1, mature:mature + len(TOPOLOGY)] = True
    return start, end, allowed


def weak_labels(example: Example) -> torch.Tensor:
    allowed = torch.zeros((MAX_RESIDUES, STATE_COUNT), dtype=torch.bool)
    states = STATES[example.label]
    regions = SIGNAL_REGIONS[example.label]
    for position in range(example.cut, min(len(example.tags), MAX_RESIDUES)):
        allowed[position, states.index(example.tags[position])] = True
    if not regions:
        return allowed

    cut = example.cut
    allowed[:cut, :len(regions)] = True
    if len(regions) == 1:
        return allowed

    def constrain(position: int, region: str) -> None:
        allowed[position].fill_(False)
        allowed[position, regions.index(region)] = True

    constrain(0, "n")
    constrain(cut - 1, regions[-1])
    h_start = 1
    h_stop = cut - int(regions[-1] == "c")
    sequence = example.protein.sequence
    if "RR" in regions:
        rr_start = sequence.find("RR", 1, h_stop - 1)
        allowed[:cut, regions.index("RR")] = False
        if rr_start != -1:
            constrain(rr_start, "RR")
            constrain(rr_start + 1, "RR")
            h_start = rr_start + 2

    if h_start < h_stop and regions[-1] == "c":
        def hydropathy(position: int) -> float:
            window = sequence[max(0, position - 3):min(cut, position + 4)]
            return sum(HYDROPATHY.get(aa, 0.0) for aa in window) / len(window)

        constrain(max(range(h_start, h_stop), key=hydropathy), "h")
    return allowed


class SignalCRF(nn.Module):
    def __init__(self, encoder_width: int):
        super().__init__()
        self.emissions = nn.Sequential(
            nn.Linear(encoder_width + 1, 64), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(64, len(CLASSES) * STATE_COUNT),
        )
        start, end, allowed = state_constraints()
        self.register_buffer("start_allowed", start)
        self.register_buffer("end_allowed", end)
        self.register_buffer("transition_allowed", allowed)
        self.start = nn.Parameter(torch.zeros(start.shape))
        self.end = nn.Parameter(torch.zeros(end.shape))
        self.trans = nn.Parameter(torch.zeros(allowed.shape))

    def emission_scores(self, features: torch.Tensor, eukarya: torch.Tensor) -> torch.Tensor:
        group = eukarya[:, None, None].expand(-1, features.shape[1], 1)
        return self.emissions(torch.cat((features.float(), group.float()), dim=-1)).unflatten(
            -1, (len(CLASSES), STATE_COUNT)
        )

    def transitions(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.start.masked_fill(~self.start_allowed, BLOCKED),
            self.end.masked_fill(~self.end_allowed, BLOCKED),
            self.trans.masked_fill(~self.transition_allowed, BLOCKED),
        )

    @staticmethod
    def partition(
        emissions: torch.Tensor, lengths: torch.Tensor,
        start: torch.Tensor, end: torch.Tensor, trans: torch.Tensor,
    ) -> torch.Tensor:
        # Each block contains paths for a single sequence class.
        alpha = emissions[:, 0] + start
        for position in range(1, emissions.shape[1]):
            next_alpha = (alpha.unsqueeze(-1) + trans).logsumexp(dim=-2) + emissions[:, position]
            shape = (-1,) + (1,) * (alpha.ndim - 1)
            alpha = torch.where((lengths > position).view(shape), next_alpha, alpha)
        return (alpha + end).logsumexp(dim=-1)

    def log_partitions(self, scores: torch.Tensor, lengths: torch.Tensor, eukarya: torch.Tensor) -> torch.Tensor:
        result = self.partition(scores, lengths, *self.transitions())
        return result.masked_fill(eukarya[:, None] & (torch.arange(len(CLASSES), device=scores.device) > 1), BLOCKED)

    def loss(self, features, lengths, eukarya, labels, allowed, class_weights):
        scores = self.emission_scores(features, eukarya)
        start, end, trans = self.transitions()
        normalizer = self.log_partitions(scores, lengths, eukarya).logsumexp(dim=-1)
        # The numerator sums every path consistent with the weak region labels.
        constrained = scores[torch.arange(len(labels), device=labels.device), :, labels].masked_fill(~allowed, BLOCKED)
        numerator = self.partition(constrained, lengths, start[labels], end[labels], trans[labels])
        weights = class_weights[labels]
        return ((normalizer - numerator) / lengths * weights).mean()

    def decode(
        self, scores: torch.Tensor, lengths: torch.Tensor, labels: torch.Tensor,
        *, confidence: bool, per_residue: bool = False,
    ):
        batch = torch.arange(len(labels), device=labels.device)
        emissions = scores[batch, :, labels]
        start, end, trans = (part[labels] for part in self.transitions())
        best = emissions[:, 0] + start
        forward = [best]
        backpointers = []
        for position in range(1, emissions.shape[1]):
            candidate, previous = (best.unsqueeze(-1) + trans).max(dim=-2)
            best = torch.where((lengths > position)[:, None], candidate + emissions[:, position], best)
            backpointers.append(previous)
            if confidence:
                forward.append((forward[-1].unsqueeze(-1) + trans).logsumexp(dim=-2) + emissions[:, position])

        state = (best + end).argmax(dim=-1)
        path = torch.zeros(emissions.shape[:2], dtype=torch.long, device=emissions.device)
        for position in reversed(range(emissions.shape[1])):
            path[:, position] = state
            if position:
                state = torch.where(lengths > position, backpointers[position - 1][batch, state], state)

        signal_lengths = labels.new_tensor([len(regions) for regions in SIGNAL_REGIONS])[labels]
        valid = torch.arange(emissions.shape[1], device=lengths.device) < lengths[:, None]
        cut = ((path < signal_lengths[:, None]) & valid).sum(dim=-1)
        if not confidence:
            return path, cut, None, None

        backward = [end]
        for position in reversed(range(1, emissions.shape[1])):
            next_beta = (trans + (emissions[:, position] + backward[-1]).unsqueeze(-2)).logsumexp(dim=-1)
            backward.append(torch.where((lengths > position)[:, None], next_beta, end))
        alpha = torch.stack(forward, dim=1)
        beta = torch.stack(backward[::-1], dim=1)
        normalizer = (alpha[batch, lengths - 1] + end).logsumexp(dim=-1)
        marginals = (alpha + beta - normalizer[:, None, None]).softmax(dim=-1) if per_residue else None
        selected_cut = cut.clamp(min=1, max=emissions.shape[1] - 1)
        last_sp = (signal_lengths - 1).clamp_min(0)
        cut = torch.where(cut < lengths, cut, torch.zeros_like(cut))
        # Sum the probability of the selected signal-to-mature transition.
        left = alpha[batch, selected_cut - 1, last_sp]
        right = trans[batch, last_sp] + emissions[batch, selected_cut] + beta[batch, selected_cut]
        right = right.masked_fill(torch.arange(STATE_COUNT, device=labels.device) < signal_lengths[:, None], BLOCKED)
        probability = (left + right.logsumexp(dim=-1) - normalizer).exp().clamp(max=1)
        return path, cut, torch.where(cut > 0, probability, torch.zeros_like(probability)), marginals
