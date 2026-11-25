from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


CLASSES = ("OTHER", "SP", "LIPO", "TAT", "TATLIPO", "PILIN")
CLASS_TYPES = ("OTHER", "Sec/SPI", "Sec/SPII", "Tat/SPI", "Tat/SPII", "Sec/SPIII")
SIGNAL_TAGS = ("", "S", "L", "T", "T", "P")
MAX_RESIDUES = 70
CANONICAL_RESIDUES = frozenset("ACDEFGHIKLMNPQRSTVWY")


@dataclass(frozen=True)
class Protein:
    identifier: str
    sequence: str


@dataclass(frozen=True)
class Example:
    protein: Protein
    label: int
    kingdom: str
    fold: int
    tags: str
    cut: int

    @property
    def eukarya(self) -> bool:
        return self.kingdom == "EUKARYA"


def normalize_sequence(sequence: str) -> str:
    sequence = "".join(sequence.split()).upper().removesuffix("*")
    if not sequence or not sequence.isascii() or not sequence.isalpha():
        raise ValueError("expected a nonempty amino acid sequence with ASCII letters")
    return sequence


def model_input(sequence: str) -> str:
    return "".join(aa if aa in CANONICAL_RESIDUES else "X" for aa in sequence[:MAX_RESIDUES])


def read_fasta(path: Path) -> Iterator[Protein]:
    identifier = None
    fragments = []
    seen = set()
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if identifier is not None:
                    yield Protein(identifier, normalize_sequence("".join(fragments)))
                header = line[1:].split()
                if not header or header[0] in seen:
                    raise ValueError(f"empty or duplicate FASTA identifier at line {line_number}")
                identifier = header[0]
                seen.add(identifier)
                fragments = []
            elif identifier is None:
                raise ValueError(f"sequence before FASTA header at line {line_number}")
            else:
                fragments.append(line)
    if identifier is None:
        raise ValueError("FASTA contains no records")
    yield Protein(identifier, normalize_sequence("".join(fragments)))


def read_training(path: Path) -> tuple[list[Example], int]:
    examples = []
    indices_by_input: dict[str, set[int]] = defaultdict(set)
    with path.open(encoding="utf-8-sig") as handle:
        lines = [line.strip() for line in handle if line.strip()]
    for header, sequence, tags in zip(lines[::3], lines[1::3], lines[2::3], strict=True):
        if not header.startswith(">"):
            raise ValueError(f"expected a three-line training record, got {header[:30]!r}")
        identifier, kingdom, category, fold = header[1:].split("|")
        label = CLASSES.index("OTHER" if category == "NO_SP" else category)
        sequence = normalize_sequence(sequence)
        if int(fold) not in (0, 1, 2) or kingdom not in {"EUKARYA", "ARCHAEA", "POSITIVE", "NEGATIVE"}:
            raise ValueError(f"unknown fold or organism group in {header!r}")
        if len(tags) != len(sequence) or (kingdom == "EUKARYA" and label > 1):
            raise ValueError(f"inconsistent training record {header!r}")
        cut = len(tags) - len(tags.lstrip(SIGNAL_TAGS[label])) if label else 0
        if (label and not 0 < cut < min(len(sequence), MAX_RESIDUES)) or set(tags[cut:]) - set("IMO"):
            raise ValueError(f"invalid signal peptide or mature labels in {header!r}")
        indices_by_input[model_input(sequence)].add(int(fold))
        examples.append(Example(Protein(identifier, sequence), label, kingdom, int(fold), tags, cut))
    filtered = [item for item in examples if len(indices_by_input[model_input(item.protein.sequence)]) == 1]
    return filtered, len(examples) - len(filtered)
