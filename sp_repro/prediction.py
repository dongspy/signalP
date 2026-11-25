import csv
import json
import textwrap
from itertools import groupby, islice
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import quote

import torch

from . import __version__
from .crf import STATES, SignalCRF
from .data import CLASSES, CLASS_TYPES, MAX_RESIDUES, Protein, read_fasta
from .encoder import FrozenEncoder, device_for, sha256


@torch.no_grad()
def predict_batch(proteins, encoder, model, eukaryotic, per_residue):
    features, lengths = encoder.encode(proteins)
    eukarya = torch.full((len(proteins),), eukaryotic, dtype=torch.bool)
    scores = model.emission_scores(features, eukarya)
    probabilities = model.log_partitions(scores, lengths, eukarya).softmax(dim=-1)
    labels = probabilities.argmax(dim=-1)
    paths, cuts, cut_probabilities, marginals = model.decode(
        scores, lengths, labels, confidence=True, per_residue=per_residue
    )
    for i, protein in enumerate(proteins):
        label = labels[i].item()
        cut = cuts[i].item()
        length = lengths[i].item()
        states = STATES[label]
        path = [states[state] for state in paths[i, :length].tolist()]
        position = 1
        regions = []
        for name, residues in groupby(path[:cut]):
            end = position + len(list(residues)) - 1
            regions.append({"name": name, "start": position, "end": end})
            position = end + 1
        result = {
            "id": protein.identifier,
            "sequence_length": len(protein.sequence),
            "analyzed_residues": length,
            "prediction": CLASSES[label],
            "signal_type": CLASS_TYPES[label],
            "probabilities": dict(zip(CLASSES, probabilities[i].tolist(), strict=True)),
            "cleavage": None,
            "regions": regions,
        }
        if cut:
            result["cleavage"] = {"after": cut, "before": cut + 1, "conditional_probability": cut_probabilities[i].item()}
            if label in (2, 4) and protein.sequence[cut] == "C":
                result["lipobox_cysteine_position"] = cut + 1
        if marginals is not None:
            result["per_residue"] = {
                "conditional_on": CLASSES[label], "states": states, "path": path,
                "probabilities": marginals[i, :length, :len(states)].tolist(),
            }
        yield protein, result


def write_gff(handle, protein, start, end, attributes, kind="signal_peptide"):
    identifier = quote(protein.identifier, safe="-_.~:")
    handle.write(f"{identifier}\tsp-repro\t{kind}\t{start}\t{end}\t.\t.\t.\t{attributes}\n")


def export_predictions(path: Path, records, metadata: dict) -> int:
    path.mkdir()
    with (
        (path / "prediction_results.txt").open("w", encoding="utf-8", newline="") as table,
        (path / "output.gff3").open("w", encoding="utf-8") as gff,
        (path / "region_output.gff3").open("w", encoding="utf-8") as region_gff,
        (path / "processed_entries.fasta").open("w", encoding="utf-8") as fasta,
        (path / "output.json").open("w", encoding="utf-8") as results,
    ):
        writer = csv.writer(table, delimiter="\t", lineterminator="\n")
        writer.writerow(["# ID", "Prediction", *CLASSES, "CS Position"])
        gff.write("##gff-version 3\n")
        region_gff.write("##gff-version 3\n")
        results.write('{"metadata": ' + json.dumps(metadata, ensure_ascii=False) + ',\n"predictions": [')
        count = 0
        for protein, result in records:
            cut = result["cleavage"]
            cs = f"CS pos: {cut['after']}-{cut['before']}. Pr: {cut['conditional_probability']:.6f}" if cut else ""
            writer.writerow([protein.identifier, result["prediction"], *(f"{p:.6f}" for p in result["probabilities"].values()), cs])
            if cut:
                feature_id = f"signal_{count + 1}"
                write_gff(gff, protein, 1, cut["after"], f"ID={feature_id};Name={quote(result['signal_type'], safe='')}")
                for region in result["regions"]:
                    attributes = f"Name={region['name']};signal_peptide_id={feature_id}"
                    write_gff(region_gff, protein, region["start"], region["end"], attributes, "region")
                fasta.write(f">{protein.identifier}\n" + "\n".join(textwrap.wrap(protein.sequence[cut["after"]:], 70)) + "\n")
            separator = ",\n" if count else "\n"
            results.write(separator + json.dumps(result, ensure_ascii=False, allow_nan=False))
            count += 1
        results.write("\n]}\n")
    return count


def predict(args) -> None:
    if args.out.exists():
        raise ValueError(f"choose a new output directory: {args.out}")
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != 1:
        raise ValueError("checkpoint schema version is unsupported")
    encoder = FrozenEncoder(args.encoder, device_for(args.device))
    if encoder.fingerprint != checkpoint["identity"]["encoder_sha256"]:
        raise ValueError("encoder files do not match the checkpoint fingerprints")
    model = SignalCRF(checkpoint["encoder_width"])
    model.load_state_dict(checkpoint["state"])
    model.eval()
    inputs = read_fasta(args.fasta)

    def records():
        while batch := list(islice(inputs, args.batch_size)):
            yield from predict_batch(batch, encoder, model, args.organism == "eukarya", args.per_residue)

    metadata = {
        "software_version": __version__, "organism": args.organism,
        "checkpoint_sha256": sha256(args.model), "encoder_sha256": encoder.fingerprint,
        "max_analyzed_residues": MAX_RESIDUES, "coordinate_base": 1,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".sp-repro-", dir=args.out.parent) as staging:
        output = Path(staging) / "output"
        count = export_predictions(output, records(), metadata)
        output.rename(args.out)
    print(f"Predicted {count} proteins; results: {args.out}")
