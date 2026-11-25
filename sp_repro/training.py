import json
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch

from .crf import STATE_COUNT, SignalCRF, weak_labels
from .data import CLASSES, MAX_RESIDUES, Example, read_training
from .encoder import FrozenEncoder, device_for, sha256
from .metrics import report


def write_checkpoint(payload: dict, path: Path) -> None:
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_features(
    examples: list[Example], encoder: FrozenEncoder, batch_size: int,
    cache: Path | None, identity: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cache is not None and cache.exists():
        payload = torch.load(cache, weights_only=True, mmap=True)
        if payload["identity"] == identity:
            return payload["features"], payload["lengths"]

    features = torch.empty((len(examples), MAX_RESIDUES, encoder.width), dtype=torch.float16)
    lengths = torch.empty(len(examples), dtype=torch.long)
    for start in range(0, len(examples), batch_size):
        batch = [item.protein for item in examples[start:start + batch_size]]
        encoded, count = encoder.encode(batch)
        features[start:start + len(batch)] = encoded
        lengths[start:start + len(batch)] = count
        if start % (batch_size * 20) == 0 or start + len(batch) == len(examples):
            print(f"Encoded {start + len(batch)}/{len(examples)} sequences on {encoder.device}", flush=True)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        write_checkpoint({"identity": identity, "features": features, "lengths": lengths}, cache)
    return features, lengths


@torch.no_grad()
def evaluate(model, features, lengths, eukarya, examples, indices, batch_size, device, by_kingdom=False):
    model.eval()
    labels = []
    cuts = []
    for batch in indices.split(batch_size):
        count = lengths[batch].to(device)
        group = eukarya[batch].to(device)
        scores = model.emission_scores(features[batch].to(device), group)
        predicted = model.log_partitions(scores, count, group).argmax(dim=-1)
        _, cut, _, _ = model.decode(scores, count, predicted, confidence=False)
        labels.extend(predicted.tolist())
        cuts.extend(cut.tolist())
    selected = [examples[index] for index in indices.tolist()]
    metrics = report(selected, labels, cuts)
    if by_kingdom:
        metrics["by_kingdom"] = {}
        for kingdom in sorted({item.kingdom for item in selected}):
            positions = [i for i, item in enumerate(selected) if item.kingdom == kingdom]
            metrics["by_kingdom"][kingdom] = report(
                [selected[i] for i in positions], [labels[i] for i in positions], [cuts[i] for i in positions]
            )
    return metrics


def train(args) -> None:
    if args.validation_fold == args.test_fold:
        raise ValueError("validation and test folds must be different")
    torch.manual_seed(args.seed)
    examples, dropped = read_training(args.data)
    if args.limit_per_class:
        counts = Counter()
        sampled = []
        for item in examples:
            key = item.fold, item.label
            if counts[key] < args.limit_per_class:
                sampled.append(item)
                counts[key] += 1
        examples = sampled

    encoder = FrozenEncoder(args.encoder, device_for(args.encoder_device))
    model_device = device_for(args.crf_device)
    identity = {
        "dataset_sha256": sha256(args.data),
        "encoder_sha256": encoder.fingerprint,
        "max_residues": MAX_RESIDUES,
        "encoding_version": 1,
        "limit_per_class": args.limit_per_class,
        "excluded_input_overlap": dropped,
    }
    features, lengths = load_features(examples, encoder, args.encoder_batch_size, args.cache, identity)
    width = encoder.width
    del encoder
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    labels = torch.tensor([item.label for item in examples])
    eukarya = torch.tensor([item.eukarya for item in examples])
    folds = torch.tensor([item.fold for item in examples])
    valid = torch.where(folds == args.validation_fold)[0]
    test = torch.where(folds == args.test_fold)[0]
    training_fold = ({0, 1, 2} - {args.validation_fold, args.test_fold}).pop()
    training = torch.where(folds == training_fold)[0]
    if not all(len(part) for part in (training, valid, test)):
        raise ValueError("the dataset must contain all three folds")
    allowed = torch.zeros((len(examples), MAX_RESIDUES, STATE_COUNT), dtype=torch.bool)
    for index in training.tolist():
        allowed[index] = weak_labels(examples[index])
    counts = torch.bincount(labels[training], minlength=len(CLASSES)).float()
    weights = (counts.max() / counts.clamp_min(1)).pow(args.class_weight_power)
    weights = (weights / (weights * counts).sum() * counts.sum()).to(model_device)

    model = SignalCRF(width).to(model_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    generator = torch.Generator().manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.out / "model.pt"
    if checkpoint_path.exists() or (args.out / "metrics.json").exists():
        raise ValueError(f"output already contains a training run: {args.out}")
    print(f"Training on {model_device} with folds train={training_fold}, validation={args.validation_fold}, test={args.test_fold}", flush=True)
    best = -float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in training[torch.randperm(len(training), generator=generator)].split(args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(
                features[batch].to(model_device), lengths[batch].to(model_device), eukarya[batch].to(model_device),
                labels[batch].to(model_device), allowed[batch].to(model_device), weights,
            )
            if not torch.isfinite(loss):
                raise ValueError("training produced a nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item() * len(batch)
        validation = evaluate(model, features, lengths, eukarya, examples, valid, args.batch_size, model_device)
        score = validation["macro_mcc2_signal"]
        if score is None:
            raise ValueError("validation fold contains no positive labels")
        history.append({"epoch": epoch, "loss": total_loss / len(training), "validation": validation})
        print(f"Epoch {epoch}/{args.epochs}  loss={total_loss / len(training):.4f}  validation macro MCC2={score:.4f}", flush=True)
        if score > best:
            best = score
            write_checkpoint({
                "schema_version": 1, "encoder_width": width, "identity": identity,
                "seed": args.seed, "train_fold": training_fold, "validation_fold": args.validation_fold,
                "test_fold": args.test_fold, "epoch": epoch, "state": model.state_dict(),
            }, checkpoint_path)

    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=model_device)
    model.load_state_dict(checkpoint["state"])
    test_report = evaluate(model, features, lengths, eukarya, examples, test, args.batch_size, model_device, by_kingdom=True)
    output = {
        "identity": identity,
        "train_fold": training_fold, "validation_fold": args.validation_fold, "test_fold": args.test_fold,
        "seed": args.seed, "best_epoch": checkpoint["epoch"], "history": history, "test": test_report,
        "settings": {
            "torch": str(torch.__version__), "batch_size": args.batch_size,
            "class_weight_power": args.class_weight_power, "learning_rate": args.learning_rate,
        },
    }
    (args.out / "metrics.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Held-out fold macro MCC2={test_report['macro_mcc2_signal']:.4f}; report: {args.out / 'metrics.json'}", flush=True)
