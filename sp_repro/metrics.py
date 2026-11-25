import math

from .data import CLASSES, CLASS_TYPES, Example


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def mcc(tp, fp, tn, fn):
    return ratio(tp * tn - fp * fn, math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))


def report(examples: list[Example], predictions: list[int], cuts: list[int]) -> dict:
    rows = list(zip(examples, predictions, cuts, strict=True))
    matrix = [[0] * len(CLASSES) for _ in CLASSES]
    for item, predicted, _ in rows:
        matrix[item.label][predicted] += 1

    classes = {}
    for index, category in enumerate(CLASS_TYPES):
        actual = sum(matrix[index])
        predicted = sum(row[index] for row in matrix)
        tp = matrix[index][index]
        fp = predicted - tp
        fn = actual - tp
        negatives = (0,) if index else range(1, len(CLASSES))
        fp1 = sum(matrix[negative][index] for negative in negatives)
        tn1 = sum(sum(matrix[negative]) for negative in negatives) - fp1
        descriptor = {
            "n": actual,
            "predicted": predicted,
            "precision": ratio(tp, predicted),
            "recall": ratio(tp, actual),
            "mcc1": mcc(tp, fp1, tn1, fn),
            "mcc2": mcc(tp, fp, len(rows) - tp - fp - fn, fn),
        }
        if index:
            for tolerance in (0, 1, 2, 3):
                hits = sum(item.label == estimate == index and abs(item.cut - cut) <= tolerance for item, estimate, cut in rows)
                descriptor[f"cleavage_{tolerance}"] = {
                    "hits": hits,
                    "precision": ratio(hits, predicted),
                    "recall": ratio(hits, actual),
                }
        classes[category] = descriptor

    hard_negatives = [(item, predicted) for item, predicted, _ in rows if item.label == 0 and "M" in item.tags[:40]]
    positive_mcc = [item["mcc2"] or 0.0 for label, item in classes.items() if label != "OTHER" and item["n"]]
    return {
        "n": len(rows),
        "accuracy": ratio(sum(matrix[index][index] for index in range(len(CLASSES))), len(rows)),
        "macro_mcc2_signal": ratio(sum(positive_mcc), len(positive_mcc)),
        "confusion_matrix_class_order": CLASS_TYPES,
        "confusion_matrix": matrix,
        "classes": classes,
        "n_terminal_tm_negatives": {
            "n": len(hard_negatives),
            "false_positives": sum(predicted != 0 for _, predicted in hard_negatives),
        },
    }
