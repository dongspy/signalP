import itertools
import unittest

import torch

from sp_repro.crf import STATE_COUNT, SignalCRF, weak_labels
from sp_repro.data import CLASSES, MAX_RESIDUES, Example, Protein


class ConditionalRandomFieldTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_partition_viterbi_and_cleavage_posterior_against_enumeration(self):
        model = SignalCRF(1)
        generator = torch.Generator().manual_seed(17)
        with torch.no_grad():
            for parameter in (model.start, model.end, model.trans):
                parameter.copy_(torch.randn(parameter.shape, generator=generator))
        emissions = torch.randn((2, 6, len(CLASSES), STATE_COUNT), generator=generator)
        lengths = torch.tensor([5, 6])
        labels = torch.tensor([1, 3])
        start, end, transitions = model.transitions()
        results = model.log_partitions(emissions, lengths, torch.zeros(2, dtype=torch.bool))
        paths, cuts, probabilities, marginals = model.decode(emissions, lengths, labels, confidence=True, per_residue=True)

        for i, length in enumerate(lengths.tolist()):
            label = labels[i].item()
            paths_and_scores = []
            for path in itertools.product(range(STATE_COUNT), repeat=length):
                edges = tuple(zip(path, path[1:]))
                if not (model.start_allowed[label, path[0]] and model.end_allowed[label, path[-1]]):
                    continue
                if not all(model.transition_allowed[label, a, b] for a, b in edges):
                    continue
                score = start[label, path[0]] + end[label, path[-1]] + sum(transitions[label, a, b] for a, b in edges)
                score += sum(emissions[i, j, label, state] for j, state in enumerate(path))
                paths_and_scores.append((path, score))

            scores = torch.stack([score for _, score in paths_and_scores])
            best = paths_and_scores[scores.argmax()][0]
            mature = 3 if label == 1 else 4
            cut = sum(state < mature for state in best)
            hits = [sum(state < mature for state in path) == cut for path, _ in paths_and_scores]
            self.assertAlmostEqual(results[i, label].item(), scores.logsumexp(0).item(), places=5)
            self.assertEqual(paths[i, :length].tolist(), list(best))
            self.assertEqual(cuts[i].item(), cut)
            self.assertAlmostEqual(probabilities[i].item(), scores.softmax(0)[hits].sum().item(), places=5)
            for position, row in enumerate(marginals[i, :length]):
                expected = sum(weight for weight, (path, _) in zip(scores.softmax(0), paths_and_scores) if path[position] == best[position])
                self.assertAlmostEqual(row[best[position]].item(), expected.item(), places=5)

    def test_partial_labels_produce_finite_loss_and_gradients(self):
        model = SignalCRF(2)
        positive = Example(Protein("control_a", "MKTLLAAVAAAAAKQVA"), 1, "EUKARYA", 0, "S" * 11 + "O" * 6, 11)
        negative = Example(Protein("control_b", "MKLLAGVTAAKES"), 0, "NEGATIVE", 0, "I" * 5 + "M" * 4 + "O" * 4, 0)
        examples = [positive, negative]
        loss = model.loss(
            torch.zeros((2, MAX_RESIDUES, 2)), torch.tensor([len(item.tags) for item in examples]),
            torch.tensor([True, False]), torch.tensor([1, 0]),
            torch.stack([weak_labels(item) for item in examples]), torch.ones(len(CLASSES)),
        )
        self.assertTrue(0 < loss.item() < 100)
        loss.backward()
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
