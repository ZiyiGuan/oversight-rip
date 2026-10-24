"""Certificate and artifact boundary regressions; no torch/numpy needed."""

import itertools
import json
import tempfile
import unittest
from pathlib import Path

from survey_sparse import (
    JUDGE_CHUNK, SELECTED_IMAGE_IDS, VIEW_COUNTS, atomic_json, canonical_masks, canonical_table,
    certificate, checked_int, judge_table, merge_shards, parse_image_ids, selection_bundle,
    tool_source_hashes, validate_idx, validate_masks, write_bundle,
)


class CertificateTests(unittest.TestCase):
    def test_exact_distance_all_tiny_boolean_oracles(self):
        # For every 5-vote table over three classes, independently enumerate
        # all Boolean answer corruptions for each of the ten candidate labels.
        for table in itertools.product(range(3), repeat=5):
            result = certificate(list(table), 0, 5, 3, 1)
            for label in range(10):
                bits = [int(value == label) for value in table]
                original = sum(bits) >= 3
                distances = []
                for altered in itertools.product((0, 1), repeat=5):
                    if (sum(altered) >= 3) != original:
                        distances.append(sum(a != b for a, b in zip(bits, altered)))
                self.assertEqual(result["flip_distances"][label], min(distances))
            self.assertEqual(result["min_flip_distance"], min(result["flip_distances"]))
            self.assertEqual(result["certified_flips"], result["min_flip_distance"] - 1)

    def test_robust_incorrect_and_abstaining_are_not_correct(self):
        wrong = certificate([7] * 8, 2, 8, 5, 1)
        self.assertTrue(wrong["strong_whole_circuit"])
        self.assertFalse(wrong["strong_correct"])
        abstain = certificate(list(range(8)), 2, 8, 5, 1)
        self.assertEqual(abstain["accepted_label"], -1)
        self.assertTrue(abstain["positive_whole_circuit"])
        self.assertFalse(abstain["correctly_accepted"])

    def test_threshold_boundary_has_zero_radius(self):
        accepted = certificate([1] * 5 + [2] * 3, 1, 8, 5, 1)
        rejected = certificate([1] * 4 + [2] * 4, 1, 8, 5, 1)
        self.assertEqual(accepted["certified_flips"], 0)
        self.assertEqual(rejected["certified_flips"], 0)

    def test_malformed_tables_and_parameters_rejected(self):
        for table in ([0] * 7, [0] * 9, [0] * 7 + [10], [0] * 7 + [-1], [0] * 7 + [True]):
            with self.assertRaises(ValueError):
                certificate(table, 0, 8, 5, 1)
        for label in (-1, 10, True, "0"):
            with self.assertRaises(ValueError):
                certificate([0] * 8, label, 8, 5, 1)
        with self.assertRaises(ValueError):
            certificate([0] * 8, 0, 8, 4, 1)
        with self.assertRaises(ValueError):
            checked_int(True, "count", 1, 10)


class ArtifactTests(unittest.TestCase):
    def test_canonical_original_text_encodings(self):
        self.assertEqual(canonical_table([7, 0, 9]), b"709\n")
        self.assertEqual(canonical_masks([[1, 2, 3, 4, 5, 6]]), b"1 2 3 4 5 6\n")

    def test_mask_capacity_and_schema_checks(self):
        image = [1] * 784
        valid = [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 6]]
        validate_masks(valid, image, 2)
        cases = [valid[:1], [valid[0], valid[0]], [[0, 1, 2, 3, 4, 784], valid[1]],
                 [[0, 1, 2, 3, 4, 4], valid[1]], [[1, 0, 2, 3, 4, 5], valid[1]]]
        for masks in cases:
            with self.assertRaises(ValueError):
                validate_masks(masks, image, 2)
        image[0] = 0
        with self.assertRaises(ValueError):
            validate_masks(valid, image, 2)
        with self.assertRaises(ValueError):
            validate_masks(valid, [1] * 6 + [0] * 778, 2)

    def test_idx_truncation_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_idx(b"", b"")

    def test_selection_rejects_duplicate_missing_or_ineligible_ids(self):
        criterion = {"minimum_certified_flips": 1, "image_ids": [1, 2]}
        config = {"r": 8, "threshold": 5, "criterion": criterion, "sources": {}}
        good = [{"image_id": 1, "correctly_accepted": True, "certified_flips": 1},
                {"image_id": 2, "correctly_accepted": True, "certified_flips": 3}]
        self.assertEqual(selection_bundle(config, good)["selected_count"], 2)
        self.assertEqual(selection_bundle(config, good)["criterion"], criterion)
        for rows in ([good[0]] * 2, [good[1], good[0]], [good[0]],
                     [good[0], dict(good[1], correctly_accepted=False)],
                     [good[0], dict(good[1], certified_flips=0)],
                     [good[0], dict(good[1], image_id=3)]):
            with self.assertRaises(ValueError):
                selection_bundle(config, rows)

    def test_fixed_identifiers_are_the_benchmark_instances(self):
        self.assertEqual(len(SELECTED_IMAGE_IDS), 50)
        self.assertEqual(list(SELECTED_IMAGE_IDS), sorted(set(SELECTED_IMAGE_IDS)))
        self.assertEqual(SELECTED_IMAGE_IDS[0], 0)
        self.assertEqual(SELECTED_IMAGE_IDS[-1], 96)
        self.assertEqual(parse_image_ids(",".join(map(str, SELECTED_IMAGE_IDS))), list(SELECTED_IMAGE_IDS))
        self.assertEqual(parse_image_ids("3, 5,9"), [3, 5, 9])
        for text in ("", "5,3", "3,3", "-1", "10000", "a"):
            with self.assertRaises(ValueError):
                parse_image_ids(text)
        # The seven bundled sizes and the three large sizes generated by the split pipeline
        # (scripts/generate_large_bundle.py); the certificate loop stays on the bundled sizes, the large ones are
        # covered by the pipeline's own R = 256 identity test and would need 512 MB lists here.
        self.assertEqual(VIEW_COUNTS, (256, 1024, 4096, 16384, 65536, 262144, 1048576, 1 << 22, 1 << 24, 1 << 26))
        for r in VIEW_COUNTS[:7]:
            self.assertEqual(certificate([0] * r, 0, r, r // 2 + 1, r // 8)["certified_flips"], r // 2 - 1)
        above = max(VIEW_COUNTS) + 1
        with self.assertRaises(ValueError):
            certificate([0] * 8, 0, above, above // 2 + 1, 1)

    def test_judge_table_is_one_call_cut_at_whole_batches(self):
        # prepare_sparse.judge_predictions batches by 256 from the start of its
        # list, so slices that are whole batches leave every batch unchanged;
        # and every R the committed bundles use is still a single call.
        self.assertEqual(JUDGE_CHUNK % 256, 0)
        self.assertGreaterEqual(JUDGE_CHUNK, 65536)
        calls = []

        def predict(part):
            calls.append((part[0][0], len(part)))
            return [mask[0] % 10 for mask in part]

        masks = [[i] for i in range(2 * JUDGE_CHUNK + 300)]
        self.assertEqual(judge_table(predict, masks), [i % 10 for i in range(len(masks))])
        self.assertEqual(calls, [(0, JUDGE_CHUNK), (JUDGE_CHUNK, JUDGE_CHUNK), (2 * JUDGE_CHUNK, 300)])
        calls.clear()
        judge_table(predict, masks[:65536])
        self.assertEqual(calls, [(0, 65536)])

    def test_atomic_json_writes_complete_readable_document(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selection.json"
            atomic_json(path, {"images": [1, 2]})
            self.assertEqual(json.loads(path.read_text()), {"images": [1, 2]})
            self.assertFalse(path.with_suffix(".json.tmp").exists())


def survey_run(directory, ids, threads=4, elapsed=1.0, tools=None):
    """A completed R = 8 survey run over `ids`, written by the script's own
    writers exactly as main() writes one."""
    r = 8
    config = {"schema": "mnist_sparse_selected_config_v1", "r": r, "threshold": r // 2 + 1,
              "threads": threads,
              "criterion": {"minimum_certified_flips": 1, "require_correct": True,
                            "ordering": "ascending_test_image_id", "selection": "fixed_image_ids",
                            "image_ids": list(ids), "strong_radius": r // 8},
              "sources": {"test_images_sha256": "images", "test_labels_sha256": "labels",
                          "tool_source_sha256": tools or tool_source_hashes()}}
    rows = []
    for image_id in ids:
        table = [image_id % 10] * 6 + [(image_id + 1) % 10] * 2
        row = certificate(table, image_id % 10, r, r // 2 + 1, r // 8)
        row.update({"image_id": image_id, "table": canonical_table(table).decode("ascii").rstrip("\n"),
                    "table_sha256": "t", "mask_registry_sha256": "m", "image_sha256": "p",
                    "canonical_replay_checked": True, "masks": [[image_id, 101, 202, 303, 404, 505]],
                    "image_pixels": [image_id, 0, 255]})
        rows.append(row)
    environment = {"python": "3", "numpy": "n", "torch": "t", "device": "cpu", "threads": threads,
                   "deterministic_algorithms": True}
    directory.mkdir()
    atomic_json(directory / "config.json", config)
    write_bundle(directory, config, rows, [], elapsed, environment)
    return directory


class ShardMergeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def empty(self, name):
        path = self.root / name
        path.mkdir()
        return path

    def test_merge_writes_the_single_run_bytes(self):
        whole = survey_run(self.root / "whole", [1, 2, 4, 7], elapsed=9.0)
        shards = [survey_run(self.root / "b", [2, 7], elapsed=2.5),
                  survey_run(self.root / "a", [1, 4], elapsed=3.0)]
        merged = self.empty("merged")
        manifest = merge_shards(shards, 8, [1, 2, 4, 7], merged)
        for name in ("selected.json", "selected.csv", "config.json"):
            self.assertEqual((merged / name).read_bytes(), (whole / name).read_bytes(), name)
        single = json.loads((whole / "manifest.json").read_text())
        self.assertNotIn("shards", single)
        self.assertEqual(json.loads((merged / "manifest.json").read_text()), manifest)
        self.assertEqual(manifest["elapsed_seconds"], 5.5)
        self.assertEqual([shard["image_ids"] for shard in manifest["shards"]], [[1, 4], [2, 7]])
        self.assertEqual({k: v for k, v in manifest.items() if k not in ("elapsed_seconds", "shards")},
                         {k: v for k, v in single.items() if k != "elapsed_seconds"})

    def test_merge_refuses_before_writing_anything(self):
        a = survey_run(self.root / "a", [1, 4])
        b = survey_run(self.root / "b", [2])
        foreign = {name: "0" * 64 for name in tool_source_hashes()}
        failed = self.empty("failed")
        (failed / "failures.json").write_text("{}\n")
        cases = {
            "an image in no shard": ([a], 8, [1, 2, 4]),
            "an image in two shards": ([a, b, survey_run(self.root / "c", [4])], 8, [1, 2, 4]),
            "a shard outside the requested ids": ([a, b], 8, [1, 2]),
            "different threads": ([a, survey_run(self.root / "d", [2], threads=1)], 8, [1, 2, 4]),
            "a different R": ([a, b], 16, [1, 2, 4]),
            "other tool sources": ([survey_run(self.root / "e", [1, 4], tools=foreign),
                                    survey_run(self.root / "f", [2], tools=foreign)], 8, [1, 2, 4]),
            "a failed shard": ([a, failed], 8, [1, 2, 4]),
        }
        for n, (label, (dirs, r, ids)) in enumerate(cases.items()):
            output = self.empty(f"out{n}")
            with self.assertRaises(ValueError, msg=label):
                merge_shards(dirs, r, ids, output)
            self.assertEqual(list(output.iterdir()), [], label)
        (b / "selected.csv").write_text("tampered\n")
        output = self.empty("tampered")
        with self.assertRaises(ValueError):
            merge_shards([a, b], 8, [1, 2, 4], output)
        self.assertEqual(list(output.iterdir()), [])




if __name__ == "__main__":
    unittest.main()
