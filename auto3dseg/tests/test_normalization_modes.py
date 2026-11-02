# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import importlib
import sys
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from parameterized import parameterized

from monai.bundle import ConfigParser
from monai.data import MetaTensor
from monai.transforms import Compose

ALGO_TEMPLATES = Path(__file__).resolve().parents[1] / "algorithm_templates"
sys.path.insert(0, str(ALGO_TEMPLATES))

ALGORITHMS = ("segresnet", "segresnet2d")
EXPORT_CASES = (
    ("mri", {}, {}, "meanstd"),
    ("ct", {}, {}, "range"),
    ("mri", {"image2": "mri"}, {}, "meanstd"),
    ("ct", {"image2": "ct"}, {}, "range"),
    ("mri", {"image2": "meanstd"}, {}, "meanstd"),
    ("mri", {"image2": "none"}, {}, ["meanstd", "none"]),
    ("ct", {"image_z": "pet", "image_a": "mri"}, {}, ["range", "pet", "meanstd"]),
    ("mri", {"image2": "none"}, {"normalize_mode": "range"}, ["range", "none"]),
    ("mri", {"image2": "none"}, {"normalize_mode": ["none", "range"]}, ["none", "range"]),
    ("mri", {"image2": "none"}, {"normalize_mode": ["ct", "none"]}, ["range", "none"]),
    ("mri", {"image2": "none"}, {"normalize_mode": ["mri", "none"]}, ["meanstd", "none"]),
    ("mri", {"image2": "none"}, {}, ["range", "none"], {"normalize_mode": ["ct", "none"]}),
    ("ct", {"image2": "none"}, {}, ["range", "none"], None, {"normalize_mode#1": "mri"}, ["range", "meanstd"]),
    ("mri", {"image2": "none"}, {}, ["range", "none"], {"normalize_mode": "range"}),
    ("mri", {"image2": "none"}, {}, ["meanstd", "pet"], {"extra_modalities": {"image2": "pet"}}),
    ("mri", {"image2": "none"}, {}, ["none", "range"], {"normalize_mode": ["none", "range"]}),
    ("mri", {"image2": "none"}, {}, ["meanstd", "none"], None, {"extra_modalities": {"image2": "mri"}}, "meanstd"),
    (
        "mri",
        {"image2": "none"},
        {},
        ["meanstd", "none"],
        None,
        {"extra_modalities": {}, "input_channels": 1},
        "meanstd",
    ),
    (
        "mri",
        {"image2": "none"},
        {},
        ["meanstd", "none"],
        None,
        {"extra_modalities": {"image2": "none", "image3": "ct"}, "input_channels": 3},
        ["meanstd", "none", "range"],
    ),
    (
        "mri",
        {"image2": "none"},
        {},
        ["meanstd", "none"],
        None,
        {"extra_modalities": {"image2": "mri"}, "normalize_mode": ["none", "range"]},
        ["none", "range"],
    ),
    (
        "mri",
        {"image2": "mri"},
        {"normalize_mode": ["meanstd", "none"]},
        ["meanstd", "none"],
        None,
        {"extra_modalities": {"image2": "mri"}},
        ["meanstd", "none"],
    ),
    ("mri", None, {}, ["meanstd", "none"], {"extra_modalities": {"image2": "none"}, "input_channels": 2}),
    ("mri", None, {}, ["meanstd", "none"], {"extra_modalities": {"image2": "none"}, "input_channels": 3}),
    (
        "mri",
        None,
        {},
        "meanstd",
        None,
        {"extra_modalities": {"image2": "none"}, "input_channels": 2},
        ["meanstd", "none"],
    ),
    ("mri", {"image2": "ct"}, {}, ["meanstd", "range"], None, {"normalize_mode#1": "none"}, ["meanstd", "none"]),
    (
        "mri",
        {"image2": "none"},
        {},
        ["meanstd", "none"],
        {"normalize_mode": ["meanstd", "none"]},
        {"extra_modalities": {"image2": "mri"}},
        ["meanstd", "none"],
    ),
    (
        "mri",
        {"image_z": "none", "image_a": "ct"},
        {},
        ["meanstd", "none", "range"],
        None,
        {"extra_modalities": {"image_a": "ct", "image_z": "none"}},
        ["meanstd", "range", "none"],
    ),
    (
        "mri",
        {"image_z": "none", "image_a": "ct"},
        {"normalize_mode": ["none", "none", "range"]},
        ["none", "none", "range"],
        None,
        {"extra_modalities": {"image_a": "ct", "image_z": "none"}, "normalize_mode": "meanstd"},
        ["meanstd", "range", "none"],
    ),
)


class TestNormalizationModes(unittest.TestCase):
    @parameterized.expand([(algo, fresh_stats) for algo in ALGORITHMS for fresh_stats in (False, True)])
    def test_rejected_mode_override_preserves_refill_state(self, algo_name, fresh_stats):
        """A rejected export must not change modes used by a later cached export."""
        algo_class = getattr(importlib.import_module(f"{algo_name}.scripts.algo"), f"{algo_name.capitalize()}Algo")
        algo = algo_class(template_path=str(ALGO_TEMPLATES))
        algo.name = algo_name
        modes = ["none", "none"] if fresh_stats else ["meanstd", "none"]
        algo.fill_records = {
            "hyper_parameters.yaml": {
                "normalize_mode": modes,
                "extra_modalities": {"image2": "none"},
                "input_channels": 2,
            }
        }
        algo._normalize_mode_is_derived = not fresh_stats
        previous_records = copy.deepcopy(algo.fill_records)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = {
                "modality": "mri",
                "extra_modalities": {"image2": "none"},
                "dataroot": temp_dir,
                "datalist": str(root / "datalist.json"),
            }
            stats = {
                "stats_summary": {
                    "n_cases": 2,
                    "image_stats": {
                        "channels": {"max": 1},
                        "spacing": {
                            "median": [1, 1, 1],
                            "percentile_10_0": [1, 1, 1],
                            "percentile_00_5": [1, 1, 1],
                            "percentile_99_5": [1, 1, 1],
                        },
                        "sizemm": {"median": [32, 32, 32], "percentile_90_0": [32, 32, 32]},
                    },
                    "image_foreground_stats": {"intensity": {"percentile_00_5": -2, "percentile_99_5": 2}},
                    "label_stats": {"labels": [0, 1]},
                }
            }
            source_file, stats_file = root / "input.yaml", root / "datastats.yaml"
            ConfigParser.export_config_file(source, str(source_file), sort_keys=False)
            ConfigParser.export_config_file(stats, str(stats_file))
            algo.set_data_source(str(source_file))
            algo.set_data_stats(str(stats_file))
            with self.assertRaisesRegex(ValueError, "normalize_mode.*input"):
                algo.export_to_disk(temp_dir, "rejected", fill_with_datastats=fresh_stats, normalize_mode=["ct"])
            self.assertEqual(algo.fill_records, previous_records)
            # Persist and restore after rejection to detect a corrupted provenance flag too.
            state = copy.deepcopy(algo.state_dict())
            algo = algo_class()
            algo.load_state_dict(state)
            overrides = {"extra_modalities": {"image2": "mri"}} if fresh_stats else {}
            algo.export_to_disk(temp_dir, "recovered", fill_with_datastats=False, **overrides)
            config = ConfigParser.load_config_file(str(root / "recovered/configs/hyper_parameters.yaml"))
        self.assertEqual(config["normalize_mode"], modes)
        self.assertEqual(algo.fill_records["hyper_parameters.yaml"]["normalize_mode"], modes)

    @parameterized.expand([(algo, indexed) for algo in ALGORITHMS for indexed in (False, True)])
    def test_reordered_inputs_require_complete_explicit_modes(self, algo_name, indexed):
        """Reordering keys must not silently assign explicit modes to different images."""
        algo_class = getattr(importlib.import_module(f"{algo_name}.scripts.algo"), f"{algo_name.capitalize()}Algo")
        algo = algo_class(template_path=str(ALGO_TEMPLATES))
        algo.name = algo_name
        algo.fill_records = {
            "hyper_parameters.yaml": {
                "normalize_mode": ["meanstd", "none", "range"],
                "extra_modalities": {"image_z": "none", "image_a": "ct"},
                "input_channels": 3,
            }
        }
        # An indexed override makes even a previously derived list explicit.
        algo._normalize_mode_is_derived = indexed
        state = copy.deepcopy(algo.state_dict())
        algo = algo_class()
        algo.load_state_dict(state)
        reordered = {"image_a": "ct", "image_z": "none"}
        overrides = {"normalize_mode#1": "none"} if indexed else {}
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "normalize_mode.*input"):
                algo.export_to_disk(
                    temp_dir, "bundle", fill_with_datastats=False, extra_modalities=reordered, **overrides
                )
            self.assertEqual(
                list(algo.fill_records["hyper_parameters.yaml"]["extra_modalities"]), ["image_z", "image_a"]
            )
            algo.export_to_disk(
                temp_dir,
                "bundle",
                fill_with_datastats=False,
                extra_modalities=reordered,
                normalize_mode=["meanstd", "range", "none"],
            )
            config = ConfigParser.load_config_file(str(Path(temp_dir) / "bundle/configs/hyper_parameters.yaml"))
        self.assertEqual(list(config["extra_modalities"]), ["image_a", "image_z"])
        self.assertEqual(config["normalize_mode"], ["meanstd", "range", "none"])
        self.assertEqual(algo.fill_records["hyper_parameters.yaml"]["normalize_mode"], config["normalize_mode"])
        self.assertEqual(config["input_channels"], 3)
        builder_class = importlib.import_module(f"{algo_name}.scripts.segmenter").DataTransformBuilder
        builder = builder_class(
            roi_size=[2, 2, 2],
            normalize_mode=config["normalize_mode"],
            extra_modalities=config["extra_modalities"],
            normalize_params={"intensity_bounds": [0, 10]},
        )
        label_input = MetaTensor(torch.full((1, 2, 2), 7.0))
        result = Compose(builder.get_normalize_transforms())(
            {
                "image": MetaTensor(torch.ones(1, 2, 2)),
                "image_a": MetaTensor(torch.ones(1, 2, 2)),
                "image_z": label_input,
            }
        )
        torch.testing.assert_close(result["image"][2], label_input[0])

    @parameterized.expand(
        [
            (algo, extra, count, modes)
            for algo in ALGORITHMS
            for extra, count, modes in (
                ({}, 1, ["meanstd"]),
                ({"image2": "none", "image3": "ct"}, 3, ["meanstd", "none", "range"]),
            )
        ]
    )
    def test_changed_inputs_require_matching_explicit_modes(self, algo_name, extra, count, modes):
        """Reject a stale explicit list at export, before an invalid bundle reaches preprocessing."""
        algo_class = getattr(importlib.import_module(f"{algo_name}.scripts.algo"), f"{algo_name.capitalize()}Algo")
        algo = algo_class(template_path=str(ALGO_TEMPLATES))
        algo.name = algo_name
        algo.fill_records = {
            "hyper_parameters.yaml": {
                "normalize_mode": ["meanstd", "none"],
                "extra_modalities": {"image2": "none"},
                "input_channels": 2,
            }
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "normalize_mode.*input"):
                algo.export_to_disk(
                    temp_dir, "bundle", fill_with_datastats=False, extra_modalities=extra, input_channels=count
                )
            algo.export_to_disk(
                temp_dir,
                "bundle",
                fill_with_datastats=False,
                extra_modalities=extra,
                input_channels=count,
                normalize_mode=modes,
            )
            config = ConfigParser.load_config_file(str(Path(temp_dir) / "bundle/configs/hyper_parameters.yaml"))
        self.assertEqual(config["normalize_mode"], modes)
        self.assertEqual(algo.fill_records["hyper_parameters.yaml"]["normalize_mode"], modes)

    @parameterized.expand(
        [
            (algo, previous, updated, old_channels, new_channels)
            for algo in ALGORITHMS
            for previous, updated, old_channels, new_channels in (
                ({}, {"image2": "none"}, 1, 2),
                ({"image2": "none"}, {}, 3, 1),
                ({"image2": "none"}, {"image_a": "none", "image_b": "none"}, 3, 3),
                ({"image2": "none"}, {"image_a": "none"}, 3, 2),
            )
        ]
    )
    def test_changed_input_keys_require_explicit_channel_count(
        self, algo_name, previous, updated, old_channels, new_channels
    ):
        """Extra images may have multiple channels, so changed keys need an explicit total."""
        algo_class = getattr(importlib.import_module(f"{algo_name}.scripts.algo"), f"{algo_name.capitalize()}Algo")
        algo = algo_class(template_path=str(ALGO_TEMPLATES))
        algo.name = algo_name
        algo.fill_records = {
            "hyper_parameters.yaml": {
                "normalize_mode": "meanstd",
                "extra_modalities": previous,
                "input_channels": old_channels,
            }
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "input_channels"):
                algo.export_to_disk(temp_dir, "bundle", fill_with_datastats=False, extra_modalities=updated)
            algo.export_to_disk(
                temp_dir,
                "bundle",
                fill_with_datastats=False,
                extra_modalities=updated,
                input_channels=new_channels,
                normalize_mode="meanstd",
            )
            config = ConfigParser.load_config_file(str(Path(temp_dir) / "bundle/configs/hyper_parameters.yaml"))
        self.assertEqual(config["input_channels"], new_channels)
        builder_class = importlib.import_module(f"{algo_name}.scripts.segmenter").DataTransformBuilder
        builder = builder_class(roi_size=[2, 2, 2], normalize_mode=config["normalize_mode"], extra_modalities=updated)
        inputs = {key: MetaTensor(torch.ones(1, 2, 2)) for key in ["image", *updated]}
        result = Compose(builder.get_normalize_transforms())(inputs)
        self.assertEqual(result["image"].shape[0], new_channels)

    @parameterized.expand([(algo,) for algo in ALGORITHMS])
    def test_shared_mode_rejects_indexed_override(self, algo_name):
        """A shared scalar has no per-input indices; callers must supply a complete list."""
        algo_class = getattr(importlib.import_module(f"{algo_name}.scripts.algo"), f"{algo_name.capitalize()}Algo")
        algo = algo_class(template_path=str(ALGO_TEMPLATES))
        algo.name = algo_name
        algo.fill_records = {
            "hyper_parameters.yaml": {
                "normalize_mode": "meanstd",
                "extra_modalities": {"image2": "mri"},
                "input_channels": 2,
            }
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "normalize_mode.*list"):
                algo.export_to_disk(temp_dir, "bundle", fill_with_datastats=False, **{"normalize_mode#1": "none"})
            algo.export_to_disk(temp_dir, "bundle", fill_with_datastats=False, normalize_mode=["meanstd", "none"])
            config = ConfigParser.load_config_file(str(Path(temp_dir) / "bundle/configs/hyper_parameters.yaml"))
        self.assertEqual(config["normalize_mode"], ["meanstd", "none"])

    @parameterized.expand([(algo, *case) for algo in ALGORITHMS for case in EXPORT_CASES])
    def test_export_normalization_modes(
        self,
        algo_name,
        modality,
        extra_modalities,
        overrides,
        expected,
        export_kwargs=None,
        refill_kwargs=None,
        refill_expected=None,
    ):
        """Exported YAML must describe every input's normalization in concatenation order."""
        algo_module = importlib.import_module(f"{algo_name}.scripts.algo")
        algo_class = getattr(algo_module, f"{algo_name.capitalize()}Algo")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_source = {
                "modality": modality,
                "dataroot": temp_dir,
                "datalist": str(root / "datalist.json"),
                **overrides,
            }
            if extra_modalities is not None:
                data_source["extra_modalities"] = extra_modalities
            stats = {
                "stats_summary": {
                    "n_cases": 2,
                    "image_stats": {
                        "channels": {"max": 1},
                        "spacing": {
                            "median": [1, 1, 1],
                            "percentile_10_0": [1, 1, 1],
                            "percentile_00_5": [1, 1, 1],
                            "percentile_99_5": [1, 1, 1],
                        },
                        "sizemm": {"median": [32, 32, 32], "percentile_90_0": [32, 32, 32]},
                    },
                    "image_foreground_stats": {"intensity": {"percentile_00_5": -2, "percentile_99_5": 2}},
                    "label_stats": {"labels": [0, 1]},
                }
            }
            input_file, stats_file = root / "input.yaml", root / "datastats.yaml"
            ConfigParser.export_config_file(data_source, str(input_file), sort_keys=False)
            ConfigParser.export_config_file(stats, str(stats_file))
            algo = algo_class(template_path=str(ALGO_TEMPLATES))
            algo.name = algo_name
            algo.set_data_source(str(input_file))
            algo.set_data_stats(str(stats_file))
            algo.export_to_disk(temp_dir, "bundle", **(export_kwargs or {}))

            config = ConfigParser.load_config_file(str(root / "bundle" / "configs" / "hyper_parameters.yaml"))
            self.assertEqual(config["normalize_mode"], expected)
            self.assertEqual(algo.fill_records["hyper_parameters.yaml"]["normalize_mode"], expected)
            expected_extras = (export_kwargs or {}).get("extra_modalities", extra_modalities or {})
            self.assertEqual(list(config.get("extra_modalities", {})), list(expected_extras))
            expected_channels = (export_kwargs or {}).get("input_channels", 1 + len(expected_extras))
            self.assertEqual(config["input_channels"], expected_channels)
            if extra_modalities is None and "extra_modalities" in (export_kwargs or {}):
                algo.fill_template_config(str(stats_file), str(root / "bundle"), **export_kwargs)
                refilled = ConfigParser.load_config_file(str(root / "bundle" / "configs" / "hyper_parameters.yaml"))
                self.assertEqual(refilled["input_channels"], expected_channels)
                self.assertEqual(refilled["normalize_mode"], expected)
            # Re-export after restoring serialized Algo state, as cached bundle workflows do.
            state = copy.deepcopy(algo.state_dict())
            algo = algo_class()
            algo.load_state_dict(state)
            algo.export_to_disk(temp_dir, "restored", fill_with_datastats=False)
            restored = ConfigParser.load_config_file(str(root / "restored" / "configs" / "hyper_parameters.yaml"))
            self.assertEqual(restored["normalize_mode"], expected)
            if refill_kwargs is not None:
                algo.export_to_disk(temp_dir, "updated", fill_with_datastats=False, **refill_kwargs)
                updated = ConfigParser.load_config_file(str(root / "updated" / "configs" / "hyper_parameters.yaml"))
                self.assertEqual(updated["normalize_mode"], refill_expected)
                self.assertEqual(updated["input_channels"], 1 + len(updated.get("extra_modalities", {})))
                self.assertEqual(algo.fill_records["hyper_parameters.yaml"]["normalize_mode"], refill_expected)
                if "normalize_mode#1" in refill_kwargs:
                    algo.export_to_disk(
                        temp_dir, "explicit", fill_with_datastats=False, normalize_mode=["none", "meanstd"]
                    )
                    explicit = ConfigParser.load_config_file(
                        str(root / "explicit" / "configs" / "hyper_parameters.yaml")
                    )
                    self.assertEqual(explicit["normalize_mode"], ["none", "meanstd"])
            # A generated config must also be consumable by the training/inference pipeline.
            builder_class = importlib.import_module(f"{algo_name}.scripts.segmenter").DataTransformBuilder
            builder = builder_class(
                roi_size=[4, 4, 4],
                normalize_mode=config["normalize_mode"],
                extra_modalities=config.get("extra_modalities", {}),
                normalize_params={"intensity_bounds": config["intensity_bounds"]},
            )
            self.assertTrue(builder.get_normalize_transforms())

    @parameterized.expand([(algo,) for algo in ALGORITHMS])
    def test_list_normalizes_inputs_without_changing_label_image(self, algo_name):
        """The list takes precedence over legacy extra-modalities modes; labels keep their values."""
        builder_class = importlib.import_module(f"{algo_name}.scripts.segmenter").DataTransformBuilder
        builder = builder_class(
            roi_size=[2, 2, 2],
            normalize_mode=["meanstd", "none", "range"],
            extra_modalities={"image_z": "mri", "image_a": "none"},
            normalize_params={"intensity_bounds": [-2, 2]},
        )
        image = MetaTensor([1.0, 3.0, 5.0, 7.0]).reshape(1, 2, 2)
        labels = MetaTensor([0.0, 1.0, 2.0, 3.0]).reshape(1, 2, 2)
        ct = MetaTensor([-2.0, 0.0, 2.0, 4.0]).reshape(1, 2, 2)
        result = Compose(builder.get_normalize_transforms())({"image": image, "image_z": labels, "image_a": ct})
        expected_mri = torch.tensor([-1.3416408, -0.4472136, 0.4472136, 1.3416408]).reshape(2, 2)
        torch.testing.assert_close(result["image"][0], expected_mri)
        torch.testing.assert_close(result["image"][1], labels[0])
        torch.testing.assert_close(result["image"][2], torch.sigmoid(ct[0] / 2))
        self.assertNotIn("image_z", result)
        self.assertNotIn("image_a", result)

    @parameterized.expand([(algo,) for algo in ALGORITHMS])
    def test_legacy_scalar_keeps_extra_modality_mode(self, algo_name):
        builder_class = importlib.import_module(f"{algo_name}.scripts.segmenter").DataTransformBuilder
        builder = builder_class(roi_size=[2, 2, 2], normalize_mode="meanstd", extra_modalities={"image2": "none"})
        image = MetaTensor([1.0, 3.0, 5.0, 7.0]).reshape(1, 2, 2)
        labels = MetaTensor([0.0, 1.0, 2.0, 3.0]).reshape(1, 2, 2)
        result = Compose(builder.get_normalize_transforms())({"image": image, "image2": labels})
        torch.testing.assert_close(
            result["image"][0], torch.tensor([-1.3416408, -0.4472136, 0.4472136, 1.3416408]).reshape(2, 2)
        )
        torch.testing.assert_close(result["image"][1], labels[0])

    @parameterized.expand([(algo,) for algo in ALGORITHMS])
    def test_multichannel_primary_and_custom_key(self, algo_name):
        builder_class = importlib.import_module(f"{algo_name}.scripts.segmenter").DataTransformBuilder
        builder = builder_class(
            roi_size=[2, 2, 2],
            image_key="scan",
            normalize_mode=["meanstd", "none"],
            extra_modalities={"prior": "none"},
        )
        image = MetaTensor([[1.0, 3.0, 5.0, 7.0], [2.0, 6.0, 10.0, 14.0]]).reshape(2, 2, 2)
        labels = MetaTensor([0.0, 1.0, 2.0, 3.0]).reshape(1, 2, 2)
        result = Compose(builder.get_normalize_transforms())({"scan": image, "prior": labels})
        expected = torch.tensor([-1.3416408, -0.4472136, 0.4472136, 1.3416408]).reshape(2, 2)
        torch.testing.assert_close(result["scan"][0], expected)
        torch.testing.assert_close(result["scan"][1], expected)
        torch.testing.assert_close(result["scan"][2], labels[0])

    @parameterized.expand([(algo,) for algo in ALGORITHMS])
    def test_nifti_preprocessing_preserves_label_input(self, algo_name):
        """Exercise the complete inference preprocessing pipeline with real image files."""
        if algo_name == "segresnet2d":
            builder_class = importlib.import_module(f"{algo_name}.scripts.segmenter_2d").DataTransformBuilder2D
        else:
            builder_class = importlib.import_module(f"{algo_name}.scripts.segmenter").DataTransformBuilder
        builder = builder_class(
            roi_size=[2, 2, 2], normalize_mode=["meanstd", "none"], extra_modalities={"image2": "none"}
        )
        image = np.arange(1, 9, dtype=np.float32).reshape(2, 2, 2)
        labels = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path, label_path = Path(temp_dir) / "image.nii.gz", Path(temp_dir) / "prior.nii.gz"
            nib.save(nib.Nifti1Image(image, np.eye(4)), image_path)
            nib.save(nib.Nifti1Image(labels, np.eye(4)), label_path)
            result = builder(augment=False)({"image": str(image_path), "image2": str(label_path)})
        self.assertEqual(tuple(result["image"].shape), (2, 2, 2, 2))
        torch.testing.assert_close(result["image"][0].mean(), torch.tensor(0.0), atol=1e-6, rtol=0)
        torch.testing.assert_close(result["image"][0].std(unbiased=False), torch.tensor(1.0))
        torch.testing.assert_close(result["image"][1], torch.from_numpy(labels))
        self.assertNotIn("image2", result)

    @parameterized.expand(
        [(algo, modes) for algo in ALGORITHMS for modes in ([], ["meanstd"], ["meanstd", "none", "pet"])]
    )
    def test_list_length_must_match_input_keys(self, algo_name, modes):
        builder_class = importlib.import_module(f"{algo_name}.scripts.segmenter").DataTransformBuilder
        builder = builder_class(roi_size=[2, 2, 2], normalize_mode=modes, extra_modalities={"image2": "none"})
        with self.assertRaisesRegex(ValueError, "normalize_mode.*2"):
            builder.get_normalize_transforms()


if __name__ == "__main__":
    unittest.main()
