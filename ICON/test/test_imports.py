import unittest


class TestImports(unittest.TestCase):
    def test_imports_CPU(self):
        import icon_registration
        import icon_registration.networks
        import icon_registration.data

    def test_requirements_match_cfg_CPU(self):
        from inspect import getsourcefile
        import os.path as path, sys
        import configparser

        current_dir = path.dirname(path.abspath(getsourcefile(lambda: 0)))
        parent_dir = current_dir[: current_dir.rfind(path.sep)]

        with open(parent_dir + "/requirements.txt") as f:
            requirements_txt = "\n" + f.read()
        requirements_cfg = configparser.ConfigParser()
        requirements_cfg.read(parent_dir + "/setup.cfg")
        requirements_cfg = requirements_cfg["options"]["install_requires"] + "\n"
        self.assertEqual(requirements_txt, requirements_cfg)

    def test_pytorch_cuda(self):
        import torch

        x = torch.Tensor([7]).cuda
