# Auto3DSeg algorithm templates

## Template testing

A unit test script is provided to evaluate the integrity of all algorithm templates in `auto3dseg/algorithm_templates`. This includes a 2-epoch training and the inference of trained models on a single GPU ("cuda:0") in the testing process.


```
python auto3dseg/tests/test_algo_templates.py
python auto3dseg/tests/test_gpu_customization.py
```

## Version control

If the folder `auto3dseg` is changed, a new `version` and the corresponding `changelog` should be added into the `metadata.json` file.

## Adding new templates

### Class/Folder naming convention

- Folder name: name of the algorithm in lowercase
- Class name: folder name with the first letter in upper case + "Algo". e.g. UnetAlgo, SegresnetAlgo, etc.

## Testing with MONAI core

Start a docker container:

```bash
docker run --ipc=host --net=host --gpus all -ti --rm projectmonai/monai
```

To test the github fork branch (assuming `my_test_branch` of `https://github.com/my_github/research-contributions.git`),
run the following script:
```bash
cd /tmp/
git clone --depth 1 --branch dev  https://github.com/project-monai/monai.git
git clone --depth 1 --branch my_test_branch https://github.com/my_github/research-contributions.git
cp -r research-contributions/auto3dseg/algorithm_templates/ monai/
cd monai/
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
MONAI_TESTING_ALGO_TEMPLATE=algorithm_templates python -m unittest -vvv tests.test_auto3dseg_ensemble
MONAI_TESTING_ALGO_TEMPLATE=algorithm_templates python -m unittest -vvv tests.test_auto3dseg_hpo
MONAI_TESTING_ALGO_TEMPLATE=algorithm_templates python -m unittest -vvv tests.test_integration_autorunner
MONAI_TESTING_ALGO_TEMPLATE=algorithm_templates python -m unittest -vvv tests.test_integration_gpu_customization
```
