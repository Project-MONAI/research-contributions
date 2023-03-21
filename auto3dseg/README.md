# Auto3DSeg algorithm templates

## Template testing

A unit test script is provided to evaluate the integrity of all algorithm templates in `auto3dseg/algorithm_templates`. This include a 2-epoch training and the inference of trained models on a single GPU ("cuda:0") in the testing process.


```
python auto3dseg/tests/test_algo_templates.py
python auto3dseg/tests/test_gpu_customization.py
```

## Adding new templates

### Class/Folder naming convention

- Folder name: name of the algorithm in lower cases
- Class name: folder name with the first letter in upper case + "Algo". e.g. UnetAlgo, SegresnetAlgo, etc.
