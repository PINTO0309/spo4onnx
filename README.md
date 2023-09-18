# spo4onnx
Simple tool for partial optimization of ONNX.

Further optimize some models that cannot be optimized with [onnx-optimizer](https://github.com/onnx/optimizer) and [onnxsim](https://github.com/daquexian/onnx-simplifier) by several tens of percent. In particular, models containing `Einsum` and `OneHot`. In other words, the goal is to raise the optimization capacity of [onnxsim](https://github.com/daquexian/onnx-simplifier).

https://github.com/PINTO0309/simple-onnx-processing-tools

[![Downloads](https://static.pepy.tech/personalized-badge/spo4onnx?period=total&units=none&left_color=grey&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/spo4onnx) ![GitHub](https://img.shields.io/github/license/PINTO0309/spo4onnx?color=2BAF2B) [![PyPI](https://img.shields.io/pypi/v/spo4onnx?color=2BAF2B)](https://pypi.org/project/spo4onnx/) [![CodeQL](https://github.com/PINTO0309/spo4onnx/workflows/CodeQL/badge.svg)](https://github.com/PINTO0309/spo4onnx/actions?query=workflow%3ACodeQL)

```
pip install -U spo4onnx \
&& pip install -U onnx \
&& pip install -U onnxruntime \
&& pip install onnxsim \
&& python3 -m pip install -U onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com
```

https://github.com/PINTO0309/spo4onnx/releases/download/model/wd-v1-4-moat-tagger-v2.onnx

![Kazam_screencast_00060_](https://github.com/PINTO0309/spo4onnx/assets/33194443/2fa84a50-a26c-47c9-99f7-845732adffb8)

1. Temporarily downgrade onnxsim to `0.4.30` to perform my own optimization sequence.
2. After the optimization process is complete, reinstall the original onnxsim version to restore the environment.
3. The first version modifies two OPs, `Einsum` and `OneHot`, which hinder optimization and boost the optimization operation by onnxsim to maximum performance.
4. Not all models will be effective, but the larger and more complex the structure and the larger the model, the more effective this unique optimization behavior will be.
5. I have already identified models that can reduce redundant operations by up to 30%-60%.
6. An example of the most extreme optimization of my model is shown in the figure below. Example of optimization from 9,988 OP to 3,927 OP. The assumption is that this is an example of a huge ONNX with undefined Hieght and Width dimensions, set to fixed resolution and my special optimization technique applied. By making OPs such as `Tile` disappear and embedded in the model as INT64 constants, the final model file size is increased, but the model structure is greatly optimized.

https://github.com/PINTO0309/spo4onnx/releases/download/model/high_frequency_stereo_matching_kitti_iter01_1x3xHxW.onnx

![image](https://github.com/PINTO0309/spo4onnx/assets/33194443/dfb36e72-6898-4d71-a0bf-f6187b5bd877)

![image](https://github.com/PINTO0309/spo4onnx/assets/33194443/6efceb56-5e7e-4d88-b368-35342cfe0fcc)

![image](https://github.com/PINTO0309/spo4onnx/assets/33194443/d50adf77-4859-4c5e-8322-ef6698c1a771)

Verify that the inference works properly.
```bash
sit4onnx \
-if high_frequency_stereo_matching_kitti_iter05_1x3x192x320.onnx \
-oep tensorrt

INFO: file: high_frequency_stereo_matching_kitti_iter05_1x3x192x320.onnx
INFO: providers: ['TensorrtExecutionProvider', 'CPUExecutionProvider']
INFO: input_name.1: left shape: [1, 3, 192, 320] dtype: float32
INFO: input_name.2: right shape: [1, 3, 192, 320] dtype: float32
INFO: test_loop_count: 10
INFO: total elapsed time:  185.7011318206787 ms
INFO: avg elapsed time per pred:  18.57011318206787 ms
INFO: output_name.1: output shape: [1, 1, 192, 320] dtype: float32

sit4onnx \
-if high_frequency_stereo_matching_kitti_iter05_1x3x192x320.onnx \
-oep cpu

INFO: file: high_frequency_stereo_matching_kitti_iter05_1x3x192x320.onnx
INFO: providers: ['CPUExecutionProvider']
INFO: input_name.1: left shape: [1, 3, 192, 320] dtype: float32
INFO: input_name.2: right shape: [1, 3, 192, 320] dtype: float32
INFO: test_loop_count: 10
INFO: total elapsed time:  4090.1401042938232 ms
INFO: avg elapsed time per pred:  409.0140104293823 ms
INFO: output_name.1: output shape: [1, 1, 192, 320] dtype: float32
```

## 1. CLI Usage
```
spo4onnx -h

spo4onnx \
[-h] \
-if INPUT_ONNX_FILE_PATH \
[-of OUTPUT_ONNX_FILE_PATH] \
[-ois OVERWRITE_INPUT_SHAPE [OVERWRITE_INPUT_SHAPE ...]] \
[-ot OPTIMIZATION_TIMES] \
[-tov TARGET_ONNXSIM_VERSION]
[-n]

options:
  -h, --help
    show this help message and exit

  -if INPUT_ONNX_FILE_PATH, --input_onnx_file_path INPUT_ONNX_FILE_PATH
    Input onnx file path.

  -of OUTPUT_ONNX_FILE_PATH, --output_onnx_file_path OUTPUT_ONNX_FILE_PATH
    Output onnx file path. If not specified,
    it will overwrite the onnx specified in --input_onnx_file_path.

  -ois OVERWRITE_INPUT_SHAPE [OVERWRITE_INPUT_SHAPE ...], \
    --overwrite_input_shape OVERWRITE_INPUT_SHAPE [OVERWRITE_INPUT_SHAPE ...]
    Overwrite the input shape.
    The format is "input_1:dim0,...,dimN" "input_2:dim0,...,dimN" "input_3:dim0,...,dimN"
    When there is only one input, for example, "data:1,3,224,224"
    When there are multiple inputs, for example, "data1:1,3,224,224" "data2:1,3,112,112" "data3:5"
    A value of 1 or more must be specified.
    Numerical values other than dynamic dimensions are ignored.

  -ot OPTIMIZATION_TIMES, --optimization_times OPTIMIZATION_TIMES
    Number of times the optimization process is performed.
    If zero is specified, the tool automatically calculates the number of optimization times.
    Default: 0

  -tov TARGET_ONNXSIM_VERSION, --target_onnxsim_version TARGET_ONNXSIM_VERSION
    Version number of the onnxsim used for optimization.
    Default: 0.4.30

  -n, --non_verbose
    Do not show all information logs. Only error logs are displayed.
```

## 2. In-script Usage
```python
$ python
>>> from spo4onnx import partial_optimization
>>> help(partial_optimization)

Help on function partial_optimization in module spo4onnx.onnx_partial_optimization:

partial_optimization(
    input_onnx_file_path: Optional[str] = '',
    onnx_graph: Optional[onnx.onnx_ml_pb2.ModelProto] = None,
    output_onnx_file_path: Optional[str] = '',
    overwrite_input_shape: Optional[Dict] = None,
    optimization_times: Optional[int] = 0,
    target_onnxsim_version: Optional[str] = '0.4.30',
    non_verbose: Optional[bool] = False,
) -> onnx.onnx_ml_pb2.ModelProto

    Parameters
    ----------
    input_onnx_file_path: Optional[str]
        Input onnx file path.
        Either input_onnx_file_path or onnx_graph must be specified.
        Default: ''

    onnx_graph: Optional[onnx.ModelProto]
        onnx.ModelProto.
        Either input_onnx_file_path or onnx_graph must be specified.
        onnx_graph If specified, ignore input_onnx_file_path and process onnx_graph.

    output_onnx_file_path: Optional[str]
        Output onnx file path.
        If not specified, it will overwrite the onnx specified in --input_onnx_file_path.

    overwrite_input_shape: Optional[Dict]
        Overwrite the input shape.
        The format is
        {'data1': [1, 3, 224, 224], 'data2': [1, 224], 'data3': [1]}

    optimization_times: Optional[int]
        Number of times the optimization process is performed.
        If zero is specified, the tool automatically calculates the number of optimization times.
        Default: 0

    non_verbose: Optional[bool]
        Do not show all information logs. Only error logs are displayed.

    Returns
    -------
    onnx_graph: onnx.ModelProto
        Optimized onnx ModelProto
```
