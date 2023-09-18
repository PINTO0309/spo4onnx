# spo4onnx
Simple tool for partial optimization of ONNX.

Further optimize some models that cannot be optimized with [onnx-optimizer](https://github.com/onnx/optimizer) and [onnxsim](https://github.com/daquexian/onnx-simplifier) by several tens of percent. In particular, models containing `Einsum` and `OneHot`. In other words, the goal is to raise the optimization capacity of [onnxsim](https://github.com/daquexian/onnx-simplifier).

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
