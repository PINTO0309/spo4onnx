# spo4onnx
Simple tool for partial optimization of ONNX

1. Temporarily downgrade onnxsim to `0.4.30` to perform my own optimization sequence.
2. After the optimization process is complete, reinstall the original onnxsim version to restore the environment.
3. The first version modifies two OPs, `Einsum` and `OneHot`, which hinder optimization and boost the optimization operation by onnxsim to maximum performance.
4. Not all models will be effective, but the larger and more complex the structure and the larger the model, the more effective this unique optimization behavior will be.
5. I have already identified models that can reduce redundant operations by up to 30%-40%.
6. An example of the most extreme optimization of my model is shown in the figure below. Example of optimization from 9,988 OP to 3,927 OP. The assumption is that this is an example of a huge ONNX with undefined Hieght and Width dimensions, set to fixed resolution and my special optimization technique applied.

![image](https://github.com/PINTO0309/spo4onnx/assets/33194443/dfb36e72-6898-4d71-a0bf-f6187b5bd877)

![image](https://github.com/PINTO0309/spo4onnx/assets/33194443/6efceb56-5e7e-4d88-b368-35342cfe0fcc)

![image](https://github.com/PINTO0309/spo4onnx/assets/33194443/d50adf77-4859-4c5e-8322-ef6698c1a771)

