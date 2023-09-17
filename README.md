# spo4onnx
Simple tool for partial optimization of ONNX

1. Temporarily downgrade onnxsim to `0.4.30` to perform my own optimization sequence.
2. After the optimization process is complete, reinstall the original onnxsim version to restore the environment.
3. The first version modifies two OPs, `Einsum` and `OneHot`, which hinder optimization and boost the optimization operation by onnxsim to maximum performance.
4. Not all models will be effective, but the larger and more complex the structure and the larger the model, the more effective this unique optimization behavior will be.
5. I have already identified models that can reduce redundant operations by up to 30%-40%.
