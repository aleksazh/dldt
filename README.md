# wasm_dldt

Bring dldt to the Web Platform using WebAssembly.

At the moment, OpenCV is able to import and inference such models as Caffe, Tensorflow, ONNX, Torch, Darknet and Intel Models. But OpenCV.js (OpenCV for a browser) can't use [Intel Models](https://download.01.org/opencv/2019/open_model_zoo/R2/20190628_153000_models_bin/) because such functionality depends on external project (OpenVINO) which is not included in WASM (OpenCV.js).
So we are working on cmake configuration files and build scripts to integrate Inference Engine from OpenVINO to OpenCV.js. This will allow the use of Intel optimized models also called Model Zoo for IA architecture in a browser.
Model Zoo for Intel Architecture contains Intel optimizations for running deep learning workloads on Intel Xeon Scalable processors.
