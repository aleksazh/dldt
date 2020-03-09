# wasm_dldt

## Bring dldt to the Web Platform using WebAssembly.

At the moment, OpenCV is able to import and inference such models as Caffe, Tensorflow, ONNX, Torch, Darknet and Intel Models. But OpenCV.js (OpenCV for a browser) can't use [Intel Models](https://download.01.org/opencv/2019/open_model_zoo/R2/20190628_153000_models_bin/) because such functionality depends on external project (OpenVINO) which is not included in WASM (OpenCV.js).
So we are working on cmake configuration files and build scripts to integrate Inference Engine from OpenVINO to OpenCV.js. This will allow the use of Intel optimized models also called Model Zoo for IA architecture in a browser.
Model Zoo for Intel Architecture contains Intel optimizations for running deep learning workloads on Intel Xeon Scalable processors.

## How to run

### Build DLDT (inference engine)

Execute `embuild.py` python script to build the project:

```sh
python embuild.py <build_dir>
```

With `pthreads` support:

```sh
python embuild.py build --threads
```

Clean build dir:

```sh
python embuild.py build --clean_build_dir
```

Skip config or do config only:

```sh
python embuild.py build--skip_config/--config_only
```

Add any extra cmake options:

```sh
python embuild.py build --cmake option="-D...=ON"
```

Add any extra Emscripten options:

```sh
python embuild.py build --build_flags="-s EXPORTED_FUNCTIONS=['_myFunction']"
```

### Build DLDT in cmd (inference engine)

```sh
cd inference-engine
mkdir build
cd build
emconfigure cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="-s USE_PTHREADS=0 -s SIDE_MODULE=1" -DCMAKE_CXX_FLAGS="-s USE_PTHREADS=0 -s SIDE_MODULE=1" ..
emmake make --jobs=$(nproc --all)
```

After that, you have static libraries: inference engine, some plugins and extensions in the folder `dldt/inference-engine/bin/Release/lib`.

If you need a shared version of a library use `emcc` tool to convert it:

```sh
emcc libcpu_extension.a -s SIDE_MODULE=1 -o libcpu_extension.wasm
```

If you wish you can rename `.wasm` to `.so` but it does not make sense.

Instead of `emconfigure` and `emmake` you can also use `-DCMAKE_TOOLCHAIN_FILE` flag.

### Build OpenCV

OpenCV uses static library of inference engine and shared libraries of plugins and extensions (because of `dlopen` calls).

First, set path to inference engine lib in `/opt/intel/openvino_2019.3.376/deployment_tools/inference_engine/share/InferenceEngineConfig.cmake`.

Comment line:
```sh
find_library(IE_RELEASE_LIBRARY inference_engine "${IE_LIB_DIR}" NO_DEFAULT_PATH)
```
and use instead:
```
set(IE_RELEASE_LIBRARY "/home/sasha/workspace/projects/dldt/inference-engine/bin/Release/lib/libinference_engine.a")
```

Second, add extra flags inside `def get_build_flags(self)` in `build_js.py`.

`MAIN_MODULE` is mandatory:

```python
flags += "-s MAIN_MODULE=2 "
```

Actually, it is better to use `-s MAIN_MODULE=1` but we have duplicate symbol errors using this flag which are still not fixed:

```
wasm-ld: error: duplicate symbol: cv::ChessBoardDetector::cleanFoundConnectedQuads(std::__2::vector<cv::ChessBoardQuad*, std::__2::allocator<cv::ChessBoardQuad*> >&)
>>> defined in ../../lib/libopencv_calib3d.a(calibinit.cpp.o)
>>> defined in ../../lib/libopencv_calib3d.a(calibinit.cpp.o)
....
```

Add exported functions:

```python
flags += "-s EXPORTED_FUNCTIONS='['__ZNSt3__212basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEC1ERKS5_', '___cxa_guard_acquire', '___cxa_guard_release', '__Znwm', '_memcmp', '__ZdlPv']' "
```

Without this you have errors like:
```
Assertion failed: missing linked function '___cxa_guard_release'. perhaps a side module was not linked in? if this global was expected to arrive from a system library, try to build the MAIN_MODULE with EMCC_FORCE_STDLIBS=1 in the environment.
```

If you want to use a shared library which needs to be opened through `dlopen` use `--preload-file` or `--embed-file` flag like:
```python
flags += "--preload-file /home/sasha/workspace/projects/opencv/build_wasm/modules/js/libcpu_extension.wasm "
```

Finally, build opencv.js:
```sh
python ./platforms/js/build_js.py build_wasm --build_wasm --emscripten_dir="/home/sasha/workspace/projects/emsdk/upstream/emscripten"
```

# [OpenVINO™ Toolkit](https://01.org/openvinotoolkit) - Deep Learning Deployment Toolkit repository
[![Stable release](https://img.shields.io/badge/version-2020.1-green.svg)](https://github.com/opencv/dldt/releases/tag/2020.1)
[![Apache License Version 2.0](https://img.shields.io/badge/license-Apache_2.0-green.svg)](LICENSE)

This toolkit allows developers to deploy pre-trained deep learning models 
through a high-level C++ Inference Engine API integrated with application logic. 

This open source version includes two components: namely [Model Optimizer] and 
[Inference Engine], as well as CPU, GPU and heterogeneous plugins to accelerate 
deep learning inferencing on Intel® CPUs and Intel® Processor Graphics. 
It supports pre-trained models from the [Open Model Zoo], along with 100+ open 
source and public models in popular formats such as Caffe\*, TensorFlow\*, 
MXNet\* and ONNX\*. 

## Repository components:
* [Inference Engine]
* [Model Optimizer]

## License
Deep Learning Deployment Toolkit is licensed under [Apache License Version 2.0](LICENSE).
By contributing to the project, you agree to the license and copyright terms therein 
and release your contribution under these terms.

## Documentation
* [OpenVINO™ Release Notes](https://software.intel.com/en-us/articles/OpenVINO-RelNotes)
* [OpenVINO™ Inference Engine Build Instructions](build-instruction.md)
* [Get Started with Deep Learning Deployment Toolkit on Linux](get-started-linux.md)\*
* [Introduction to Deep Learning Deployment Toolkit](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_Introduction.html)
* [Inference Engine Developer Guide](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_Deep_Learning_Inference_Engine_DevGuide.html)
* [Model Optimizer Developer Guide](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)

## How to Contribute
We welcome community contributions to the Deep Learning Deployment Toolkit 
repository. If you have an idea how to improve the product, please share it 
with us doing the following steps:

* Make sure you can build the product and run all tests and samples with your patch
* In case of a larger feature, provide relevant unit tests and one or more sample
* Submit a pull request at https://github.com/opencv/dldt/pulls

We will review your contribution and, if any additional fixes or modifications 
are necessary, may give some feedback to guide you. Your pull request will be 
merged into GitHub* repositories if accepted.

## Support
Please report questions, issues and suggestions using:

* The `openvino` [tag on StackOverflow]\*
* [GitHub* Issues](https://github.com/opencv/dldt/issues) 
* [Forum](https://software.intel.com/en-us/forums/computer-vision)

---
\* Other names and brands may be claimed as the property of others.

[Open Model Zoo]:https://github.com/opencv/open_model_zoo
[Inference Engine]:https://software.intel.com/en-us/articles/OpenVINO-InferEngine
[Model Optimizer]:https://software.intel.com/en-us/articles/OpenVINO-ModelOptimizer
[tag on StackOverflow]:https://stackoverflow.com/search?q=%23openvino
