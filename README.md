# wasm_dldt

## Bring dldt to the Web Platform using WebAssembly.

At the moment, OpenCV is able to import and inference such models as Caffe, Tensorflow, ONNX, Torch, Darknet and Intel Models. But OpenCV.js (OpenCV for a browser) can't use [Intel Models](https://download.01.org/opencv/2019/open_model_zoo/R2/20190628_153000_models_bin/) because such functionality depends on external project (OpenVINO) which is not included in WASM (OpenCV.js).
So we are working on cmake configuration files and build scripts to integrate Inference Engine from OpenVINO to OpenCV.js. This will allow the use of Intel optimized models also called Model Zoo for IA architecture in a browser.
Model Zoo for Intel Architecture contains Intel optimizations for running deep learning workloads on Intel Xeon Scalable processors.

## How to run

### Build DLDT (inference engine)

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




