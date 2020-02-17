# wasm_dldt

## Bring dldt to the Web Platform using WebAssembly.

At the moment, OpenCV is able to import and inference such models as Caffe, Tensorflow, ONNX, Torch, Darknet and Intel Models. But OpenCV.js (OpenCV for a browser) can't use [Intel Models](https://download.01.org/opencv/2019/open_model_zoo/R2/20190628_153000_models_bin/) because such functionality depends on external project (OpenVINO) which is not included in WASM (OpenCV.js).
So we are working on cmake configuration files and build scripts to integrate Inference Engine from OpenVINO to OpenCV.js. This will allow the use of Intel optimized models also called Model Zoo for IA architecture in a browser.
Model Zoo for Intel Architecture contains Intel optimizations for running deep learning workloads on Intel Xeon Scalable processors.

## How to run - change the paths accordingly (embed/preload and various options)

cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="-s USE_PTHREADS=0 -s SIDE_MODULE=1 -s ASSERTIONS=1 --preload-file /home/sasha/workspace/projects/dldt/inference-engine/bin/Release/lib/libcpu_extension.so@libcpu_extension.so -g4 --source-map-base" -DCMAKE_CXX_FLAGS="-s USE_PTHREADS=0 -s SIDE_MODULE=1 -s ASSERTIONS=1 --preload-file /home/sasha/workspace/projects/dldt/inference-engine/bin/Release/lib/libcpu_extension.so@libcpu_extension.so -g4 --source-map-base" -DCMAKE_TOOLCHAIN_FILE=/home/sasha/workspace/projects/emsdk/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake ..

make --jobs=$(nproc --all)


cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="-s USE_PTHREADS=0 -s SIDE_MODULE=1 -s ASSERTIONS=1 --preload-file /home/sasha/workspace/projects/dldt/inference-engine/bin/Release/lib/libcpu_extension.so -g4 --source-map-base" -DCMAKE_CXX_FLAGS="-s USE_PTHREADS=0 -s SIDE_MODULE=1 -s ASSERTIONS=1 --preload-file /home/sasha/workspace/projects/dldt/inference-engine/bin/Release/lib/libcpu_extension.so -g4 --source-map-base" -DCMAKE_TOOLCHAIN_FILE=/home/sasha/workspace/projects/emsdk/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake ..


cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="-s USE_PTHREADS=0 -s SIDE_MODULE=1 -s ASSERTIONS=1 --embed-file /home/sasha/workspace/projects/dldt/inference-engine/bin/Release/lib/libcpu_extension.so -g4 --source-map-base" -DCMAKE_CXX_FLAGS="-s USE_PTHREADS=0 -s SIDE_MODULE=1 -s ASSERTIONS=1 --embed-file /home/sasha/workspace/projects/dldt/inference-engine/bin/Release/lib/libcpu_extension.so -g4 --source-map-base" -DCMAKE_TOOLCHAIN_FILE=/home/sasha/workspace/projects/emsdk/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake ..
