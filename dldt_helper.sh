#!/bin/sh

echo "Creating wasm from static library"
cd bin/x86/Release/lib/
emcc libMKLDNNPlugin.a -s SIDE_MODULE=1 -s ASSERTIONS=1 -s DISABLE_EXCEPTION_CATCHING=0 -s LINKABLE=1 -s NO_EXIT_RUNTIME=1 -s EXPORTED_FUNCTIONS=['_sched_getaffinity','__ZN6mkldnn4impl3cpu14engine_factoryE','_ZTVN6ngraph4pass8ValidateE','__ZTV21mkldnn_primitive_desc'] -o libMKLDNNPlugin.wasm
echo "Rename .wasm to .so"
mv libMKLDNNPlugin.wasm libMKLDNNPlugin.so
echo "Copy files to opencv project"
mv libMKLDNNPlugin.so ../../../../../WebCamera/samples/faceDetection/
cp plugins.xml ../../../../../opencv/build_wasm/modules/js/
echo "Done."
