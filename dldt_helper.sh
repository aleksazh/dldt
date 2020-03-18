#!/bin/sh

echo "Creating wasm from static library"
cd bin/x86/Release/lib/
emcc libMKLDNNPlugin.a -s SIDE_MODULE=1 -s EXTRA_EXPORTED_RUNTIME_METHODS=['__ZTI21mkldnn_primitive_desc'] -s EXPORTED_FUNCTIONS=['__ZTI21mkldnn_primitive_desc'] --use-preload-plugins -o libMKLDNNPlugin.wasm
echo "Rename .wasm to .so"
mv libMKLDNNPlugin.wasm libMKLDNNPlugin.so
echo "Copy files to opencv project"
mv libMKLDNNPlugin.so ../../../../../opencv/build_wasm/modules/js/
cp plugins.xml ../../../../../opencv/build_wasm/modules/js/
echo "Done."
