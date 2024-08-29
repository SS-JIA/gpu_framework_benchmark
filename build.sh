if [ -z "$ANDROID_NDK" ]; then
    echo "Please set ANDROID_NDK!"
    exit
fi

if [ -z "$ANDROID_ABI" ]; then
    ANDROID_ABI=arm64-v8a
fi

cmake . \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=$ANDROID_ABI \
    -DANDROID_STL=c++_static \
    -DMNN_VULKAN:BOOL=ON \
    -DMNN_OPENCL:BOOL=ON \
    -DMNN_OPENCL_PROFILE=ON \
    -DMNN_GPU_TRACE=ON \
    -B cmake-out
cmake --build cmake-out -j16 --target benchmark
