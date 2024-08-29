if [ -z "$ANDROID_NDK" ]; then
    echo "Please set ANDROID_NDK!"
    exit
fi

if [ -z "$ANDROID_ABI" ]; then
    ANDROID_ABI=arm64-v8a
fi

rm -rf cmake-out
cmake . \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=$ANDROID_ABI \
    -B cmake-out
cmake --build cmake-out -j16
