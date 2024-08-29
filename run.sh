find . -name "*.so" | while read solib; do
    adb push $solib  /data/local/tmp/
done

adb push cmake-out/benchmark /data/local/tmp
adb shell LD_LIBRARY_PATH=/data/local/tmp /data/local/tmp/benchmark
