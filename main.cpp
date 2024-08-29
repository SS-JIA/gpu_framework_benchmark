#include <iostream>
#include <memory>
#include <random>

#include <core/Backend.hpp>

#include <MNN/MNNForwardType.h>
#include <MNN/Tensor.hpp>

#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Optimizer.hpp>

#include <MNN/AutoTime.hpp>

#include <backend/cpu/CPUBackend.hpp>
// #include <backend/opencl/core/runtime/OpenCLRuntime.hpp>

void fill_random(
    std::vector<float>& data,
    const float min = 0.0,
    const float max = 1.0) {
  std::default_random_engine rng;
  std::uniform_real_distribution<float> dist(min, max);

  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = 1; // dist(rng);
  }
}

void run_mnn_matmul(const int L) {
  using namespace MNN;
  using namespace MNN::Express;

  auto exe = Executor::getGlobalExecutor();
  MNN::BackendConfig config;
  MNNForwardType forwardType = MNN_FORWARD_OPENCL;
  config.precision = MNN::BackendConfig::Precision_Low;
  exe->setGlobalExecutorConfig(forwardType, config, 4);
  // OpenCLRuntime* runtime = (OpenCLRuntime*)(exe->getRuntime().first[MNN_FORWARD_OPENCL].get());
  // runtime->setGpuMode(MNN_GPU_MEMORY_BUFFER);

  {
    AUTOTIME;

    auto input_a = _Input({L, L}, Dimensionformat::NCHW);
    auto input_b = _Input({L, L}, Dimensionformat::NCHW);

    std::vector<float> a_data(L * L);
    fill_random(a_data);
    std::vector<float> b_data(L * L);
    fill_random(b_data);

    ::memcpy(input_a->writeMap<float>(), a_data.data(), a_data.size() * sizeof(float));
    ::memcpy(input_b->writeMap<float>(), b_data.data(), b_data.size() * sizeof(float));

    VARP out_1  = _MatMul(input_a, input_b, false, false);
    VARP out_2  = _MatMul(out_1, input_b, false, false);
    VARP out_3  = _MatMul(out_2, input_b, false, false);
    VARP out_4  = _MatMul(out_3, input_b, false, false);
    VARP out_5  = _MatMul(out_4, input_b, false, false);

    auto outputPtr = out_5->readMap<float>();
  }
}

int main() {
  run_mnn_matmul(512);
  std::cout << "\ndone!" << std::endl;
  return 0;
}
