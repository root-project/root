// For the licensing terms see $ROOTSYS/LICENSE.
#include "TMVA/RGPUModel.hxx"
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>
#ifdef SOFIE_TEST_HIP
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#define GPU(name) hip##name
#else
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#define GPU(name) cuda##name
#endif

using namespace TMVA::Experimental::SOFIE;
namespace {
struct Stream {
   GPU(Stream_t) value {};
   Stream()
   {
      if (GPU(StreamCreateWithFlags)(&value, GPU(StreamNonBlocking)))
         throw std::runtime_error("stream creation failed");
   }
   ~Stream()
   {
      GPU(StreamSynchronize)(value);
      GPU(StreamDestroy)(value);
   }
};
struct Buffer {
   void *data = nullptr;
   explicit Buffer(std::size_t bytes)
   {
      if (GPU(Malloc)(&data, bytes))
         throw std::bad_alloc();
   }
   ~Buffer() { GPU(Free)(data); }
};
float Value(float x)
{
   return x;
}
float Value(__half x)
{
   return __half2float(x);
}
} // namespace

TEST(SofieGPUDevice, ExternalStreamAndWorkspace)
{
   int count = 0;
   if (GPU(GetDeviceCount)(&count) != GPU(Success) || !count)
      GTEST_SKIP() << "No GPU available";
   int device = -1;
   ASSERT_EQ(GPU(GetDevice)(&device), GPU(Success));
#ifdef SOFIE_TEST_HIP
   hipDeviceProp_t properties;
#else
   cudaDeviceProp properties;
#endif
   ASSERT_EQ(GPU(GetDeviceProperties)(&properties, device), GPU(Success));
#ifdef SOFIE_TEST_HIP
   const RGPUModel::CompileOptions options{RGPUModel::Backend::HIP, SOFIE_TEST_COMPILER, properties.gcnArchName};
#else
   const RGPUModel::CompileOptions options{RGPUModel::Backend::CUDA, SOFIE_TEST_COMPILER,
                                           "sm_" + std::to_string(properties.major) + std::to_string(properties.minor)};
#endif
   for (bool half : {false, true}) {
      RGPUModel::Layer first{2, 2, {1.f, -2.f, .5f, 1.f}, {.25f, -.5f}, 1.f, 1.f, true};
      RGPUModel::Layer last{2, 1, {2.f, -1.f}, {.125f}};
      std::vector<RGPUModel::Layer> layers{first};
      for (int i = 0; i < 4; ++i)
         layers.push_back({2, 2, {1.f, 0.f, 0.f, 1.f}, {0.f, 0.f}, 1.f, 1.f, true});
      layers.push_back(last);
      RGPUModel model(half ? RGPUModel::Precision::Float16 : RGPUModel::Precision::Float32, std::move(layers));
      model.Compile(options);
      EXPECT_THROW(model.Compile(options), std::logic_error);
      Stream stream;
      Buffer arena(model.WorkspaceSize(7)), input(14 * sizeof(float)), output(7 * sizeof(float));
      auto session = model.CreateSession(stream.value, device, 7);
      EXPECT_THROW(session->SetWorkspace(arena.data, 1), std::invalid_argument);
      EXPECT_THROW(session->SetWorkspace(static_cast<char *>(arena.data) + 1, model.WorkspaceSize(7)),
                   std::invalid_argument);
      session->SetWorkspace(arena.data, model.WorkspaceSize(7));
      EXPECT_THROW(session->Infer(input.data, output.data, 8), std::invalid_argument);
      EXPECT_THROW(session->Infer(arena.data, output.data, 1), std::invalid_argument);
      EXPECT_THROW(session->Infer(input.data, input.data, 1), std::invalid_argument);
      auto run = [&](auto scalar) {
         using T = decltype(scalar);
         std::vector<T> host(14), result(7);
         for (std::size_t i = 0; i < host.size(); ++i)
            host[i] = T(float(i) * .125f - .625f);
         for (std::size_t batch : {1u, 3u, 7u}) {
            ASSERT_EQ(
               GPU(MemcpyAsync)(input.data, host.data(), batch * 2 * sizeof(T), GPU(MemcpyHostToDevice), stream.value),
               GPU(Success));
            session->Infer(input.data, output.data, batch);
            ASSERT_EQ(
               GPU(MemcpyAsync)(result.data(), output.data, batch * sizeof(T), GPU(MemcpyDeviceToHost), stream.value),
               GPU(Success));
            ASSERT_EQ(GPU(StreamSynchronize)(stream.value), GPU(Success));
            for (std::size_t row = 0; row < batch; ++row) {
               const float x = Value(host[2 * row]), y = Value(host[2 * row + 1]);
               const float a = std::max(0.f, Value(T(x - 2.f * y + .25f)));
               const float b = std::max(0.f, Value(T(.5f * x + y - .5f)));
               EXPECT_NEAR(Value(result[row]), Value(T(2.f * a - b + .125f)), half ? .002f : 1.e-6f);
            }
         }
      };
      if (half)
         run(__half{});
      else
         run(float{});
      ASSERT_EQ(GPU(StreamSynchronize)(stream.value), GPU(Success));
   }
}
