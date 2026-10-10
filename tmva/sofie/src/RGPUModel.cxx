// Author: Christian Sonnabend, 2026
// For the licensing terms see $ROOTSYS/LICENSE.
#include "TMVA/RGPUModel.hxx"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <utility>
#ifndef _WIN32
#include <dlfcn.h>
#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char **environ;
#endif

namespace TMVA::Experimental::SOFIE {
namespace {
std::size_t Multiply(std::size_t a, std::size_t b)
{
   if (b && a > std::numeric_limits<std::size_t>::max() / b)
      throw std::overflow_error("SOFIE GPU: tensor size overflow");
   return a * b;
}
std::size_t Add(std::size_t a, std::size_t b)
{
   if (a > std::numeric_limits<std::size_t>::max() - b)
      throw std::overflow_error("SOFIE GPU: workspace size overflow");
   return a + b;
}
std::size_t Align(std::size_t bytes)
{
   return Add(bytes, 255) & ~std::size_t(255);
}
} // namespace

struct RGPUModel::Program {
   using Upload = const char *(*)(void *, int, void *, const float *, std::size_t);
   using Invoke = const char *(*)(void *, int, void *, const void *, void *, std::size_t, std::size_t);
   void *library = nullptr;
   Upload upload = nullptr;
   Invoke invoke = nullptr;
   ~Program()
   {
#ifndef _WIN32
      if (library)
         dlclose(library);
#endif
   }
};

RGPUModel::RGPUModel(Precision precision, std::vector<Layer> layers, std::size_t fixedBatch)
   : fPrecision(precision), fLayers(std::move(layers)), fFixedBatch(fixedBatch)
{
   if (fLayers.empty())
      throw std::invalid_argument("SOFIE GPU: empty network");
   if (precision != Precision::Float32 && precision != Precision::Float16)
      throw std::invalid_argument("SOFIE GPU: unsupported precision");
   std::size_t previous = fLayers.front().inputs;
   for (const auto &layer : fLayers) {
      if (!layer.inputs || !layer.outputs || previous != layer.inputs ||
          layer.weights.size() != Multiply(layer.inputs, layer.outputs) || layer.bias.size() != layer.outputs ||
          !std::isfinite(layer.alpha) || !std::isfinite(layer.beta))
         throw std::invalid_argument("SOFIE GPU: invalid dense layer");
      fWeights.insert(fWeights.end(), layer.weights.begin(), layer.weights.end());
      fWeights.insert(fWeights.end(), layer.bias.begin(), layer.bias.end());
      fWidth = std::max(fWidth, layer.outputs);
      previous = layer.outputs;
   }
}
std::size_t RGPUModel::InputSize() const
{
   return fLayers.front().inputs;
}
std::size_t RGPUModel::OutputSize() const
{
   return fLayers.back().outputs;
}
std::size_t RGPUModel::WorkspaceSize(std::size_t maxBatch) const
{
   if (!maxBatch || (fFixedBatch && maxBatch != fFixedBatch))
      throw std::invalid_argument("SOFIE GPU: invalid maximum batch size");
   const auto tensor = Align(Multiply(Multiply(maxBatch, fWidth), fPrecision == Precision::Float16 ? 2 : 4));
   return Add(Align(Multiply(fWeights.size(), sizeof(float))), Multiply(2, tensor));
}

std::string RGPUModel::GenerateSource(Backend backend) const
{
   if (backend != Backend::CUDA && backend != Backend::HIP)
      throw std::invalid_argument("SOFIE GPU: unsupported backend");
   const std::string api = backend == Backend::HIP ? "hip" : "cuda";
   std::ostringstream out;
   out.imbue(std::locale::classic());
   out << std::scientific << std::setprecision(std::numeric_limits<float>::max_digits10);
   out << "#include <" << (backend == Backend::HIP ? "hip/hip_runtime.h" : "cuda_runtime.h") << ">\n"
       << "#include <" << (backend == Backend::HIP ? "hip/hip_fp16.h" : "cuda_fp16.h") << ">\n"
       << "#include <cstddef>\n#include <cstdint>\n#include <cstdlib>\n"
       << "using Scalar = " << (fPrecision == Precision::Float16 ? "__half" : "float") << ";\n"
       << "using Stream = " << api << "Stream_t;\n";
   out << R"cpp(
__device__ float read(float x) { return x; }
__device__ float read(__half x) { return __half2float(x); }
__device__ void write(float &x, float y) { x = y; }
__device__ void write(__half &x, float y) { x = __float2half_rn(y); }
template <size_t K, size_t N, bool Relu>
__global__ void dense(const Scalar *input, const float *weights, const float *bias,
                      Scalar *output, size_t batch, float alpha, float beta)
{
   for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < batch * N;
        i += size_t(blockDim.x) * gridDim.x) {
      const size_t row = i / N, col = i % N;
      float sum = 0.f;
      for (size_t k = 0; k < K; ++k) sum += read(input[row * K + k]) * weights[col * K + k];
      Scalar value;
      write(value, alpha * sum + beta * bias[col]);
      if (Relu && read(value) < 0.f) write(value, 0.f);
      output[i] = value;
   }
}
)cpp";
   out << "static const char *checkDevice(int expected) { int actual = -1; auto e = " << api
       << "GetDevice(&actual); if (e != " << api << "Success) return " << api
       << "GetErrorString(e); return actual == expected ? nullptr : \"SOFIE GPU: wrong active device\"; }\n";
   out << "extern \"C\" const char *sofie_upload(void *s, int device, void *arena, const float *weights, size_t bytes) "
          "{\n"
       << "if (auto e = checkDevice(device)) return e;\n"
       << "auto e = " << api << "MemcpyAsync(arena, weights, bytes, " << api
       << "MemcpyHostToDevice, static_cast<Stream>(s));\n"
       << "return e == " << api << "Success ? nullptr : " << api << "GetErrorString(e); }\n";
   out << "extern \"C\" const char *sofie_infer(void *s, int device, void *arena, const void *input, void *output, "
          "size_t batch, size_t maxBatch) {\n"
       << "if (auto e = checkDevice(device)) return e;\n"
       << "auto stream = static_cast<Stream>(s); auto weights = static_cast<const float *>(arena);\n"
       << "size_t stride = (maxBatch * " << fWidth << " * sizeof(Scalar) + 255) & ~size_t(255);\n"
       << "auto a = reinterpret_cast<Scalar *>(static_cast<char *>(arena) + " << Align(fWeights.size() * sizeof(float))
       << ");\n"
       << "auto b = reinterpret_cast<Scalar *>(reinterpret_cast<char *>(a) + stride);\n";
   std::size_t offset = 0;
   for (std::size_t i = 0; i < fLayers.size(); ++i) {
      const auto &l = fLayers[i];
      const auto count = l.inputs * l.outputs;
      out << "dense<" << l.inputs << ", " << l.outputs << ", " << (l.relu ? "true" : "false")
          << "><<<unsigned((batch * " << l.outputs << " + 255) / 256 > 65535 ? 65535 : (batch * " << l.outputs
          << " + 255) / 256), 256, 0, stream>>>("
          << (i == 0 ? "static_cast<const Scalar *>(input)" : (i % 2 ? "a" : "b")) << ", weights + " << offset
          << ", weights + " << offset + count << ", "
          << (i + 1 == fLayers.size() ? "static_cast<Scalar *>(output)" : (i % 2 ? "b" : "a")) << ", batch, " << l.alpha
          << "f, " << l.beta << "f);\n"
          << "if (auto e = " << api << "GetLastError(); e != " << api << "Success) return " << api
          << "GetErrorString(e);\n";
      offset += count + l.outputs;
   }
   out << "return nullptr;\n}\n";
   return out.str();
}

void RGPUModel::Compile(const CompileOptions &options)
{
   if (fProgram)
      throw std::logic_error("SOFIE GPU: model already compiled");
   const auto source = GenerateSource(options.backend);
   if (options.architecture.empty() ||
       options.architecture.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_:+-") !=
          std::string::npos)
      throw std::invalid_argument("SOFIE GPU: provide a CUDA/HIP architecture");
#ifndef _WIN32
   std::string pattern = (std::filesystem::temp_directory_path() / "sofie-gpu-XXXXXX").string();
   if (!mkdtemp(pattern.data()))
      throw std::runtime_error("SOFIE GPU: cannot create compilation directory");
   struct TemporaryDirectory {
      std::filesystem::path path;
      ~TemporaryDirectory()
      {
         std::error_code e;
         std::filesystem::remove_all(path, e);
      }
   } directory{pattern};
   const auto input = directory.path / "model.cu", library = directory.path / "model.so",
              log = directory.path / "compiler.log";
   {
      std::ofstream file(input);
      file << source;
      if (!file)
         throw std::runtime_error("SOFIE GPU: cannot write source");
   }
   const bool hip = options.backend == Backend::HIP;
   std::vector<std::string> args{options.compiler.empty() ? (hip ? "hipcc" : "nvcc") : options.compiler, "-std=c++17",
                                 "-O2", "-shared"};
   if (hip) {
      args.emplace_back("-fPIC");
      args.emplace_back("--offload-arch=" + options.architecture);
   } else {
      args.emplace_back("-Xcompiler=-fPIC");
      args.emplace_back("-arch=" + options.architecture);
   }
   args.insert(args.end(), {input.string(), "-o", library.string()});
   std::vector<char *> argv;
   for (auto &arg : args)
      argv.push_back(arg.data());
   argv.push_back(nullptr);
   posix_spawn_file_actions_t actions;
   if (posix_spawn_file_actions_init(&actions))
      throw std::runtime_error("SOFIE GPU: cannot initialize compiler process");
   int error =
      posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, log.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
   if (!error)
      error = posix_spawn_file_actions_adddup2(&actions, STDOUT_FILENO, STDERR_FILENO);
   pid_t pid = 0;
   if (!error)
      error = posix_spawnp(&pid, argv[0], &actions, nullptr, argv.data(), environ);
   posix_spawn_file_actions_destroy(&actions);
   if (error)
      throw std::runtime_error("SOFIE GPU: cannot start compiler " + args[0] + " (error " + std::to_string(error) +
                               ")");
   int status = 0;
   pid_t waited;
   do {
      waited = waitpid(pid, &status, 0);
   } while (waited < 0 && errno == EINTR);
   if (waited < 0 || !WIFEXITED(status) || WEXITSTATUS(status)) {
      std::ifstream file(log);
      std::ostringstream diagnostic;
      diagnostic << file.rdbuf();
      throw std::runtime_error("SOFIE GPU: compilation failed\n" + diagnostic.str());
   }
   auto program = std::make_shared<Program>();
   program->library = dlopen(library.c_str(), RTLD_NOW | RTLD_LOCAL);
   if (!program->library)
      throw std::runtime_error(std::string("SOFIE GPU: cannot load model: ") + dlerror());
   program->upload = reinterpret_cast<Program::Upload>(dlsym(program->library, "sofie_upload"));
   program->invoke = reinterpret_cast<Program::Invoke>(dlsym(program->library, "sofie_infer"));
   if (!program->upload || !program->invoke)
      throw std::runtime_error("SOFIE GPU: missing generated entry point");
   fProgram = std::move(program);
#else
   throw std::runtime_error("SOFIE GPU: runtime compilation currently requires a POSIX host");
#endif
}

RGPUModel::Session::Session(std::shared_ptr<Program> program, std::vector<float> weights, void *stream, int device,
                            std::size_t maxBatch, std::size_t fixedBatch, std::size_t bytes, std::size_t inputBytes,
                            std::size_t outputBytes)
   : fProgram(std::move(program)),
     fWeights(std::move(weights)),
     fStream(stream),
     fDevice(device),
     fMaxBatch(maxBatch),
     fFixedBatch(fixedBatch),
     fBytes(bytes),
     fInputBytesPerRow(inputBytes),
     fOutputBytesPerRow(outputBytes)
{
   if (device < 0)
      throw std::invalid_argument("SOFIE GPU: invalid device index");
}
RGPUModel::Session::~Session() = default;
std::unique_ptr<RGPUModel::Session> RGPUModel::CreateSession(void *stream, int device, std::size_t maxBatch) const
{
   if (!fProgram)
      throw std::logic_error("SOFIE GPU: compile the model before creating a session");
   const auto elementSize = fPrecision == Precision::Float16 ? 2u : 4u;
   return std::unique_ptr<Session>(new Session(fProgram, fWeights, stream, device, maxBatch, fFixedBatch,
                                               WorkspaceSize(maxBatch), Multiply(InputSize(), elementSize),
                                               Multiply(OutputSize(), elementSize)));
}
void RGPUModel::Session::SetWorkspace(void *workspace, std::size_t bytes)
{
   if (!workspace || reinterpret_cast<std::uintptr_t>(workspace) % WorkspaceAlignment() || bytes < fBytes)
      throw std::invalid_argument("SOFIE GPU: missing, undersized or unaligned workspace");
   fWorkspace = nullptr;
   if (auto error = fProgram->upload(fStream, fDevice, workspace, fWeights.data(), fWeights.size() * sizeof(float)))
      throw std::runtime_error(error);
   fWorkspace = workspace;
}
void RGPUModel::Session::Infer(const void *input, void *output, std::size_t batch) const
{
   if (batch > fMaxBatch || (batch && fFixedBatch && batch != fFixedBatch))
      throw std::invalid_argument("SOFIE GPU: batch outside the prepared range");
   if (!batch)
      return;
   if (!fWorkspace || !input || !output || input == output)
      throw std::invalid_argument("SOFIE GPU: provide workspace and distinct input/output buffers");
   auto overlaps = [](const void *a, std::size_t na, const void *b, std::size_t nb) {
      const auto x = reinterpret_cast<std::uintptr_t>(a), y = reinterpret_cast<std::uintptr_t>(b);
      return x <= y ? y - x < na : x - y < nb;
   };
   const auto inputBytes = Multiply(batch, fInputBytesPerRow), outputBytes = Multiply(batch, fOutputBytesPerRow);
   if (overlaps(input, inputBytes, output, outputBytes) || overlaps(input, inputBytes, fWorkspace, fBytes) ||
       overlaps(output, outputBytes, fWorkspace, fBytes))
      throw std::invalid_argument("SOFIE GPU: input, output and workspace must not overlap");
   if (auto error = fProgram->invoke(fStream, fDevice, fWorkspace, input, output, batch, fMaxBatch))
      throw std::runtime_error(error);
}
} // namespace TMVA::Experimental::SOFIE
