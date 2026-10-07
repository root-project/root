// Author: Christian Sonnabend, 2026
// For the licensing terms see $ROOTSYS/LICENSE.
#ifndef TMVA_SOFIE_RGPUMODEL
#define TMVA_SOFIE_RGPUMODEL

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace TMVA::Experimental::SOFIE {

/// Experimental sequential Gemm/Relu backend. Generated device code has no ROOT dependency.
class RGPUModel {
public:
   enum class Backend {
      CUDA,
      HIP
   };
   enum class Precision {
      Float32,
      Float16
   };
   struct CompileOptions {
      Backend backend = Backend::CUDA;
      std::string compiler;     ///< Executable path; defaults to nvcc or hipcc.
      std::string architecture; ///< Required: e.g. sm_80 or gfx90a.
   };
   struct Layer {
      std::size_t inputs = 0, outputs = 0;
      std::vector<float> weights, bias; ///< Output-major weights, FP32 accumulation for both precisions.
      float alpha = 1.f, beta = 1.f;
      bool relu = false;
   };
   class Session;

   RGPUModel(Precision precision, std::vector<Layer> layers, std::size_t fixedBatch = 0);
   std::size_t InputSize() const;
   std::size_t OutputSize() const;
   Precision GetPrecision() const { return fPrecision; }
   std::size_t WorkspaceSize(std::size_t maxBatch) const;
   static constexpr std::size_t WorkspaceAlignment() { return 256; }
   std::string GenerateSource(Backend backend) const;
   void Compile(const CompileOptions &options);
   std::unique_ptr<Session> CreateSession(void *stream, int device, std::size_t maxBatch) const;

private:
   struct Program;
   Precision fPrecision;
   std::vector<Layer> fLayers;
   std::vector<float> fWeights;
   std::size_t fFixedBatch, fWidth = 0;
   std::shared_ptr<Program> fProgram;
};

/// Stream, input, output and workspace are borrowed. The caller must complete queued work
/// before reusing their storage or destroying the session. A session is not thread-safe.
class RGPUModel::Session {
public:
   Session(const Session &) = delete;
   Session &operator=(const Session &) = delete;
   ~Session();
   /// Rebind after the framework allocates/recycles its arena; uploads weights on the saved stream.
   void SetWorkspace(void *workspace, std::size_t bytes);
   void Infer(const void *input, void *output, std::size_t batch) const;
   std::size_t WorkspaceSize() const { return fBytes; }

private:
   friend class RGPUModel;
   Session(std::shared_ptr<Program>, std::vector<float>, void *, int, std::size_t, std::size_t, std::size_t,
           std::size_t, std::size_t);
   std::shared_ptr<Program> fProgram;
   std::vector<float> fWeights;
   void *fStream, *fWorkspace = nullptr;
   int fDevice;
   std::size_t fMaxBatch, fFixedBatch, fBytes, fInputBytesPerRow, fOutputBytesPerRow;
};

} // namespace TMVA::Experimental::SOFIE
#endif
