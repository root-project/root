/*
 * Project: RooFit
 * Authors:
 *   Emmanouil Michalainas, CERN 6 January 2021
 *
 * Copyright (c) 2021, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef ROOFIT_BATCHCOMPUTE_ROOBATCHCOMPUTE_H
#define ROOFIT_BATCHCOMPUTE_ROOBATCHCOMPUTE_H

#include <ROOT/RSpan.hxx>

#include <DllImport.h> //for R__EXTERN, needed for windows

#include <cstddef>
#include <initializer_list>
#include <memory>
#include <string>

/**
 * Namespace for dispatching RooFit computations to various backends.
 *
 * This namespace contains an interface for providing high-performance computation functions for use in
 * RooAbsReal::doEval(), see RooBatchComputeInterface.
 *
 * Furthermore, several implementations of this interface can be created, which reside in RooBatchCompute::RF_ARCH,
 * where RF_ARCH may be replaced by the architecture that this implementation targets, e.g. SSE, AVX, etc.
 *
 * Using the pointer RooBatchCompute::dispatch, a computation request can be dispatched to the fastest backend that is
 * available on a specific platform.
 */
namespace RooBatchCompute {

namespace CudaInterface {
class CudaEvent;
class CudaStream;
} // namespace CudaInterface

typedef std::span<const std::span<const double>> VarSpan;
typedef std::span<double> ArgSpan;
typedef const double *__restrict InputArr;

constexpr std::size_t bufferSize = 64;

int initCPU();
int initCUDA();

/// Minimal configuration struct to steer the evaluation of a single node with
/// the RooBatchCompute library.
class Config {
public:
   bool useCuda() const { return _cudaStream != nullptr; }
   void setCudaStream(CudaInterface::CudaStream *cudaStream) { _cudaStream = cudaStream; }
   CudaInterface::CudaStream *cudaStream() const { return _cudaStream; }

private:
   CudaInterface::CudaStream *_cudaStream = nullptr;
};

enum class Architecture {
   AVX512,
   AVX2,
   AVX,
   SSE4,
   GENERIC,
   CUDA
};

enum Computer {
   AddPdf,
   ArgusBG,
   BMixDecay,
   Bernstein,
   BifurGauss,
   BreitWigner,
   Bukin,
   CBShape,
   Chebychev,
   ChiSquare,
   DeltaFunction,
   DstD0BG,
   ExpPoly,
   Exponential,
   ExponentialNeg,
   Gamma,
   GaussModelExpBasis,
   Gaussian,
   Identity,
   Johnson,
   Landau,
   Lognormal,
   LognormalStandard,
   NegativeLogarithms,
   NormalizedPdf,
   Novosibirsk,
   Poisson,
   Polynomial,
   Power,
   ProdPdf,
   Ratio,
   TruthModelExpBasis,
   TruthModelSinBasis,
   TruthModelCosBasis,
   TruthModelLinBasis,
   TruthModelQuadBasis,
   TruthModelSinhBasis,
   TruthModelCoshBasis,
   Voigtian
};

struct ReduceNLLOutput {
   double nllSum = 0.0;
   double nllSumCarry = 0.0;
   std::size_t nInfiniteValues = 0;
   std::size_t nNonPositiveValues = 0;
   std::size_t nNaNValues = 0;
};

class AbsBuffer {
public:
   virtual ~AbsBuffer() = default;

   virtual double const *hostReadPtr() const = 0;
   virtual double const *deviceReadPtr() const = 0;

   virtual double *hostWritePtr() = 0;
   virtual double *deviceWritePtr() = 0;

   virtual void assignFromHost(std::span<const double> input) = 0;
   virtual void assignFromDevice(std::span<const double> input) = 0;
};

class AbsBufferManager {
public:
   virtual ~AbsBufferManager() = default;

   virtual std::unique_ptr<AbsBuffer> makeScalarBuffer() = 0;
   virtual std::unique_ptr<AbsBuffer> makeCpuBuffer(std::size_t size) = 0;
   virtual std::unique_ptr<AbsBuffer> makeGpuBuffer(std::size_t size) = 0;
   virtual std::unique_ptr<AbsBuffer>
   makePinnedBuffer(std::size_t size, CudaInterface::CudaStream *stream = nullptr) = 0;
};

/**
 * \class RooBatchComputeInterface
 * \ingroup roofit_dev_docs_batchcompute
 * \brief The interface which should be implemented to provide optimised computation functions for implementations of
 * RooAbsReal::doEval().
 *
 * The class RooBatchComputeInterface provides the mechanism for external modules (like RooFit) to call
 * functions from the library. The power lies in the virtual functions that can resolve to different
 * implementations for the functionality; for example, calling a function through dispatchCuda
 * will resolve to efficient CUDA implementations.
 *
 * This interface contains the signatures of the compute functions of every PDF that has an optimised implementation
 * available. These are the functions that perform the actual computations in batches.
 *
 * Several implementations of this interface may be provided, e.g. SSE, AVX, AVX2 etc. At run time, the fastest
 * implementation of this interface is selected, and using a virtual call, the computation is dispatched to the best
 * backend.
 *
 * \see RooBatchCompute::dispatch, RooBatchComputeClass, RF_ARCH
 */
class RooBatchComputeInterface {
public:
   virtual ~RooBatchComputeInterface() = default;
   virtual void compute(Config const &cfg, Computer, std::span<double> output, VarSpan, ArgSpan) = 0;

   virtual double reduceSum(Config const &cfg, InputArr input, size_t n) = 0;
   virtual ReduceNLLOutput reduceNLL(Config const &cfg, std::span<const double> probas, std::span<const double> weights,
                                     std::span<const double> offsetProbas) = 0;

   virtual Architecture architecture() const = 0;
   virtual std::string architectureName() const = 0;

   virtual std::unique_ptr<AbsBufferManager> createBufferManager() const = 0;

   virtual CudaInterface::CudaEvent *newCudaEvent(bool forTiming) const = 0;
   virtual CudaInterface::CudaStream *newCudaStream() const = 0;
   virtual void deleteCudaEvent(CudaInterface::CudaEvent *) const = 0;
   virtual void deleteCudaStream(CudaInterface::CudaStream *) const = 0;
   virtual void cudaEventRecord(CudaInterface::CudaEvent *, CudaInterface::CudaStream *) const = 0;
   virtual void cudaStreamWaitForEvent(CudaInterface::CudaStream *, CudaInterface::CudaEvent *) const = 0;
   virtual bool cudaStreamIsActive(CudaInterface::CudaStream *) const = 0;
};

/**
 * This dispatch pointer points to an implementation of the compute library, provided one has been loaded.
 * Using a virtual call, computation requests are dispatched to backends with architecture-specific functions
 * such as SSE, AVX, AVX2, etc.
 *
 * \see RooBatchComputeInterface, RooBatchComputeClass, RF_ARCH
 */
R__EXTERN RooBatchComputeInterface *dispatchCPU;
R__EXTERN RooBatchComputeInterface *dispatchCUDA;

inline Architecture cpuArchitecture()
{
   return dispatchCPU->architecture();
}

inline std::string cpuArchitectureName()
{
   return dispatchCPU->architectureName();
}

inline void compute(Config cfg, Computer comp, std::span<double> output, VarSpan vars, ArgSpan extraArgs = {})
{
   auto dispatch = cfg.useCuda() ? dispatchCUDA : dispatchCPU;
   dispatch->compute(cfg, comp, output, vars, extraArgs);
}

/// It is not possible to construct a std::span directly from an initializer
/// list (probably it will be with C++26). That's why we need an explicit
/// overload for this.
inline void compute(Config cfg, Computer comp, std::span<double> output,
                    std::initializer_list<std::span<const double>> vars, ArgSpan extraArgs = {})
{
   compute(cfg, comp, output, VarSpan{vars.begin(), vars.end()}, extraArgs);
}

inline double reduceSum(Config cfg, InputArr input, size_t n)
{
   auto dispatch = cfg.useCuda() ? dispatchCUDA : dispatchCPU;
   return dispatch->reduceSum(cfg, input, n);
}

inline ReduceNLLOutput reduceNLL(Config cfg, std::span<const double> probas, std::span<const double> weights,
                                 std::span<const double> offsetProbas)
{
   auto dispatch = cfg.useCuda() ? dispatchCUDA : dispatchCPU;
   return dispatch->reduceNLL(cfg, probas, weights, offsetProbas);
}

std::string getBatchComputeChoice();
void setBatchComputeChoice(std::string const &value);

} // End namespace RooBatchCompute

#endif
