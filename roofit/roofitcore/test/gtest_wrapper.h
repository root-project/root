#ifndef RooFit_gtest_wrapper_h
#define RooFit_gtest_wrapper_h

#include <ROOT/RVersion.hxx>

#include <gtest/gtest.h>

#if ROOT_VERSION_CODE < ROOT_VERSION(6, 32, 99)
// Backward compatibility for gtest version < 1.10.0
#ifndef INSTANTIATE_TEST_SUITE_P
#define INSTANTIATE_TEST_SUITE_P INSTANTIATE_TEST_CASE_P
#endif
#endif

#ifdef ROOFIT_LEGACY_EVAL_BACKEND
#define ROOFIT_EVAL_BACKEND_LEGACY RooFit::EvalBackend::Legacy(),
#else
#define ROOFIT_EVAL_BACKEND_LEGACY
#endif

#ifdef ROOFIT_CUDA
#define ROOFIT_EVAL_BACKEND_CUDA RooFit::EvalBackend::Cuda(),
#else
#define ROOFIT_EVAL_BACKEND_CUDA
#endif

#ifdef ROOFIT_CLAD
#define ROOFIT_EVAL_BACKEND_CODEGEN RooFit::EvalBackend::Codegen(),
#else
#define ROOFIT_EVAL_BACKEND_CODEGEN
#endif

#define ROOFIT_EVAL_BACKENDS ROOFIT_EVAL_BACKEND_LEGACY ROOFIT_EVAL_BACKEND_CUDA RooFit::EvalBackend::Cpu()

#define ROOFIT_EVAL_BACKENDS_WITH_CODEGEN ROOFIT_EVAL_BACKENDS, ROOFIT_EVAL_BACKEND_CODEGEN RooFit::EvalBackend::CodegenNoGrad()

#endif // RooFit_gtest_wrapper_h
