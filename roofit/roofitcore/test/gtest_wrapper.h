#ifndef RooFit_gtest_wrapper_h
#define RooFit_gtest_wrapper_h

#include <gtest/gtest.h>

// Backward compatibility for gtest version < 1.10.0
#ifndef INSTANTIATE_TEST_SUITE_P
#define INSTANTIATE_TEST_SUITE_P INSTANTIATE_TEST_CASE_P
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

#define ROOFIT_EVAL_BACKENDS ROOFIT_EVAL_BACKEND_LEGACY ROOFIT_EVAL_BACKEND_CUDA RooFit::EvalBackend::Cpu()

#endif // RooFit_gtest_wrapper_h
