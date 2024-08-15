/*
 * Project: RooFit
 * Authors:
 *   Emmanouil Michalainas, CERN 3 March 2021
 *
 * Copyright (c) 2021, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

/**
\file Batches.h
\class Batch
\class Batches
\ingroup roofit_dev_docs_batchcompute

These classes encapsulate the necessary data for the computations.
They are lightweight objects designed to be passed by value and also flexible,
so that they can contain data for every kind of compute function.
**/
#ifndef ROOFIT_BATCHCOMPUTE_BATCHES_H
#define ROOFIT_BATCHCOMPUTE_BATCHES_H

#include <cstdint>

namespace RooBatchCompute {

class Batch {
public:
   const double *__restrict _array = nullptr;
   bool _isVector = false;

#ifdef __CUDACC__
   __device__ constexpr double operator[](std::size_t i) const noexcept { return _isVector ? _array[i] : _array[0]; }
#else
   constexpr double operator[](std::size_t i) const noexcept { return _array[i]; }
#endif // #ifdef __CUDACC__
};

class Batches {
public:
   Batch *args = nullptr;
   double *extra;
   std::size_t nEvents = 0;
   std::size_t nBatches = 0;
   std::size_t nExtra = 0;
   double *__restrict output = nullptr;
};

} // end namespace RooBatchCompute

#endif // #ifdef ROOFIT_BATCHCOMPUTE_BATCHES_H
