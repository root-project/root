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

#ifndef ROOFIT_BATCHCOMPUTE_ROOBATCHCOMPUTETYPES_H
#define ROOFIT_BATCHCOMPUTE_ROOBATCHCOMPUTETYPES_H

#include <RooBatchCompute/DataKey.h>
#include <RooSpan.h>

#include <map>
#include <vector>


#ifdef __CUDACC__
#define __roodevice__ __device__
#define __roohost__ __host__
#define __rooglobal__ __global__
#else
#define __roodevice__
#define __roohost__
#define __rooglobal__
struct cudaEvent_t;
struct cudaStream_t;
#endif // #indef __CUDACC__

namespace RooBatchCompute {

struct RunContext;

// We have to use map instead of unordered_map because the unordered_maps from
// nvcc and gcc are not compatible sometimes.
typedef std::map<DataKey, RooSpan<const double>> DataMap;
typedef std::vector<DataKey> VarVector;
typedef std::vector<double> ArgVector;
typedef double *__restrict RestrictArr;
typedef const double *__restrict InputArr;

}

#endif
