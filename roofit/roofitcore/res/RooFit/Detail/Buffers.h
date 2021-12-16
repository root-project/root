/*
 * Project: RooFit
 * Authors:
 *   Jonas Rembser, CERN  11/2021
 *
 * Copyright (c) 2021, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef RooFit_Detail_Buffers_h
#define RooFit_Detail_Buffers_h

#include <RooBatchComputeTypes.h>

#include <memory>

namespace ROOT {
namespace Experimental {
namespace Detail {

class AbsBuffer {
public:
   virtual ~AbsBuffer() = default;

   virtual double const *cpuReadPtr() const = 0;
   virtual double const *gpuReadPtr() const = 0;

   virtual double *cpuWritePtr() = 0;
   virtual double *gpuWritePtr() = 0;
};

std::unique_ptr<AbsBuffer> makeScalarBuffer();
std::unique_ptr<AbsBuffer> makeCpuBuffer(std::size_t size);
std::unique_ptr<AbsBuffer> makeGpuBuffer(std::size_t size);
std::unique_ptr<AbsBuffer> makePinnedBuffer(std::size_t size, cudaStream_t *stream = nullptr);

} // end namespace Detail
} // end namespace Experimental
} // end namespace ROOT

#endif
