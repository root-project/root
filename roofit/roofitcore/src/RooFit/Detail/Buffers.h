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

struct BufferQueuesMaps;

class BufferManager {

public:
   BufferManager();
   ~BufferManager();

   AbsBuffer *makeScalarBuffer();
   AbsBuffer *makeCpuBuffer(std::size_t size);
   AbsBuffer *makeGpuBuffer(std::size_t size);
   AbsBuffer *makePinnedBuffer(std::size_t size, cudaStream_t *stream = nullptr);

private:
   BufferQueuesMaps *_queuesMaps;
};

} // end namespace Detail
} // end namespace Experimental
} // end namespace ROOT

#endif
