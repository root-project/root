// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 12/08/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/////////////////////////////////////////////////////////////
// CPU Buffer interface class for the generic data loader. //
/////////////////////////////////////////////////////////////

#ifndef TMVA_DNN_ARCHITECTURES_CPU_CPUBUFFER
#define TMVA_DNN_ARCHITECTURES_CPU_CPUBUFFER

#include "TMVA/DNN/DataLoader.h"
#include <vector>
#include <memory>

namespace TMVA
{
namespace DNN
{

template<typename AReal>
class TCpuBuffer
{
private:

   size_t fSize;
   size_t fOffset;
   std::shared_ptr<AReal *> fBuffer;

   struct TDestructor
   {
       void operator()(AReal ** pointer);
       friend TCpuBuffer;
   } fDestructor;

public:

    TCpuBuffer(size_t size);
    TCpuBuffer(const TCpuBuffer  &) = default;
    TCpuBuffer(      TCpuBuffer &&) = default;
    TCpuBuffer & operator=(const TCpuBuffer  &) = default;
    TCpuBuffer & operator=(      TCpuBuffer &&) = default;

    operator AReal * () const {return (* fBuffer) + fOffset;}

    TCpuBuffer GetSubBuffer(size_t offset, size_t start);

    AReal & operator[](size_t i)       {return (*fBuffer.get())[fOffset + i];}
    AReal   operator[](size_t i) const {return (*fBuffer.get())[fOffset + i];}

    void CopyFrom(TCpuBuffer &);
    void CopyTo(TCpuBuffer &);
};

} // namespace DNN
} // namespace TMVA

#endif

