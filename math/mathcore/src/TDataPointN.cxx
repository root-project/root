// @(#)root/mathcore:$Id$
// Authors: C. Gumpert    09/2011

#include "Math/TDataPointN.h"

namespace ROOT {
namespace Math {

template class TDataPointN<float>;
template <> UInt_t TDataPointN<float>::kDimension = 0;

template class TDataPointN<double>;
template <> UInt_t TDataPointN<double>::kDimension = 0;

} //namespace Math
} //namespace ROOT
