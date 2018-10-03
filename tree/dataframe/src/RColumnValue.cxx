// Author: Enrico Guiraud, Danilo Piparo CERN  09/2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RDF/RColumnValue.hxx"

#include <vector>

namespace ROOT {
namespace Internal {
namespace RDF {
// Some extern instaniations to speed-up compilation/interpretation time
// These are not active if c++17 is enabled because of a bug in our clang
// See ROOT-9499.
#if __cplusplus < 201703L
template class RColumnValue<int>;
template class RColumnValue<unsigned int>;
template class RColumnValue<char>;
template class RColumnValue<unsigned char>;
template class RColumnValue<float>;
template class RColumnValue<double>;
template class RColumnValue<Long64_t>;
template class RColumnValue<ULong64_t>;
template class RColumnValue<std::vector<int>>;
template class RColumnValue<std::vector<unsigned int>>;
template class RColumnValue<std::vector<char>>;
template class RColumnValue<std::vector<unsigned char>>;
template class RColumnValue<std::vector<float>>;
template class RColumnValue<std::vector<double>>;
template class RColumnValue<std::vector<Long64_t>>;
template class RColumnValue<std::vector<ULong64_t>>;
#endif
} // ns RDF
} // ns Internal
} // ns ROOT
