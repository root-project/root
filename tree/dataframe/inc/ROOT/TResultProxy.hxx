// Author: Enrico Guiraud, Danilo Piparo CERN  04/2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TRESULTPROXY
#define ROOT_TRESULTPROXY
#include "RResultPtr.hxx"
namespace ROOT {

namespace RDF {
template <typename T>
using TResultProxy = RResultPtr<T>;
} // End NS RDF

} // End NS ROOT

#warning The TResultProxy.hxx header has been replaced by RResultPtr.hxx

#endif
