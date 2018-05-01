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
#include "TResultPtr.hxx"
namespace ROOT {
namespace Experimental {
namespace TDF {
template <typename T>
using TResultProxy = TResultPtr<T>;
} // End NS TDF
} // End NS Experimental
} // End NS ROOT

#warning The TResultProxy.hxx header has been replaced by TResultPtr.hxx

#endif
