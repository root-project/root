/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMatrixFSparsefwd
#define ROOT_TMatrixFSparsefwd

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMatrixFSparse                                                       //
//                                                                      //
//  Forward declaration of TMatrixTSparse<Float_t>                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

template<class Element> class TMatrixTSparse;
typedef TMatrixTSparse<Float_t> TMatrixFSparse;

#endif
