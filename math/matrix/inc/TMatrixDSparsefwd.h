/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMatrixDSparsefwd
#define ROOT_TMatrixDSparsefwd

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMatrixDSparse                                                       //
//                                                                      //
//  Forward declaration of TMatrixTSparse<Double_t>                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "RtypesCore.h"

template<class Element> class TMatrixTSparse;
typedef TMatrixTSparse<Double_t> TMatrixDSparse;

#endif
