// @(#)root/base:$Name:$:$Id:$
// Author: Fons Rademakers   02/12/02

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TObjNum<T>                                                           //
//                                                                      //
// Wrap basic data type in a TObject.                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TObjNum.h"

// explicit template instantiation of the versions specified in LinkDef.h
template TObjNum<Char_t>;
template TObjNum<UChar_t>;
template TObjNum<Short_t>;
template TObjNum<UShort_t>;
template TObjNum<Int_t>;
template TObjNum<UInt_t>;
template TObjNum<Long_t>;
template TObjNum<ULong_t>;
template TObjNum<Float_t>;
template TObjNum<Double_t>;

templateClassImp(TObjNum)
