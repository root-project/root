// @(#)root/base:$Name:  $:$Id: TObjNum.cxx,v 1.1 2002/12/04 12:13:32 rdm Exp $
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
template class TObjNum<Char_t>;
template class TObjNum<UChar_t>;
template class TObjNum<Short_t>;
template class TObjNum<UShort_t>;
template class TObjNum<Int_t>;
template class TObjNum<UInt_t>;
template class TObjNum<Long_t>;
template class TObjNum<ULong_t>;
template class TObjNum<Float_t>;
template class TObjNum<Double_t>;
template class TObjNum<void*>;

templateClassImp(TObjNum)
