// @(#)root/base:$Id$
// Author: Fons Rademakers   04/08/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TRefCnt                                                             //
//                                                                      //
//  Definitions for TRefCnt, base class for reference counted objects.  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#include "TRefCnt.h"


// This definition is compiled in case nothing else is,
// in order to quiet down some fussy librarians
int gDummy_ref_cpp;

//_______________________________________________________________________
TRefCnt::TRefCnt(EReferenceFlag)
{
  // Leave fRefs alone
}
