// @(#)root/base:$Name$:$Id$
// Author: Fons Rademakers   04/05/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TObjPtr                                                              //
//                                                                      //
// Collectable generic pointer class. This is a TObject containing a    //
// void *.                                                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TObjPtr.h"

//ClassImp(TObjPtr)

Int_t TObjPtr::Compare(TObject *obj)
{
   if (fPtr == obj)
      return 0;
   else if (fPtr < obj)
      return -1;
   else
      return 1;
}
