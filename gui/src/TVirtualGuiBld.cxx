// $Id: TVirtualGuiBld.cxx,v 1.1 2004/09/08 17:16:09 brun Exp $
// Author: Valeriy Onuchin   12/08/04

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TVirtualGuiBld                                                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TVirtualGuiBld.h"


ClassImp(TVirtualGuiBld)

TVirtualGuiBld *gGuiBuilder = 0;

//______________________________________________________________________________
TVirtualGuiBld::TVirtualGuiBld()
{
   // ctor

}

//______________________________________________________________________________
TVirtualGuiBld::~TVirtualGuiBld()
{
   // dtor

}
