// $Id$
// Author: Valeriy Onuchin   12/08/04

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
/**************************************************************************

    This source is based on Xclass95, a Win95-looking GUI toolkit.
    Copyright (C) 1996, 1997 David Barth, Ricky Ralston, Hector Peraza.

    Xclass95 is free software; you can redistribute it and/or
    modify it under the terms of the GNU Library General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) any later version.

**************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TVirtualGuiBld                                                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TVirtualGuiBld.h"


ClassImp(TVirtualGuiBld)

TVirtualGuiBld *gGuiBuilder = 0;

////////////////////////////////////////////////////////////////////////////////
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
