// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEvePointSetProjectedGL.h"

//______________________________________________________________________________
// TEvePointSetProjectedGL
//
// GL-renderer for TEvePointSetProjected class.
//
// A hack around a bug in fglrx that makes rendering of projected pointsets
// terribly slow with display-lists on when rendering as crosses.

ClassImp(TEvePointSetProjectedGL)

//______________________________________________________________________________
TEvePointSetProjectedGL::TEvePointSetProjectedGL() : TPointSet3DGL()
{
   // Contructor.
   fDLCache = kFALSE; // Disable display list.
}

//______________________________________________________________________________
TEvePointSetProjectedGL::~TEvePointSetProjectedGL()
{}
