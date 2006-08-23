// @(#)root/gl:$Name:  $:$Id: TGLDrawFlags.cxx,v 1.1 2006/02/08 10:49:26 couet Exp $
// Author:  Richard Maunder  27/01/2006

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGLDrawFlags.h"

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLDrawFlags                                                          //      
//                                                                      //
//////////////////////////////////////////////////////////////////////////

ClassImp(TGLDrawFlags)

//______________________________________________________________________________
TGLDrawFlags::TGLDrawFlags(EStyle style, Short_t LOD, Bool_t sel, Bool_t secSel) :
   fStyle(style), fLOD(LOD),
   fSelection(sel), fSecSelection(secSel)
{
}

//______________________________________________________________________________
TGLDrawFlags::~TGLDrawFlags()
{
}
