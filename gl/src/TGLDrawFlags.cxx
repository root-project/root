// @(#)root/gl:$Name:  $:$Id: $
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
TGLDrawFlags::TGLDrawFlags(EStyle style, Short_t LOD) :
   fStyle(style), fLOD(LOD)
{
}

//______________________________________________________________________________
TGLDrawFlags::~TGLDrawFlags()
{
}
