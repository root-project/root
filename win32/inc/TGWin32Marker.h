// @(#)root/win32:$Name$:$Id$
// Author: Valery Fine   27/01/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TGWin32Marker
#define ROOT_TGWin32Marker

#include "Gtypes.h"
#include "TPoint.h"
#include "Windows4Root.h"

class TGWin32Marker {

private:

   int     fNumNode;    // Number of chain in the marker shape
   POINT  *fChain;      // List of the n chains to build a shaped marker
   Color_t fCindex;     // Color index of the marker;
   int     fMarkerType; // Type of the current marker

public:

   TGWin32Marker(int n=0, TPoint *xy=0,int type=0);
  ~TGWin32Marker();
   int    GetNumber();
   POINT *GetNodes();
   int    GetType();
   void   SetMarker(int n, TPoint *xy, int type);

};

#endif
