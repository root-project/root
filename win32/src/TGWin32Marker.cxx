// @(#)root/win32:$Name:  $:$Id: TGWin32Marker.cxx,v 1.2 2001/06/27 15:58:15 brun Exp $
// Author: Valery Fine   27/01/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGWin32Marker.h"

//______________________________________________________________________________
TGWin32Marker::TGWin32Marker(int n, TPoint *xy, int type) : fNumNode(n),
               fChain(0), fCindex(0), fMarkerType(type)
{
  fNumNode = n;
  fMarkerType = type;
  if (type >= 2) {
     if (fChain = new POINT[n]) {
        for( int i = 0; i < n; i++ ) {
           fChain[i].x = xy[i].GetX();
           fChain[i].y = xy[i].GetY();
        }
     }
  }
}
//______________________________________________________________________________
TGWin32Marker::~TGWin32Marker(){
//*-*-*-*-*-*-*-*-*-*-*-*Default Destructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                    ==================
  if (fChain) delete fChain;
}

//______________________________________________________________________________
int    TGWin32Marker::GetNumber(){return fNumNode;}
//______________________________________________________________________________
POINT *TGWin32Marker::GetNodes(){return fChain;}
//______________________________________________________________________________
int  TGWin32Marker::GetType(){return fMarkerType;}

//______________________________________________________________________________
void TGWin32Marker::SetMarker(int n, TPoint *xy, int type){

//*-* Did we have a chain ?

if (fMarkerType >= 2 && fNumNode != n){    // Yes, we had chain
       if (fChain) delete fChain;  // Delete the obsolete chain
       fChain = NULL;
}

//*-*  Create the new shaped marker

if (type >= 2) {
    if (!fChain) fChain = new POINT[n];
    for( int i = 0; i < n; i++ ) {
      fChain[i].x = xy[i].GetX();
      fChain[i].y = xy[i].GetY();
    }
}
else if (fChain) { delete fChain; fChain = NULL; }

fNumNode = n;
fMarkerType = type;

}
