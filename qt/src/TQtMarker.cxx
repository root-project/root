// @(#)root/qt:$Name:$:$Id:$
// Author: Valeri Fine   21/01/2002

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * Copyright (C) 2002 by Valeri Fine.                                    *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TQtRConfig.h"
#include "TQtMarker.h"

//______________________________________________________________________________
TQtMarker::TQtMarker(int n, TPoint *xy, int type) : fNumNode(n),
               fChain(0), fCindex(0), fMarkerType(type)
{
  if (type >= 2) {
#ifdef R__QTWIN32
     fChain.setPoints(n,(QCOORD *)xy);
#else
     fChain.resize(n);
     TPoint *rootPoint = xy;
     for (int i=0;i<n;i++,rootPoint++)
        fChain.setPoint(i,rootPoint->fX,rootPoint->fY);
#endif
  }
}
//______________________________________________________________________________
TQtMarker::~TQtMarker(){}
//______________________________________________________________________________
int    TQtMarker::GetNumber() const {return fNumNode;}
//______________________________________________________________________________
QPointArray &TQtMarker::GetNodes() {return fChain;}
//______________________________________________________________________________
int  TQtMarker::GetType() const {return fMarkerType;}

//______________________________________________________________________________
void TQtMarker::SetMarker(int n, TPoint *xy, int type)
{
//*-* Did we have a chain ?
  fNumNode = n;
  fMarkerType = type;
  if (fMarkerType >= 2) {
#ifdef R__QTWIN32
    fChain.setPoints(n,(QCOORD *)xy);
#else
    fChain.resize(n);
    TPoint *rootPoint = xy;
    for (int i=0;i<n;i++,rootPoint++)
       fChain.setPoint(i,rootPoint->fX,rootPoint->fY);
#endif

  }
}
