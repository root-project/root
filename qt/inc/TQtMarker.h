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

#ifndef ROOT_TQtMarker
#define ROOT_TQtMarker

#include "Gtypes.h"
#include "TPoint.h"
#include "qpoint.h"
#include "qpointarray.h"


class TQtMarker {

private:

   int     fNumNode;       // Number of chain in the marker shape
   QPointArray  fChain; // array of the n chains to build a shaped marker
   Color_t fCindex;        // Color index of the marker;
   int     fMarkerType;    // Type of the current marker

public:

   TQtMarker(int n=0, TPoint *xy=0,int type=0);
  ~TQtMarker();
   int     GetNumber() const;
   QPointArray &GetNodes();
   int     GetType() const;
   void    SetMarker(int n, TPoint *xy, int type);

};

#endif
