// @(#)root/qt:$Name:  $:$Id: TQtMarker.h,v 1.2 2004/07/28 00:12:40 rdm Exp $
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
#include "Rtypes.h"
#include "TPoint.h"
#ifndef __CINT__
#  include <qpointarray.h>
#else
   class QPointArray;
#endif

////////////////////////////////////////////////////////////////////////
//
// TQtMarker - class-utility to convert the ROOT TMarker object shape 
//             in to the Qt QPointArray.
//
////////////////////////////////////////////////////////////////////////

class TQtMarker {

private:

   int     fNumNode;       // Number of chain in the marker shape
   QPointArray  fChain;    // array of the n chains to build a shaped marker
   Color_t fCindex;        // Color index of the marker;
   int     fMarkerType;    // Type of the current marker

public:

   TQtMarker(int n=0, TPoint *xy=0,int type=0);
   virtual ~TQtMarker();
   int     GetNumber() const;
   QPointArray &GetNodes();
   int     GetType() const;
   void    SetMarker(int n, TPoint *xy, int type);
   ClassDef(TQtMarker,0) //  Convert  ROOT TMarker objects on to QPointArray
};

#endif
