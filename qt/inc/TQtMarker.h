// Author: Valeri Fine   21/01/2002
/****************************************************************************
** $Id: TQtMarker.h,v 1.1.1.1 2002/03/27 18:17:02 fine Exp $
**
** Copyright (C) 2002 by Valeri Fine.  All rights reserved.
**
** This file may be distributed under the terms of the Q Public License
** as defined by Trolltech AS of Norway and appearing in the file
** LICENSE.QPL included in the packaging of this file.
*****************************************************************************/

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
