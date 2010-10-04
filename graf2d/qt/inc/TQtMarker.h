// @(#)root/qt:$Id$
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

#ifndef ROOT_TPoint
#include "TPoint.h"
#endif

#ifndef __CINT__
#include <QPolygon>
#else
   class QPointArray;
   class QPolygon;
#endif

class TAttMarker;
class QPainter;

////////////////////////////////////////////////////////////////////////
//
// TQtMarker - class-utility to convert the ROOT TMarker object shape 
//             in to the Qt QPolygon.
//
////////////////////////////////////////////////////////////////////////

class TQtMarker {

private:

   int     fNumNode;       // Number of chain in the marker shape
   QPolygon  fChain;       // array of the n chains to build a shaped marker
   Color_t fCindex;        // Color index of the marker;
   int     fMarkerType;    // Type of the current marker
   int     fLineWidth;     // The width of the line used to outline the markers
   int     fLineOption;    // Connect the markers with the segments

public:

   TQtMarker(int n=0, TPoint *xy=0,int type=0);
   TQtMarker &operator=(const TQtMarker&);
   TQtMarker(const TQtMarker&);
   TQtMarker &operator=(const TAttMarker&);
   TQtMarker(const TAttMarker&);
   virtual ~TQtMarker();
	void  DrawPolyMarker(QPainter &p, int n, TPoint *xy);
	void  SetMarkerAttributes(const TAttMarker& markerAttributes);
	void  SetColor(Color_t mcolor);
   void  SetPenAttributes(int type);
	Color_t  GetColor() const;

   int     GetNumber()  const;
   const   QPolygon &GetNodes() const;
   int     GetType()    const;
   int     GetWidth()   const;
   void    SetMarker(int n, TPoint *xy, int type);
   ClassDef(TQtMarker,0) //  Convert  ROOT TMarker objects on to QPointArray
};

//_________________________________________________________
inline TQtMarker &TQtMarker::operator=(const TQtMarker&m) 
{
   fNumNode  = m.fNumNode;
   fChain    = m.fChain; 
   fCindex   = m.fCindex;
   fMarkerType=m.fMarkerType;
   fLineWidth =m.fLineWidth;
	return *this;
}
//_________________________________________________________
inline TQtMarker::TQtMarker(const TQtMarker&m) : fNumNode(m.fNumNode),
fChain(m.fChain), fCindex(m.fCindex),fMarkerType(m.fMarkerType),fLineWidth(m.fLineWidth) 
, fLineOption()
{}

//_________________________________________________________
inline void  TQtMarker::SetColor(Color_t mcolor) { fCindex = mcolor; }
//_________________________________________________________
inline Color_t  TQtMarker::GetColor() const { return fCindex; }

#endif
