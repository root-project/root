// @(#)root/graf:$Name$:$Id$
// Author: Rene Brun   12/05/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMarker
#define ROOT_TMarker


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMarker                                                              //
//                                                                      //
// Marker.                                                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TAttMarker
#include "TAttMarker.h"
#endif

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif


class TMarker : public TObject, public TAttMarker {

protected:
        Coord_t     fX;           //X position of marker (left,center,etc..)
        Coord_t     fY;           //Y position of marker (left,center,etc..)

public:
        TMarker();
        TMarker(Coord_t x, Coord_t y, Int_t marker);
        TMarker(const TMarker &marker);
        virtual ~TMarker();
                void     Copy(TObject &marker);
        virtual Int_t    DistancetoPrimitive(Int_t px, Int_t py);
        virtual void     Draw(Option_t *option="");
        virtual void     DrawMarker(Coord_t x, Coord_t y);
        virtual void     ExecuteEvent(Int_t event, Int_t px, Int_t py);
        Coord_t          GetX()  {return fX;}
        Coord_t          GetY()  {return fY;}
        virtual void     ls(Option_t *option="");
        virtual void     Paint(Option_t *option="");
        virtual void     PaintMarker(Coord_t x, Coord_t y);
        virtual void     PaintMarkerNDC(Coord_t u, Coord_t v);
        virtual void     Print(Option_t *option="");
        virtual void     SavePrimitive(ofstream &out, Option_t *option);
        virtual void     SetX(Coord_t x) { fX = x;} // *MENU*
        virtual void     SetY(Coord_t y) { fY = y;} // *MENU*

        static  void     DisplayMarkerTypes();

        ClassDef(TMarker,1)  //Marker
};

#endif

