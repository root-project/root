// @(#)root/graf:$Name$:$Id$
// Author: Rene Brun   12/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TLine
#define ROOT_TLine


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TLine                                                                //
//                                                                      //
// A line segment.                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif

#ifndef ROOT_TAttLine
#include "TAttLine.h"
#endif


class TLine : public TObject, public TAttLine {

protected:
        Coord_t      fX1;           //X of 1st point
        Coord_t      fY1;           //Y of 1st point
        Coord_t      fX2;           //X of 2nd point
        Coord_t      fY2;           //Y of 2nd point

public:
        // TLine status bits
        enum { kLineNDC = BIT(14) };

        TLine();
        TLine(Coord_t x1, Coord_t y1,Coord_t x2, Coord_t  y2);
        TLine(const TLine &line);
        virtual ~TLine();
                void   Copy(TObject &line);
        virtual Int_t  DistancetoPrimitive(Int_t px, Int_t py);
        virtual TLine *DrawLine(Coord_t x1, Coord_t y1,Coord_t x2, Coord_t y2);
        virtual TLine *DrawLineNDC(Coord_t x1, Coord_t y1,Coord_t x2, Coord_t y2);
        virtual void   ExecuteEvent(Int_t event, Int_t px, Int_t py);
        Coord_t        GetX1() {return fX1;}
        Coord_t        GetX2() {return fX2;}
        Coord_t        GetY1() {return fY1;}
        Coord_t        GetY2() {return fY2;}
        virtual void   ls(Option_t *option="");
        virtual void   Paint(Option_t *option="");
        virtual void   PaintLine(Coord_t x1, Coord_t y1,Coord_t x2, Coord_t  y2);
        virtual void   PaintLineNDC(Coord_t u1, Coord_t v1,Coord_t u2, Coord_t  v2);
        virtual void   Print(Option_t *option="");
        virtual void   SavePrimitive(ofstream &out, Option_t *option);
        virtual void   SetX1(Coord_t x1) {fX1=x1;}
        virtual void   SetX2(Coord_t x2) {fX2=x2;}
        virtual void   SetY1(Coord_t y1) {fY1=y1;}
        virtual void   SetY2(Coord_t y2) {fY2=y2;}

        ClassDef(TLine,1)  //A line segment
};

#endif
