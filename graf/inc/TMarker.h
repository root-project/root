// @(#)root/graf:$Name:  $:$Id: TMarker.h,v 1.1.1.1 2000/05/16 17:00:50 rdm Exp $
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
        Double_t     fX;           //X position of marker (left,center,etc..)
        Double_t     fY;           //Y position of marker (left,center,etc..)

public:
        TMarker();
        TMarker(Double_t x, Double_t y, Int_t marker);
        TMarker(const TMarker &marker);
        virtual ~TMarker();
                void     Copy(TObject &marker);
        virtual Int_t    DistancetoPrimitive(Int_t px, Int_t py);
        virtual void     Draw(Option_t *option="");
        virtual void     DrawMarker(Double_t x, Double_t y);
        virtual void     ExecuteEvent(Int_t event, Int_t px, Int_t py);
        Double_t         GetX()  {return fX;}
        Double_t         GetY()  {return fY;}
        virtual void     ls(Option_t *option="");
        virtual void     Paint(Option_t *option="");
        virtual void     PaintMarker(Double_t x, Double_t y);
        virtual void     PaintMarkerNDC(Double_t u, Double_t v);
        virtual void     Print(Option_t *option="");
        virtual void     SavePrimitive(ofstream &out, Option_t *option);
        virtual void     SetX(Double_t x) { fX = x;} // *MENU*
        virtual void     SetY(Double_t y) { fY = y;} // *MENU*

        static  void     DisplayMarkerTypes();

        ClassDef(TMarker,2)  //Marker
};

#endif

