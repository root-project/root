// @(#)root/gpad:$Name$:$Id$
// Author: Rene Brun   12/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TAttCanvas
#define ROOT_TAttCanvas


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TAttCanvas                                                           //
//                                                                      //
// Canvas attributes.                                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_Gtypes
#include "Gtypes.h"
#endif

#ifndef ROOT_Htypes
#include "Htypes.h"
#endif


class TAttCanvas {
private:
        Coord_t    fXBetween;        //X distance between pads
        Coord_t    fYBetween;        //Y distance between pads
        Coord_t    fTitleFromTop;    //Y distance of Global Title from top
        Coord_t    fXdate;           //X position where to draw the date
        Coord_t    fYdate;           //X position where to draw the date
        Coord_t    fAdate;           //Alignment for the date

public:
        TAttCanvas();
        virtual ~TAttCanvas();
        virtual void     Copy(TAttCanvas &attcanvas);
        Coord_t          GetAdate() { return fAdate;}
        Coord_t          GetTitleFromTop() { return fTitleFromTop;}
        Coord_t          GetXBetween() { return fXBetween;}
        Coord_t          GetXdate() { return fXdate;}
        Coord_t          GetYBetween() { return fYBetween;}
        Coord_t          GetYdate() { return fYdate;}
        virtual void     Print(Option_t *option="");
        virtual void     ResetAttCanvas(Option_t *option="");
        virtual void     SetAdate(Coord_t adate) { fAdate=adate;}
        virtual void     SetTitleFromTop(Coord_t titlefromtop)
                                        { fTitleFromTop=titlefromtop;}
        virtual void     SetXBetween(Coord_t xbetween) { fXBetween=xbetween;}
        virtual void     SetXdate(Coord_t xdate) { fXdate=xdate;}
        virtual void     SetYBetween(Coord_t ybetween) { fYBetween=ybetween;}
        virtual void     SetYdate(Coord_t ydate) { fYdate=ydate;}

        ClassDef(TAttCanvas,1)  //Canvas attributes
};

#endif

