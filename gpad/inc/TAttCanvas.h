// @(#)root/gpad:$Name:  $:$Id: TAttCanvas.h,v 1.1.1.1 2000/05/16 17:00:41 rdm Exp $
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
        Float_t    fXBetween;        //X distance between pads
        Float_t    fYBetween;        //Y distance between pads
        Float_t    fTitleFromTop;    //Y distance of Global Title from top
        Float_t    fXdate;           //X position where to draw the date
        Float_t    fYdate;           //X position where to draw the date
        Float_t    fAdate;           //Alignment for the date

public:
        TAttCanvas();
        virtual ~TAttCanvas();
        virtual void     Copy(TAttCanvas &attcanvas);
        Float_t          GetAdate() { return fAdate;}
        Float_t          GetTitleFromTop() { return fTitleFromTop;}
        Float_t          GetXBetween() { return fXBetween;}
        Float_t          GetXdate() { return fXdate;}
        Float_t          GetYBetween() { return fYBetween;}
        Float_t          GetYdate() { return fYdate;}
        virtual void     Print(Option_t *option="");
        virtual void     ResetAttCanvas(Option_t *option="");
        virtual void     SetAdate(Float_t adate) { fAdate=adate;}
        virtual void     SetTitleFromTop(Float_t titlefromtop)
                                        { fTitleFromTop=titlefromtop;}
        virtual void     SetXBetween(Float_t xbetween) { fXBetween=xbetween;}
        virtual void     SetXdate(Float_t xdate) { fXdate=xdate;}
        virtual void     SetYBetween(Float_t ybetween) { fYBetween=ybetween;}
        virtual void     SetYdate(Float_t ydate) { fYdate=ydate;}

        ClassDef(TAttCanvas,1)  //Canvas attributes
};

#endif

