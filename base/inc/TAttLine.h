// @(#)root/base:$Name:  $:$Id: TAttLine.h,v 1.1.1.1 2000/05/16 17:00:39 rdm Exp $
// Author: Rene Brun   28/11/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TAttLine
#define ROOT_TAttLine


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TAttLine                                                             //
//                                                                      //
// Line attributes.                                                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_Gtypes
#include "Gtypes.h"
#endif


class TAttLine {

protected:
        Color_t    fLineColor;           //line color
        Style_t    fLineStyle;           //line style
        Width_t    fLineWidth;           //line width

public:
        TAttLine();
        TAttLine(Color_t lcolor,Style_t lstyle, Width_t lwidth);
        virtual ~TAttLine();
                void     Copy(TAttLine &attline);
        Int_t            DistancetoLine(Int_t px, Int_t py, Double_t xp1, Double_t yp1, Double_t xp2, Double_t yp2 );
        Color_t          GetLineColor() {return fLineColor;}
        Style_t          GetLineStyle() {return fLineStyle;}
        Width_t          GetLineWidth() {return fLineWidth;}
        virtual void     Modify();
        virtual void     ResetAttLine(Option_t *option="");
        virtual void     SaveLineAttributes(ofstream &out, const char *name, Int_t coldef=1, Int_t stydef=1, Int_t widdef=1);
        virtual void     SetLineAttributes(); // *MENU*
        virtual void     SetLineColor(Color_t lcolor) { fLineColor = lcolor;}
        virtual void     SetLineStyle(Style_t lstyle) { fLineStyle = lstyle;}
        virtual void     SetLineWidth(Width_t lwidth) { fLineWidth = lwidth;}

        ClassDef(TAttLine,1)  //Line attributes
};

#endif

