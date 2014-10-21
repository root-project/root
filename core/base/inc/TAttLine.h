// @(#)root/base:$Id$
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

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif
#ifndef ROOT_Riosfwd
#include "Riosfwd.h"
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

   void             Copy(TAttLine &attline) const;
   Int_t            DistancetoLine(Int_t px, Int_t py, Double_t xp1, Double_t yp1, Double_t xp2, Double_t yp2 );
   virtual Color_t  GetLineColor() const {return fLineColor;}
   virtual Style_t  GetLineStyle() const {return fLineStyle;}
   virtual Width_t  GetLineWidth() const {return fLineWidth;}
   virtual void     Modify();
   virtual void     ResetAttLine(Option_t *option="");
   virtual void     SaveLineAttributes(std::ostream &out, const char *name, Int_t coldef=1, Int_t stydef=1, Int_t widdef=1);
   virtual void     SetLineAttributes(); // *MENU*
   virtual void     SetLineColor(Color_t lcolor) { fLineColor = lcolor;}
   virtual void     SetLineColorAlpha(Color_t lcolor, Float_t lalpha);
   virtual void     SetLineStyle(Style_t lstyle) { fLineStyle = lstyle;}
   virtual void     SetLineWidth(Width_t lwidth) { fLineWidth = lwidth;}

   ClassDef(TAttLine,2);  //Line attributes
};

   enum ELineStyle { kSolid = 1, kDashed, kDotted, kDashDotted };

#endif

