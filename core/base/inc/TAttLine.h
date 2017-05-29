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


#include "Rtypes.h"

class TAttLine {

protected:
   Color_t    fLineColor;           ///< Line color
   Style_t    fLineStyle;           ///< Line style
   Width_t    fLineWidth;           ///< Line width

public:

   TAttLine();
   TAttLine(Color_t lcolor,Style_t lstyle, Width_t lwidth);
   virtual ~TAttLine();

   void             Copy(TAttLine &attline) const;
   Int_t            DistancetoLine(Int_t px, Int_t py, Double_t xp1, Double_t yp1, Double_t xp2, Double_t yp2 );
   virtual Color_t  GetLineColor() const {return fLineColor;} ///< Return the line color
   virtual Style_t  GetLineStyle() const {return fLineStyle;} ///< Return the line style
   virtual Width_t  GetLineWidth() const {return fLineWidth;} ///< Return the line width
   virtual void     Modify();
   virtual void     ResetAttLine(Option_t *option="");
   virtual void     SaveLineAttributes(std::ostream &out, const char *name, Int_t coldef=1, Int_t stydef=1, Int_t widdef=1);
   virtual void     SetLineAttributes(); // *MENU*
   virtual void     SetLineColor(Color_t lcolor) { fLineColor = lcolor;} ///< Set the line color
   virtual void     SetLineColorAlpha(Color_t lcolor, Float_t lalpha);
   virtual void     SetLineStyle(Style_t lstyle) { fLineStyle = lstyle;} ///< Set the line style
   virtual void     SetLineWidth(Width_t lwidth) { fLineWidth = lwidth;} ///< Set the line width

   ClassDef(TAttLine,2);  //Line attributes
};

   enum ELineStyle { kSolid = 1, kDashed, kDotted, kDashDotted };

#endif

