// @(#)root/base:$Id$
// Author: Rene Brun   12/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TAttAxis
#define ROOT_TAttAxis

#include "Rtypes.h"


class TAttAxis {
protected:
   Int_t        fNdivisions;   ///< Number of divisions(10000*n3 + 100*n2 + n1)
   Color_t      fAxisColor;    ///< Color of the line axis
   Color_t      fLabelColor;   ///< Color of labels
   Style_t      fLabelFont;    ///< Font for labels
   Float_t      fLabelOffset;  ///< Offset of labels
   Float_t      fLabelSize;    ///< Size of labels
   Float_t      fTickLength;   ///< Length of tick marks
   Float_t      fTitleOffset;  ///< Offset of axis title
   Float_t      fTitleSize;    ///< Size of axis title
   Color_t      fTitleColor;   ///< Color of axis title
   Style_t      fTitleFont;    ///< Font for axis title

public:
   TAttAxis();
   virtual          ~TAttAxis();
   void     Copy(TAttAxis &attaxis) const;
   virtual Int_t    GetNdivisions()  const {return fNdivisions;}
   virtual Color_t  GetAxisColor()   const {return fAxisColor;}
   virtual Color_t  GetLabelColor()  const {return fLabelColor;}
   virtual Style_t  GetLabelFont()   const {return fLabelFont;}
   virtual Float_t  GetLabelOffset() const {return fLabelOffset;}
   virtual Float_t  GetLabelSize()   const {return fLabelSize;}
   virtual Int_t    GetMaxDigits()   const {return (Int_t)(fNdivisions/1000000);}
   virtual Float_t  GetTitleOffset() const {return fTitleOffset;}
   virtual Float_t  GetTitleSize()   const {return fTitleSize;}
   virtual Float_t  GetTickLength()  const {return fTickLength;}
   virtual Color_t  GetTitleColor()  const {return fTitleColor;}
   virtual Style_t  GetTitleFont()   const {return fTitleFont;}
   virtual void     ResetAttAxis(Option_t *option="");
   virtual void     SaveAttributes(std::ostream &out, const char *name, const char *subname);
   virtual void     SetNdivisions(Int_t n=510, Bool_t optim=kTRUE);   // *MENU*
   virtual void     SetNdivisions(Int_t n1, Int_t n2, Int_t n3, Bool_t optim=kTRUE);
   virtual void     SetAxisColor(Color_t color=1, Float_t alpha=1.);  // *MENU*
   virtual void     SetLabelColor(Color_t color=1, Float_t alpha=1.); // *MENU*
   virtual void     SetLabelFont(Style_t font=62);                    // *MENU*
   virtual void     SetLabelOffset(Float_t offset=0.005);             // *MENU*
   virtual void     SetLabelSize(Float_t size=0.04);                  // *MENU*
   virtual void     SetMaxDigits(Int_t maxDigits = 5);                // *MENU*
   virtual void     SetTickLength(Float_t length=0.03);               // *MENU*
   virtual void     SetTickSize(Float_t size=0.03) {SetTickLength(size);}
   virtual void     SetTitleOffset(Float_t offset=1);                 // *MENU*
   virtual void     SetTitleSize(Float_t size=0.04);                  // *MENU*
   virtual void     SetTitleColor(Color_t color=1);                   // *MENU*
   virtual void     SetTitleFont(Style_t font=62);                    // *MENU*

   ClassDef(TAttAxis,4);  //Axis attributes
};

#endif

