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


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TAttAxis                                                             //
//                                                                      //
// Axis attributes.                                                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif
#ifndef ROOT_Riosfwd
#include "Riosfwd.h"
#endif


class TAttAxis {
protected:
   Int_t        fNdivisions;   //Number of divisions(10000*n3 + 100*n2 + n1)
   Color_t      fAxisColor;    //color of the line axis
   Color_t      fLabelColor;   //color of labels
   Style_t      fLabelFont;    //font for labels
   Float_t      fLabelOffset;  //offset of labels
   Float_t      fLabelSize;    //size of labels
   Float_t      fTickLength;   //length of tick marks
   Float_t      fTitleOffset;  //offset of axis title
   Float_t      fTitleSize;    //size of axis title
   Color_t      fTitleColor;   //color of axis title
   Style_t      fTitleFont;    //font for axis title

public:
   TAttAxis();
   virtual ~TAttAxis();
   void     Copy(TAttAxis &attaxis) const;
   Int_t    GetNdivisions()  const {return fNdivisions;}
   Color_t  GetAxisColor()   const {return fAxisColor;}
   Color_t  GetLabelColor()  const {return fLabelColor;}
   Style_t  GetLabelFont()   const {return fLabelFont;}
   Float_t  GetLabelOffset() const {return fLabelOffset;}
   Float_t  GetLabelSize()   const {return fLabelSize;}
   Float_t  GetTitleOffset() const {return fTitleOffset;}
   Float_t  GetTitleSize()   const {return fTitleSize;}
   Float_t  GetTickLength()  const {return fTickLength;}
   Color_t  GetTitleColor()  const {return fTitleColor;}
   Style_t  GetTitleFont()   const {return fTitleFont;}
   void     ResetAttAxis(Option_t *option="");
   void     SaveAttributes(ostream &out, const char *name, const char *subname);
   void     SetNdivisions(Int_t n=510, Bool_t optim=kTRUE);  // *MENU*
   void     SetNdivisions(Int_t n1, Int_t n2, Int_t n3, Bool_t optim=kTRUE);
   void     SetAxisColor(Color_t color=1);        // *MENU*
   void     SetLabelColor(Color_t color=1);       // *MENU*
   void     SetLabelFont(Style_t font=62);        // *MENU*
   void     SetLabelOffset(Float_t offset=0.005); // *MENU*
   void     SetLabelSize(Float_t size=0.04);      // *MENU*
   void     SetTickLength(Float_t length=0.03);   // *MENU*
   void     SetTitleOffset(Float_t offset=1);     // *MENU*
   void     SetTitleSize(Float_t size=0.04);      // *MENU*
   void     SetTitleColor(Color_t color=1);       // *MENU*
   void     SetTitleFont(Style_t font=62);        // *MENU*

   ClassDef(TAttAxis,4);  //Axis attributes
};

#endif

