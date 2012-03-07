// @(#)root/graf:$Id$
// Author: Rene Brun   16/05/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TCutG
#define ROOT_TCutG

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TCutG                                                                //
//                                                                      //
// A Graphical cut.                                                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGraph
#include "TGraph.h"
#endif

class TH2;

class TCutG : public TGraph {

protected:
   TString      fVarX;         //X variable
   TString      fVarY;         //Y variable
   TObject     *fObjectX;      //!pointer to an object corresponding to X
   TObject     *fObjectY;      //!pointer to an object corresponding to Y

public:
   TCutG();
   TCutG(const TCutG &cutg);
   TCutG(const char *name, Int_t n);
   TCutG(const char *name, Int_t n, const Float_t *x, const Float_t *y);
   TCutG(const char *name, Int_t n, const Double_t *x, const Double_t *y);
   virtual ~TCutG();
   
   TCutG &operator=(const TCutG &);
   virtual Double_t Area() const;
   virtual void     Center(Double_t &cx, Double_t &cy) const;
   TObject         *GetObjectX() const {return fObjectX;}
   TObject         *GetObjectY() const {return fObjectY;}
   const char      *GetVarX() const {return fVarX.Data();}
   const char      *GetVarY() const {return fVarY.Data();}
   virtual Double_t IntegralHist(TH2 *h, Option_t *option="") const;
   virtual void     SavePrimitive(ostream &out, Option_t *option = "");
   virtual void     SetObjectX(TObject *obj);
   virtual void     SetObjectY(TObject *obj);
   virtual void     SetVarX(const char *varx); // *MENU*
   virtual void     SetVarY(const char *vary); // *MENU*

   ClassDef(TCutG,2)  // A Graphical cut.
};

#endif
