// @(#)root/graf:$Name$:$Id$
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

class TCutG : public TGraph {

protected:
   TString      fVarX;         //X variable
   TString      fVarY;         //Y variable
   TObject     *fObjectX;      //pointer to an object corresponding to X
   TObject     *fObjectY;      //pointer to an object corresponding to Y

public:
   TCutG();
   TCutG(const char *name, Int_t n, Float_t *x=0, Float_t *y=0);
   virtual ~TCutG();
   TObject       *GetObjectX() {return fObjectX;}
   TObject       *GetObjectY() {return fObjectY;}
   const char    *GetVarX() const {return fVarX.Data();}
   const char    *GetVarY() const {return fVarY.Data();}
   virtual Int_t  IsInside(Float_t x, Float_t y);
   virtual void   SavePrimitive(ofstream &out, Option_t *option);
   virtual void   SetObjectX(TObject *obj) {fObjectX = obj;}
   virtual void   SetObjectY(TObject *obj) {fObjectY = obj;}
   virtual void   SetVarX(const char *varx); // *MENU*
   virtual void   SetVarY(const char *vary); // *MENU*

   ClassDef(TCutG,1)  // A Graphical cut.
};

#endif
