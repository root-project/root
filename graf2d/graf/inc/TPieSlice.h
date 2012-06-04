/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TPieSlice
#define ROOT_TPieSlice
#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TString
#include <TString.h>
#endif
#ifndef ROOT_TAttText
#include <TAttText.h>
#endif
#ifndef ROOT_TAttFill
#include <TAttFill.h>
#endif
#ifndef ROOT_TAttLine
#include <TAttLine.h>
#endif
#ifndef ROOT_TPie
#include <TPie.h>
#endif

class TPieSlice : public TNamed, public TAttFill, public TAttLine {

private:
   Bool_t   fIsActive;        //! True if is the slice under the mouse pointer

protected:
   TPie *fPie;             // The TPie object that contain this slice
   Double_t fValue;        //value value of this slice
   Double_t fRadiusOffset; //roffset offset from the center of the pie

public:
   TPieSlice();
   TPieSlice(const char *, const char *, TPie*, Double_t val=0);
   virtual ~TPieSlice() {;}

   virtual Int_t  DistancetoPrimitive(Int_t,Int_t);
   Double_t       GetRadiusOffset();
   Double_t       GetValue();
   void           SavePrimitive(std::ostream &out, Option_t *opts="");
   void           SetIsActive(Bool_t is) { fIsActive = is; }
   void           SetRadiusOffset(Double_t);  // *MENU*
   void           SetValue(Double_t);         // *MENU*

   friend class TPie;

   ClassDef(TPieSlice,1)            // Slice of a pie chart graphics class
};

#endif
