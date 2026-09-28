/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TPieSlice
#define ROOT_TPieSlice

#include "TNamed.h"
#include "TAttFill.h"
#include "TAttLine.h"

class TPie;

class TPieSlice : public TNamed, public TAttFill, public TAttLine {

protected:
   TPie *fPie = nullptr;       ///< The TPie object that contain this slice
   Double_t fValue = 0;        ///< value value of this slice
   Double_t fRadiusOffset = 0; ///< offset from the center of the pie

public:
   TPieSlice();
   TPieSlice(const char *, const char *, TPie *, Double_t val = 0);
   ~TPieSlice() override {}

   void           Copy(TObject &slice) const override;
   Int_t          DistancetoPrimitive(Int_t,Int_t) override;
   void           ExecuteEvent(Int_t,Int_t,Int_t) override;
   Double_t       GetRadiusOffset() const;
   Double_t       GetValue() const;
   void           SavePrimitive(std::ostream &out, Option_t *opts="") override;
   void           SetIsActive(Bool_t) R__DEPRECATED(6, 46, "No longer used.") {}
   void           SetRadiusOffset(Double_t);  // *MENU*
   void           SetValue(Double_t);         // *MENU*

   ClassDefOverride(TPieSlice,1)            // Slice of a pie chart graphics class
};

#endif
