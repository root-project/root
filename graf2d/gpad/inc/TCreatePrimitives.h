// @(#)root/gpad:$Id: TCreatePrimitives.h,v 1.0

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TCreatePrimitives
#define ROOT_TCreatePrimitives


#include "TLatex.h"
#include "TCurlyArc.h"
#include "TArrow.h"
#include "TArc.h"
#include "TPavesText.h"
#include "TPaveLabel.h"
#include "TDiamond.h"
#include "TGraph.h"

#include "TPad.h"


class TCreatePrimitives {

private:

   static TPave *fgPave;
   static TPaveText *fgPaveText;
   static TPavesText *fgPavesText;
   static TDiamond *fgDiamond;
   static TPaveLabel *fgPaveLabel;
   static TBox *fgPadBBox;

   static void ExecuteLine(TPad &pad, Int_t mode, Int_t event, Int_t px, Int_t py);
   static void ExecuteText(TPad &pad, Int_t mode, Int_t event, Int_t px, Int_t py);
   static void ExecutePolyLine(TPad &parent, Int_t mode, Int_t event, Int_t px, Int_t py);

public:

   TCreatePrimitives();
   virtual ~TCreatePrimitives();
   static void R__DEPRECATED(7,00, "Will be removed from public API") Ellipse(Int_t event, Int_t px, Int_t py,Int_t mode);
   static void R__DEPRECATED(7,00, "Will be removed from public API") Line(Int_t event, Int_t px, Int_t py, Int_t mode);
   static void Pad(Int_t event, Int_t px, Int_t py, Int_t);
   static void Pave(Int_t event, Int_t px, Int_t py, Int_t mode);
   static void R__DEPRECATED(7,00, "Will be removed from public API") PolyLine(Int_t event, Int_t px, Int_t py, Int_t mode);
   static void R__DEPRECATED(7,00, "Will be removed from public API") Text(Int_t event, Int_t px, Int_t py, Int_t mode);

   static void ExecuteEvent(TPad &pad, Int_t edit_mode, Int_t event, Int_t px, Int_t py);
};

#endif
