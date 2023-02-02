// @(#)root/graf:$Id$
// Author: Rene Brun   17/10/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TPaveLabel
#define ROOT_TPaveLabel


#include "TPave.h"
#include "TAttText.h"


class TPaveLabel : public TPave, public TAttText {

protected:
   TString      fLabel;         ///< Label written at the center of Pave

public:
   TPaveLabel();
   TPaveLabel(Double_t x1, Double_t y1,Double_t x2 ,Double_t y2, const char *label, Option_t *option="br");
   TPaveLabel(const TPaveLabel &pavelabel);
   TPaveLabel& operator=(const TPaveLabel &pavelabel);
   virtual ~TPaveLabel();

   void                Copy(TObject &pavelabel) const override;
   void                Draw(Option_t *option="") override;
   virtual TPaveLabel *DrawPaveLabel(Double_t x1, Double_t y1,Double_t x2 ,Double_t y2,
                                     const char *label, Option_t *option="");
   const char         *GetLabel() const {return fLabel.Data();}
   const char         *GetTitle() const override {return fLabel.Data();}
   void                Paint(Option_t *option="") override;
   virtual void        PaintPaveLabel(Double_t x1, Double_t y1,Double_t x2 ,Double_t y2,
                                      const char *label, Option_t *option="");
   void                SavePrimitive(std::ostream &out, Option_t *option = "") override;
   virtual void        SetLabel(const char *label) {fLabel = label;} // *MENU*

   ClassDefOverride(TPaveLabel,1)  //PaveLabel. A Pave with a label
};

#endif

