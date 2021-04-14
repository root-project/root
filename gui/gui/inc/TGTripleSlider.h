// @(#)root/gui:$Id$
// Author: Bertrand Bellenot   20/01/06

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGTripleSlider
#define ROOT_TGTripleSlider


#include "TGFrame.h"
#include "TGWidget.h"
#include "TGDoubleSlider.h"

class TGTripleVSlider : public TGDoubleVSlider {

protected:
   Int_t            fCz;           ///< vertical pointer position in pixel coordinates
   Double_t         fSCz;          ///< vertical pointer position
   Bool_t           fConstrained;  ///< kTRUE if pointer is constrained to slider edges
   Bool_t           fRelative;     ///< kTRUE if pointer position is relative to slider
   const TGPicture *fPointerPic;   ///< picture to draw pointer

   virtual void     DoRedraw();
   virtual void     SetPointerPos(Int_t z, Int_t opt = 0);

public:
   TGTripleVSlider(const TGWindow *p = 0, UInt_t h = 1, UInt_t type = 1, Int_t id = -1,
                   UInt_t options = kVerticalFrame,
                   Pixel_t back = GetDefaultFrameBackground(),
                   Bool_t reversed = kFALSE,
                   Bool_t mark_ends = kFALSE,
                   Bool_t constrained = kTRUE,
                   Bool_t relative = kFALSE);

   virtual ~TGTripleVSlider();

   virtual void      PointerPositionChanged() { Emit("PointerPositionChanged()"); } //*SIGNAL*
   virtual void      DrawPointer();
   virtual Float_t   GetPointerPosition() const {
      return (Float_t) GetPointerPositionD();
   }
   virtual Long64_t  GetPointerPositionL() const {
      return (Long64_t) GetPointerPositionD();
   }
   virtual Double_t  GetPointerPositionD() const {
      if (fReversedScale) return fVmin + fVmax - fSCz;
      else return fSCz;
   }
   virtual Bool_t    HandleButton(Event_t *event);
   virtual Bool_t    HandleConfigureNotify(Event_t* event);
   virtual Bool_t    HandleMotion(Event_t *event);
   virtual void      SetConstrained(Bool_t on = kTRUE);
   virtual void      SetPointerPosition(Double_t pos);
   virtual void      SetPointerPosition(Float_t pos) {
      SetPointerPosition((Double_t) pos);
   }
   virtual void      SetPointerPosition(Long64_t pos) {
      SetPointerPosition((Double_t) pos);
   }
   virtual void      SetRelative(Bool_t rel = kTRUE) { fRelative = rel; }
   virtual void      SavePrimitive(std::ostream &out, Option_t *option = "");

   ClassDef(TGTripleVSlider,0)  // Vertical triple slider widget
};


class TGTripleHSlider : public TGDoubleHSlider {

protected:
   Int_t            fCz;           ///< horizontal pointer position in pixel coordinates
   Double_t         fSCz;          ///< vertical pointer position
   Bool_t           fConstrained;  ///< kTRUE if pointer is constrained to slider edges
   Bool_t           fRelative;     ///< kTRUE if pointer position is relative to slider
   const TGPicture *fPointerPic;   ///< picture to draw pointer

   virtual void     DoRedraw();
   virtual void     SetPointerPos(Int_t z, Int_t opt = 0);

public:
   TGTripleHSlider(const TGWindow *p = 0, UInt_t w = 1, UInt_t type = 1, Int_t id = -1,
                   UInt_t options = kHorizontalFrame,
                   Pixel_t back = GetDefaultFrameBackground(),
                   Bool_t reversed = kFALSE,
                   Bool_t mark_ends = kFALSE,
                   Bool_t constrained = kTRUE,
                   Bool_t relative = kFALSE);

   virtual ~TGTripleHSlider();

   virtual void      PointerPositionChanged() { Emit("PointerPositionChanged()"); } //*SIGNAL*
   virtual void      DrawPointer();
   virtual Float_t   GetPointerPosition() const {
      return (Float_t) GetPointerPositionD();
   }
   virtual Double_t  GetPointerPositionD() const {
      if (fReversedScale) return fVmin + fVmax - fSCz;
      else return fSCz;
   }
   virtual Long64_t  GetPointerPositionL() const {
      return (Long64_t) GetPointerPositionD();
   }
   virtual Bool_t    HandleButton(Event_t *event);
   virtual Bool_t    HandleConfigureNotify(Event_t* event);
   virtual Bool_t    HandleMotion(Event_t *event);
   virtual void      SetConstrained(Bool_t on = kTRUE);
   virtual void      SetPointerPosition(Double_t pos);
   virtual void      SetPointerPosition(Float_t pos) {
      SetPointerPosition((Double_t) pos);
   }
   virtual void      SetPointerPosition(Long64_t pos) {
      SetPointerPosition((Double_t) pos);
   }
   virtual void      SetRelative(Bool_t rel = kTRUE) { fRelative = rel; }
   virtual void      SavePrimitive(std::ostream &out, Option_t *option = "");

   ClassDef(TGTripleHSlider,0)  // Horizontal triple slider widget
};

#endif
