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

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGTripleVSlider and TGTripleHSlider                                  //
//                                                                      //
// TripleSlider inherit from DoubleSlider widgets and allow easy        //
// selection of a min, max and pointer value out of a range.            //
// The pointer position can be constrained to edges of slider and / or  //
// can be relative to the slider position.                              //
//                                                                      //
// To change the min value press the mouse near to the left / bottom    //
// edge of the slider.                                                  //
// To change the max value press the mouse near to the right / top      //
// edge of the slider.                                                  //
// To change both values simultaneously press the mouse near to the     //
// center of the slider.                                                //
// To change pointer value press the mouse on the pointer and drag it   //
// to the desired position                                              //
//                                                                      //
// Dragging the slider will generate the event:                         //
// kC_VSLIDER, kSL_POS, slider id, 0  (for vertical slider)             //
// kC_HSLIDER, kSL_POS, slider id, 0  (for horizontal slider)           //
//                                                                      //
// Pressing the mouse will generate the event:                          //
// kC_VSLIDER, kSL_PRESS, slider id, 0  (for vertical slider)           //
// kC_HSLIDER, kSL_PRESS, slider id, 0  (for horizontal slider)         //
//                                                                      //
// Releasing the mouse will generate the event:                         //
// kC_VSLIDER, kSL_RELEASE, slider id, 0  (for vertical slider)         //
// kC_HSLIDER, kSL_RELEASE, slider id, 0  (for horizontal slider)       //
//                                                                      //
// Moving the pointer will generate the event:                          //
// kC_VSLIDER, kSL_POINTER, slider id, 0  (for vertical slider)         //
// kC_HSLIDER, kSL_POINTER, slider id, 0  (for horizontal slider)       //
//                                                                      //
// Use the functions GetMinPosition(), GetMaxPosition() and             //
// GetPosition() to retrieve the position of the slider.                //
// Use the function GetPointerPosition() to retrieve the position of    //
// the pointer                                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif
#ifndef ROOT_TGWidget
#include "TGWidget.h"
#endif
#ifndef ROOT_TGDoubleSlider
#include "TGDoubleSlider.h"
#endif

class TGTripleVSlider : public TGDoubleVSlider {

protected:
   Int_t            fCz;           // vertical pointer position in pixel coordinates
   Float_t          fSCz;          // vertical pointer position
   Bool_t           fConstrained;  // kTRUE if pointer is constrained to slider edges
   Bool_t           fRelative;     // kTRUE if pointer position is relative to slider
   const TGPicture *fPointerPic;   // picture to draw pointer

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
      if (fReversedScale) return fVmin + fVmax - fSCz;
      else return fSCz;
   }
   virtual Bool_t    HandleButton(Event_t *event);
   virtual Bool_t    HandleConfigureNotify(Event_t* event);
   virtual Bool_t    HandleMotion(Event_t *event);
   virtual void      SetConstrained(Bool_t on = kTRUE);
   virtual void      SetPointerPosition(Float_t pos);
   virtual void      SetRelative(Bool_t rel = kTRUE) { fRelative = rel; }
   virtual void      SavePrimitive(ostream &out, Option_t *option = "");

   ClassDef(TGTripleVSlider,0)  // Vertical triple slider widget
};


class TGTripleHSlider : public TGDoubleHSlider {

protected:
   Int_t            fCz;           // horizontal pointer position in pixel coordinates
   Float_t          fSCz;          // vertical pointer position
   Bool_t           fConstrained;  // kTRUE if pointer is constrained to slider edges
   Bool_t           fRelative;     // kTRUE if pointer position is relative to slider
   const TGPicture *fPointerPic;   // picture to draw pointer

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
      if (fReversedScale) return fVmin + fVmax - fSCz;
      else return fSCz;
   }
   virtual Bool_t    HandleButton(Event_t *event);
   virtual Bool_t    HandleConfigureNotify(Event_t* event);
   virtual Bool_t    HandleMotion(Event_t *event);
   virtual void      SetConstrained(Bool_t on = kTRUE);
   virtual void      SetPointerPosition(Float_t pos);
   virtual void      SetRelative(Bool_t rel = kTRUE) { fRelative = rel; }
   virtual void      SavePrimitive(ostream &out, Option_t *option = "");

   ClassDef(TGTripleHSlider,0)  // Horizontal triple slider widget
};

#endif
