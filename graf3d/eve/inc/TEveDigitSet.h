// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveDigitSet
#define ROOT_TEveDigitSet

#include "TNamed.h"
#include "TQObject.h"
#include "TAtt3D.h"
#include "TAttBBox.h"

#include "TEveUtil.h"
#include "TEveElement.h"
#include "TEveFrameBox.h"
#include "TEveRGBAPalette.h"
#include "TEveChunkManager.h"
#include "TEveSecondarySelectable.h"

class TRefArray;

class TEveDigitSet : public TEveElement,
                     public TNamed, public TQObject,
                     public TAtt3D,
                     public TAttBBox,
                     public TEveSecondarySelectable
{
   friend class TEveDigitSetEditor;
   friend class TEveDigitSetGL;

   TEveDigitSet(const TEveDigitSet&);            // Not implemented
   TEveDigitSet& operator=(const TEveDigitSet&); // Not implemented

public:
   enum ERenderMode_e { kRM_AsIs, kRM_Line, kRM_Fill };

   typedef void (*Callback_foo)(TEveDigitSet*, Int_t, TObject*);
   typedef TString (*TooltipCB_foo)(TEveDigitSet*, Int_t);

   struct DigitBase_t
   {
      // Base-class for digit representation classes.

      Int_t  fValue;    // signal value of a digit (can be direct RGBA color)
      void  *fUserData; // user-data for given digit

      DigitBase_t(Int_t v=0) : fValue(v), fUserData(nullptr) {}
   };

protected:
   TRefArray        *fDigitIds;       //  Array holding references to external objects.

   Int_t             fDefaultValue;   //  Default signal value.
   Bool_t            fValueIsColor;   //  Interpret signal value as RGBA color.
   Bool_t            fSingleColor;    //  Use the same color for all digits.
   Bool_t            fAntiFlick;      // Make extra render pass to avoid flickering when quads are too small.
   Bool_t            fOwnIds;         //  Flag specifying if id-objects are owned by the TEveDigitSet.
   TEveChunkManager  fPlex;           //  Container of digit data.
   DigitBase_t*      fLastDigit;      //! The last / current digit added to collection.
   Int_t             fLastIdx;        //! The last / current idx added to collection.

   Color_t           fColor;          //  Color used for frame (or all digis with single-color).
   TEveFrameBox*     fFrame;          //  Pointer to frame structure.
   TEveRGBAPalette*  fPalette;        //  Pointer to signal-color palette.
   ERenderMode_e     fRenderMode;     //  Render mode: as-is / line / filled.
   Bool_t            fSelectViaFrame; //  Allow selection via frame.
   Bool_t            fHighlightFrame; //  Highlight frame when object is selected.
   Bool_t            fDisableLighting;//  Disable lighting for rendering.
   Bool_t            fHistoButtons;   //  Show histogram buttons in object editor.

   Bool_t            fEmitSignals;    //  Emit signals on secondary-select.
   Callback_foo      fCallbackFoo;    //! Additional function to call on secondary-select.
   TooltipCB_foo     fTooltipCBFoo;   //! Function providing highlight tooltips when always-sec-select is active.

   DigitBase_t* NewDigit();
   void         ReleaseIds();

public:
   TEveDigitSet(const char* n="TEveDigitSet", const char* t="");
   virtual ~TEveDigitSet();

   virtual TObject* GetObject(const TEveException&) const
   { const TObject* obj = this; return const_cast<TObject*>(obj); }

   void   UseSingleColor();

   Bool_t GetAntiFlick() const   { return fAntiFlick; }
   void   SetAntiFlick(Bool_t f) { fAntiFlick = f; }

   virtual void SetMainColor(Color_t color);

   virtual void UnSelected();
   virtual void UnHighlighted();

   virtual TString GetHighlightTooltip();

   // Implemented in sub-classes:
   // virtual void Reset(EQuadType_e quadType, Bool_t valIsCol, Int_t chunkSize);

   void RefitPlex();
   void ScanMinMaxValues(Int_t& min, Int_t& max);

   // --------------------------------

   void SetCurrentDigit(Int_t idx);

   void DigitValue(Int_t value);
   void DigitColor(Color_t ci);
   void DigitColor(Color_t ci, Char_t transparency);
   void DigitColor(UChar_t r, UChar_t g, UChar_t b, UChar_t a=255);
   void DigitColor(UChar_t* rgba);

   Bool_t GetOwnIds() const     { return fOwnIds; }
   void   SetOwnIds(Bool_t o)   { fOwnIds = o; }

   void   DigitId(TObject* id);
   void   DigitUserData(void* ud);

   void   DigitId(Int_t n, TObject* id);
   void   DigitUserData(Int_t n, void* ud);

   DigitBase_t* GetDigit(Int_t n) const { return (DigitBase_t*) fPlex.Atom(n); }
   TObject*     GetId(Int_t n) const;
   void*        GetUserData(Int_t n) const;
   using TEveElement::GetUserData;

   // --------------------------------

   // Implemented in subclasses:
   // virtual void ComputeBBox();

   virtual void Paint(Option_t* option="");

   virtual void DigitSelected(Int_t idx);
   virtual void SecSelected(TEveDigitSet* qs, Int_t idx); // *SIGNAL*

   // --------------------------------

   TEveChunkManager* GetPlex() { return &fPlex; }

   TEveFrameBox* GetFrame() const { return fFrame; }
   void          SetFrame(TEveFrameBox* b);

   Bool_t GetSelectViaFrame() const    { return fSelectViaFrame; }
   void   SetSelectViaFrame(Bool_t sf) { fSelectViaFrame = sf; }

   Bool_t GetHighlightFrame() const    { return fHighlightFrame; }
   void   SetHighlightFrame(Bool_t hf) { fHighlightFrame = hf; }

   Bool_t GetValueIsColor()  const { return fValueIsColor; }

   TEveRGBAPalette* GetPalette() const { return fPalette; }
   void             SetPalette(TEveRGBAPalette* p);
   TEveRGBAPalette* AssertPalette();

   ERenderMode_e  GetRenderMode()           const { return fRenderMode; }
   void           SetRenderMode(ERenderMode_e rm) { fRenderMode = rm; }

   Bool_t GetDisableLighting() const   { return fDisableLighting; }
   void   SetDisableLighting(Bool_t l) { fDisableLighting = l; }

   Bool_t GetHistoButtons() const   { return fHistoButtons; }
   void   SetHistoButtons(Bool_t f) { fHistoButtons = f; }

   Bool_t GetEmitSignals() const   { return fEmitSignals; }
   void   SetEmitSignals(Bool_t f) { fEmitSignals = f; }

   Callback_foo GetCallbackFoo()         const { return fCallbackFoo; }
   void         SetCallbackFoo(Callback_foo f) { fCallbackFoo = f; }

   TooltipCB_foo GetTooltipCBFoo()          const { return fTooltipCBFoo; }
   void          SetTooltipCBFoo(TooltipCB_foo f) { fTooltipCBFoo = f; }

   ClassDef(TEveDigitSet, 0); // Base-class for storage of digit collections; provides transformation matrix (TEveTrans), signal to color mapping (TEveRGBAPalette) and visual grouping (TEveFrameBox).
};

#endif
