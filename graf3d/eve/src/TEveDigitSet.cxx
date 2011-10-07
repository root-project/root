// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveDigitSet.h"
#include "TEveManager.h"
#include "TEveTrans.h"

#include "TColor.h"
#include "TRefArray.h"


//______________________________________________________________________________
//
// Base-class for storage of digit collections; provides
// transformation matrix (TEveTrans), signal to color mapping
// (TEveRGBAPalette) and visual grouping (TEveFrameBox).
//
// Base-class for displaying a digit collection.
// Provdies common services for:
// - specifying signal / color per digit;
// - specifying object reference per digit;
// - controlling palette and thresholds (external object TEveRGBAPalette);
// - showing a frame around the digits (external object TEveFrameBox);
// - specifying transformation matrix for the whole collection;
//   by data-member of class TEveTrans.
//
// Use method DigitId(TObject* id) to assign additional identification
// to the last created digit. By calling SetOwnIds(kTRUE) tje
// digit-set becomes the owner of the assigned objects and deletes
// them on destruction.
// Note that TRef is used for referencing the objects and if you
// instantiate the objects just to pass them to digit-set you should
// also call  TProcessID::Get/SetObjectCount() at the beginning / end
// of processing of an event. See documentation for class TRef, in
// particular section 'ObjectNumber'.
//
// If you use value-is-color mode and want to use transparency, set
// the transparency to non-zero value so that GL-renderer will be
// properly informed.
//
// If you want to use single color for all elements call:
//   UseSingleColor()
// Palette controls will not work in this case.
//
// A pointer to a rectangle / box of class TEveFrameBox can be set via
//   void SetFrame(TEveFrameBox* b);
// A single TEveFrameBox can be shared among several digit-sets (it is
// reference-counted). The following flafs affect how the frame-box will drawn
// and used for selection and highlight:
//   Bool_t fSelectViaFrame;
//   Bool_t fHighlightFrame;
//
// TEveDigitSet is sub-lcassed from TEveSecondarySelectable -- this means
// individual digits can be selected. By calling:
//   TEveSecondarySelectable::SetAlwaysSecSelect(kTRUE);
// one can enforce immediate feedback (highlight, tooltip and select on normal
// left-mouse click) on given digit-set.
//
// See also:
//   TEveQuadSet: rectangle, hexagon or line per digit
//   TEveBoxSet   a 3D box per digit

ClassImp(TEveDigitSet);

//______________________________________________________________________________
TEveDigitSet::TEveDigitSet(const char* n, const char* t) :
   TEveElement     (fColor),
   TNamed          (n, t),

   fDigitIds       (0),
   fDefaultValue   (kMinInt),
   fValueIsColor   (kFALSE),
   fSingleColor    (kFALSE),
   fAntiFlick      (kTRUE),
   fOwnIds         (kFALSE),
   fPlex           (),
   fLastDigit      (0),
   fLastIdx        (-1),

   fColor          (kWhite),
   fFrame          (0),
   fPalette        (0),
   fRenderMode     (kRM_AsIs),
   fSelectViaFrame (kFALSE),
   fHighlightFrame (kFALSE),
   fDisableLighting(kTRUE),
   fHistoButtons   (kTRUE),
   fEmitSignals    (kFALSE),
   fCallbackFoo    (0),
   fTooltipCBFoo   (0)
{
   // Constructor.

   fCanEditMainColor        = kTRUE;
   fCanEditMainTransparency = kTRUE;
   InitMainTrans();
}

//______________________________________________________________________________
TEveDigitSet::~TEveDigitSet()
{
   // Destructor.
   // Unreference frame and palette. Destroy referenced objects if they
   // are owned by the TEveDigitSet.

   SetFrame(0);
   SetPalette(0);
   if (fOwnIds)
      ReleaseIds();
   delete fDigitIds;
}

/******************************************************************************/

//______________________________________________________________________________
TEveDigitSet::DigitBase_t* TEveDigitSet::NewDigit()
{
   // Protected method called whenever a new digit is added.

   fLastIdx   = fPlex.Size();
   fLastDigit = new (fPlex.NewAtom()) DigitBase_t(fDefaultValue);
   return fLastDigit;
}

//______________________________________________________________________________
void TEveDigitSet::ReleaseIds()
{
   // Protected method. Release and delete the referenced objects, the
   // ownership is *NOT* checked.

   if (fDigitIds)
   {
      const Int_t N = fDigitIds->GetSize();

      for (Int_t i = 0; i < N; ++i)
         delete fDigitIds->At(i);

      fDigitIds->Expand(0);
   }
}

//------------------------------------------------------------------------------

//______________________________________________________________________________
void TEveDigitSet::UseSingleColor()
{
   // Instruct digit-set to use single color for its digits.
   // Call SetMainColor/Transparency to initialize it.

   fSingleColor = kTRUE;
}

//______________________________________________________________________________
void TEveDigitSet::SetMainColor(Color_t color)
{
   // Override from TEveElement, forward to Frame.

   if (fSingleColor)
   {
      TEveElement::SetMainColor(color);
   }
   else if (fFrame)
   {
      fFrame->SetFrameColor(color);
      fFrame->StampBackPtrElements(kCBColorSelection);
   }
}

//______________________________________________________________________________
void TEveDigitSet::UnSelected()
{
   // Virtual function called when both fSelected is false and
   // fImpliedSelected is 0.

   fSelectedSet.clear();
   TEveElement::UnSelected();
}

//______________________________________________________________________________
void TEveDigitSet::UnHighlighted()
{
   // Virtual function called when both fHighlighted is false and
   // fImpliedHighlighted is 0.

   fHighlightedSet.clear();
   TEveElement::UnHighlighted();
}

//______________________________________________________________________________
TString TEveDigitSet::GetHighlightTooltip()
{
   // Return tooltip for highlighted element if always-sec-select is set.
   // Otherwise return the tooltip for this element.

   if (fHighlightedSet.empty()) return "";

   if (GetAlwaysSecSelect())
   {
      if (fTooltipCBFoo)
      {
         return (fTooltipCBFoo)(this, *fHighlightedSet.begin());
      }
      else if (fDigitIds)
      {
         TObject *o = GetId(*fHighlightedSet.begin());
         if (o)
            return TString(o->GetName());
      }
      return TString::Format("%s; idx=%d", GetElementName(), *fHighlightedSet.begin());
   }
   else
   {
      return TEveElement::GetHighlightTooltip();
   }
}


/******************************************************************************/
/******************************************************************************/

//______________________________________________________________________________
void TEveDigitSet::RefitPlex()
{
   // Instruct underlying memory allocator to regroup itself into a
   // contiguous memory chunk.

   fPlex.Refit();
}

/******************************************************************************/

//______________________________________________________________________________
void TEveDigitSet::ScanMinMaxValues(Int_t& min, Int_t& max)
{
   // Iterate over the digits and detmine min and max signal values.

   if (fValueIsColor || fPlex.Size() == 0)
   {
      min = max = 0;
      return;
   }

   min = kMaxInt;
   max = kMinInt;
   for (Int_t c=0; c<fPlex.VecSize(); ++c)
   {
      Char_t* a = fPlex.Chunk(c);
      Int_t   n = fPlex.NAtoms(c);
      while (n--)
      {
         Int_t v = ((DigitBase_t*)a)->fValue;
         if (v < min) min = v;
         if (v > max) max = v;
         a += fPlex.S();
      }
   }
   if (min == max)
      --min;
}

/******************************************************************************/

//______________________________________________________________________________
void TEveDigitSet::SetCurrentDigit(Int_t idx)
{
   // Set current digit -- the one that will receive calls to
   // DigitValue/Color/Id/UserData() functions.
   // Note that various AddXyzz() functions set the current digit to the newly
   // added one.

   fLastIdx   = idx;
   fLastDigit = GetDigit(idx);
}

//______________________________________________________________________________
void TEveDigitSet::DigitValue(Int_t value)
{
   // Set signal value for the last digit added.

   fLastDigit->fValue = value;
}

//______________________________________________________________________________
void TEveDigitSet::DigitColor(Color_t ci)
{
   // Set color for the last digit added.

   TEveUtil::ColorFromIdx(ci, (UChar_t*) & fLastDigit->fValue, kTRUE);
}

//______________________________________________________________________________
void TEveDigitSet::DigitColor(Color_t ci, Char_t transparency)
{
   // Set color for the last digit added.

   TEveUtil::ColorFromIdx(ci, (UChar_t*) & fLastDigit->fValue, transparency);
}

//______________________________________________________________________________
void TEveDigitSet::DigitColor(UChar_t r, UChar_t g, UChar_t b, UChar_t a)
{
   // Set color for the last digit added.

   UChar_t* x = (UChar_t*) & fLastDigit->fValue;
   x[0] = r; x[1] = g; x[2] = b; x[3] = a;
}

//______________________________________________________________________________
void TEveDigitSet::DigitColor(UChar_t* rgba)
{
   // Set color for the last digit added.

   UChar_t* x = (UChar_t*) & fLastDigit->fValue;
   x[0] = rgba[0]; x[1] = rgba[1]; x[2] = rgba[2]; x[3] = rgba[3];
}

//______________________________________________________________________________
void TEveDigitSet::DigitId(TObject* id)
{
   // Set external object reference for the last digit added.

   DigitId(fLastIdx, id);
}

//______________________________________________________________________________
void TEveDigitSet::DigitUserData(void* ud)
{
   // Set user-data for the last digit added.

   fLastDigit->fUserData = ud;
}

//______________________________________________________________________________
void TEveDigitSet::DigitId(Int_t n, TObject* id)
{
   // Set external object reference for digit n.

   if (!fDigitIds)
      fDigitIds = new TRefArray;

   if (fOwnIds && n < fDigitIds->GetSize() && fDigitIds->At(n))
      delete fDigitIds->At(n);

   fDigitIds->AddAtAndExpand(id, n);
}

//______________________________________________________________________________
void TEveDigitSet::DigitUserData(Int_t n, void* ud)
{
   // Set user-data for digit n.

   GetDigit(n)->fUserData = ud;
}

//______________________________________________________________________________
TObject* TEveDigitSet::GetId(Int_t n) const
{
   // Return external TObject associated with digit n.

   return fDigitIds ? fDigitIds->At(n) : 0;
}

//______________________________________________________________________________
void* TEveDigitSet::GetUserData(Int_t n) const
{
   // Get user-data associated with digit n.

   return GetDigit(n)->fUserData;
}

/******************************************************************************/
/******************************************************************************/

//______________________________________________________________________________
void TEveDigitSet::Paint(Option_t*)
{
   // Paint this object. Only direct rendering is supported.

   PaintStandard(this);
}

//______________________________________________________________________________
void TEveDigitSet::DigitSelected(Int_t idx)
{
   // Called from renderer when a digit with index idx is selected.
   // This is by-passed when always-secondary-select is active.

   DigitBase_t *qb  = GetDigit(idx);
   TObject     *obj = GetId(idx);

   if (fCallbackFoo) {
      (fCallbackFoo)(this, idx, obj);
   }
   if (fEmitSignals) {
      SecSelected(this, idx);
   } else {
      printf("TEveDigitSet::DigitSelected idx=%d, value=%d, obj=0x%lx\n",
             idx, qb->fValue, (ULong_t)obj);
      if (obj)
         obj->Print();
   }
}

//______________________________________________________________________________
void TEveDigitSet::SecSelected(TEveDigitSet* qs, Int_t idx)
{
   // Emit a SecSelected signal.
   // This is by-passed when always-secondary-select is active.

   Long_t args[2];
   args[0] = (Long_t) qs;
   args[1] = (Long_t) idx;

   Emit("SecSelected(TEveDigitSet*, Int_t)", args);
}

/******************************************************************************/
// Getters / Setters for Frame, TEveRGBAPalette, TEveTrans
/******************************************************************************/

//______________________________________________________________________________
void TEveDigitSet::SetFrame(TEveFrameBox* b)
{
   // Set TEveFrameBox pointer.

   if (fFrame == b) return;
   if (fFrame) fFrame->DecRefCount(this);
   fFrame = b;
   if (fFrame) {
      fFrame->IncRefCount(this);
      if (!fSingleColor) {
         SetMainColorPtr(fFrame->PtrFrameColor());
      }
   } else {
      SetMainColorPtr(&fColor);
   }
}

//______________________________________________________________________________
void TEveDigitSet::SetPalette(TEveRGBAPalette* p)
{
   // Set TEveRGBAPalette pointer.

   if (fPalette == p) return;
   if (fPalette) fPalette->DecRefCount();
   fPalette = p;
   if (fPalette) fPalette->IncRefCount();
}

//______________________________________________________________________________
TEveRGBAPalette* TEveDigitSet::AssertPalette()
{
   // Make sure the TEveRGBAPalette pointer is not null.
   // If it is not set, a new one is instantiated and the range is set
   // to current min/max signal values.

   if (fPalette == 0) {
      fPalette = new TEveRGBAPalette;
      if (!fValueIsColor) {
         Int_t min, max;
         ScanMinMaxValues(min, max);
         fPalette->SetLimits(min, max);
         fPalette->SetMinMax(min, max);
      }
   }
   return fPalette;
}
