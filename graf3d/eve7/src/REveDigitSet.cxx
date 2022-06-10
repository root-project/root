// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/REveDigitSet.hxx"
#include "ROOT/REveManager.hxx"
#include "ROOT/REveTrans.hxx"
#include "ROOT/REveSelection.hxx"

#include "TRefArray.h"

#include <nlohmann/json.hpp>

using namespace::ROOT::Experimental;

/** \class REveDigitSet
\ingroup REve
Base-class for storage of digit collections; provides
transformation matrix (REveTrans), signal to color mapping
(REveRGBAPalette) and visual grouping (REveFrameBox).

Base-class for displaying a digit collection.
Provides common services for:
 - specifying signal / color per digit;
 - specifying object reference per digit;
 - controlling palette and thresholds (external object REveRGBAPalette);
 - showing a frame around the digits (external object REveFrameBox);
 - specifying transformation matrix for the whole collection;
   by data-member of class REveTrans.

Use method DigitId(TObject* id) to assign additional identification
to the last created digit. By calling SetOwnIds(kTRUE) tje
digit-set becomes the owner of the assigned objects and deletes
them on destruction.
Note that TRef is used for referencing the objects and if you
instantiate the objects just to pass them to digit-set you should
also call  TProcessID::Get/SetObjectCount() at the beginning / end
of processing of an event. See documentation for class TRef, in
particular section 'ObjectNumber'.

If you use value-is-color mode and want to use transparency, set
the transparency to non-zero value so that GL-renderer will be
properly informed.

If you want to use single color for all elements call:
~~~ {.cpp}
   UseSingleColor()
~~~
Palette controls will not work in this case.

A pointer to a rectangle / box of class REveFrameBox can be set via
~~~ {.cpp}
   void SetFrame(REveFrameBox* b);
~~~
A single REveFrameBox can be shared among several digit-sets (it is
reference-counted). The following flags affect how the frame-box will drawn
and used for selection and highlight:
~~~ {.cpp}
   Bool_t fSelectViaFrame;
   Bool_t fHighlightFrame;
~~~
REveDigitSet is sub-classed from REveSecondarySelectable -- this means
individual digits can be selected. By calling:
~~~ {.cpp}
   REveSecondarySelectable::SetAlwaysSecSelect(kTRUE);
~~~
one can enforce immediate feedback (highlight, tooltip and select on normal
left-mouse click) on given digit-set.

See also:
~~~ {.cpp}
   REveQuadSet: rectangle, hexagon or line per digit
   REveBoxSet   a 3D box per digit
~~~
*/

////////////////////////////////////////////////////////////////////////////////

REveDigitSet::REveDigitSet(const char* n, const char* t) :
   REveElement     (n, t),

   fDefaultValue   (kMinInt),
   fValueIsColor   (kFALSE),
   fSingleColor    (kFALSE),
   fAntiFlick      (kTRUE),
   fDetIdsAsSecondaryIndices         (kFALSE),
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

////////////////////////////////////////////////////////////////////////////////
/// Destructor.
/// Unreference frame and palette. Destroy referenced objects if they
/// are owned by the REveDigitSet.

REveDigitSet::~REveDigitSet()
{
   SetFrame(0);
   SetPalette(0);
   if (fDetIdsAsSecondaryIndices)
      ReleaseIds();
}

////////////////////////////////////////////////////////////////////////////////
/// Protected method called whenever a new digit is added.

REveDigitSet::DigitBase_t* REveDigitSet::NewDigit()
{
   fLastIdx   = fPlex.Size();
   fLastDigit = new (fPlex.NewAtom()) DigitBase_t(fDefaultValue);
   return fLastDigit;
}

////////////////////////////////////////////////////////////////////////////////
/// Protected method. Release and delete the referenced objects, the
/// ownership is *NOT* checked.

void REveDigitSet::ReleaseIds()
{
   if (fDigitIds)
   {
      const Int_t N = fDigitIds->GetSize();

      for (Int_t i = 0; i < N; ++i)
         delete fDigitIds->At(i);

      fDigitIds->Expand(0);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Instruct digit-set to use single color for its digits.
/// Call SetMainColor/Transparency to initialize it.

void REveDigitSet::UseSingleColor()
{
   fSingleColor = kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Override from REveElement, forward to Frame.

void REveDigitSet::SetMainColor(Color_t color)
{
   if (fSingleColor)
   {
      REveElement::SetMainColor(color);
   }
   else if (fFrame)
   {
      fFrame->SetFrameColor(color);
      fFrame->StampBackPtrElements(kCBColorSelection);
   }
}

/*
////////////////////////////////////////////////////////////////////////////////
/// Virtual function called when both fSelected is false and
/// fImpliedSelected is 0.

void REveDigitSet::UnSelected()
{
   fSelectedSet.clear();
   REveElement::UnSelected();
}

////////////////////////////////////////////////////////////////////////////////
/// Virtual function called when both fHighlighted is false and
/// fImpliedHighlighted is 0.

void REveDigitSet::UnHighlighted()
{
   fHighlightedSet.clear();
   REveElement::UnHighlighted();
}

*/
////////////////////////////////////////////////////////////////////////////////
/// Return tooltip for highlighted element if always-sec-select is set.
/// Otherwise return the tooltip for this element.

std::string REveDigitSet::GetHighlightTooltip(const std::set<int>& secondary_idcs) const
{
   if (GetAlwaysSecSelect())
   {
      if (fTooltipCBFoo)
      {
        return (fTooltipCBFoo)(this, *secondary_idcs.begin());
      }
      else if (fDigitIds)
      {
         TObject *o = GetId(*secondary_idcs.begin());
         if (o)
            return o->GetName();
      }
      return TString::Format("%s; idx=%d", GetCName(), *secondary_idcs.begin()).Data();
   }
   else
   {
      return REveElement::GetHighlightTooltip(secondary_idcs);
   }
}
////////////////////////////////////////////////////////////////////////////////
/// Instruct underlying memory allocator to regroup itself into a
/// contiguous memory chunk.

void REveDigitSet::RefitPlex()
{
   fPlex.Refit();
}

////////////////////////////////////////////////////////////////////////////////
/// Iterate over the digits and determine min and max signal values.

void REveDigitSet::ScanMinMaxValues(Int_t& min, Int_t& max)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Set current digit -- the one that will receive calls to
/// DigitValue/Color/Id/UserData() functions.
/// Note that various AddXyzz() functions set the current digit to the newly
/// added one.

void REveDigitSet::SetCurrentDigit(Int_t idx)
{
   fLastIdx   = idx;
   fLastDigit = GetDigit(idx);
}

////////////////////////////////////////////////////////////////////////////////
/// Set signal value for the last digit added.

void REveDigitSet::DigitValue(Int_t value)
{
   fLastDigit->fValue = value;
}

////////////////////////////////////////////////////////////////////////////////
/// Set color for the last digit added.

void REveDigitSet::DigitColor(Color_t ci)
{
   REveUtil::ColorFromIdx(ci, (UChar_t*) & fLastDigit->fValue, kTRUE);
}

////////////////////////////////////////////////////////////////////////////////
/// Set color for the last digit added.

void REveDigitSet::DigitColor(Color_t ci, Char_t transparency)
{
   REveUtil::ColorFromIdx(ci, (UChar_t*) & fLastDigit->fValue, transparency);
}

////////////////////////////////////////////////////////////////////////////////
/// Set color for the last digit added.

void REveDigitSet::DigitColor(UChar_t r, UChar_t g, UChar_t b, UChar_t a)
{
   UChar_t* x = (UChar_t*) & fLastDigit->fValue;
   x[0] = r; x[1] = g; x[2] = b; x[3] = a;
}

////////////////////////////////////////////////////////////////////////////////
/// Set color for the last digit added.

void REveDigitSet::DigitColor(UChar_t* rgba)
{
   UChar_t* x = (UChar_t*) & fLastDigit->fValue;
   x[0] = rgba[0]; x[1] = rgba[1]; x[2] = rgba[2]; x[3] = rgba[3];
}

////////////////////////////////////////////////////////////////////////////////
/// Set external object reference for the last digit added.

void REveDigitSet::DigitId(TObject* id)
{
   DigitId(fLastIdx, id);
}

////////////////////////////////////////////////////////////////////////////////
/// Set external id for the last added digit.

void REveDigitSet::DigitId(Int_t n, TObject* id)
{
   if (!fDigitIds)
      fDigitIds = new TRefArray;

   if (fOwnIds && n < fDigitIds->GetSize() && fDigitIds->At(n))
      delete fDigitIds->At(n);

   fDigitIds->AddAtAndExpand(id, n);
}

////////////////////////////////////////////////////////////////////////////////
/// Set external object reference for digit n.

TObject* REveDigitSet::GetId(Int_t n) const
{
   return fDigitIds ? fDigitIds->At(n) : 0;
}

/*
////////////////////////////////////////////////////////////////////////////////
/// Paint this object. Only direct rendering is supported.

void REveDigitSet::Paint(Option_t*)
{
   PaintStandard(this);
}

////////////////////////////////////////////////////////////////////////////////
/// Called from renderer when a digit with index idx is selected.
/// This is by-passed when always-secondary-select is active.

void REveDigitSet::DigitSelected(Int_t idx)
{
   DigitBase_t *qb  = GetDigit(idx);
   TObject     *obj = GetId(idx);

   if (fCallbackFoo) {
      (fCallbackFoo)(this, idx, obj);
   }
   if (fEmitSignals) {
      SecSelected(this, idx);
   } else {
      printf("REveDigitSet::DigitSelected idx=%d, value=%d, obj=0x%lx\n",
             idx, qb->fValue, (ULong_t)obj);
      if (obj)
         obj->Print();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Emit a SecSelected signal.
/// This is by-passed when always-secondary-select is active.

void REveDigitSet::SecSelected(REveDigitSet* qs, Int_t idx)
{
   Longptr_t args[2];
   args[0] = (Longptr_t) qs;
   args[1] = (Longptr_t) idx;

   // Emit("SecSelected(REveDigitSet*, Int_t)", args);
}

*/
////////////////////////////////////////////////////////////////////////////////
/// Set REveFrameBox pointer.

void REveDigitSet::SetFrame(REveFrameBox* b)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Set REveRGBAPalette pointer.

void REveDigitSet::SetPalette(REveRGBAPalette* p)
{
   if (fPalette == p) return;
   if (fPalette) fPalette->DecRefCount();
   fPalette = p;
   if (fPalette) fPalette->IncRefCount();
}

////////////////////////////////////////////////////////////////////////////////
/// Make sure the REveRGBAPalette pointer is not null.
/// If it is not set, a new one is instantiated and the range is set
/// to current min/max signal values.

REveRGBAPalette* REveDigitSet::AssertPalette()
{
   if (fPalette == 0) {
      fPalette = new REveRGBAPalette;
      if (!fValueIsColor) {
         Int_t min, max;
         ScanMinMaxValues(min, max);
         fPalette->SetLimits(min, max);
         fPalette->SetMinMax(min, max);
      }
   }
   return fPalette;
}

////////////////////////////////////////////////////////////////////////////////
/// Utility function for maping digit idx with visible shape idx
bool REveDigitSet::IsDigitVisible(const DigitBase_t *d) const
{
   if (fSingleColor) {
      return true;
   } else if (fValueIsColor) {
      return d->fValue ? true : false;
   } else {
      if (fPalette)
         return d->fValue > fPalette->GetMinVal() && d->fValue < fPalette->GetMaxVal();
      else {
         printf("Error REveDigitSet::IsDigitVisible() unhadled case\n");
         return true;
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Utility function for maping digit idx with visible shape idx
int REveDigitSet::GetAtomIdxFromShapeIdx(int iShapeIdx) const
{
   int atomIdx = 0;
   int shapeIdx = 0;
   for (Int_t c = 0; c < fPlex.VecSize(); ++c) {
      Char_t *a = fPlex.Chunk(c);
      Int_t n = fPlex.NAtoms(c);
      while (n--) {
         if (IsDigitVisible((DigitBase_t *)a)) {
            if (shapeIdx == iShapeIdx)
               return atomIdx;
            shapeIdx++;
         }
         atomIdx++;
         a += fPlex.S();
      }
   }

   printf("REveDigitSet::GetAtomIdxFromShapeIdx Error locating atom idx from shape idx %d\n", iShapeIdx);
   return -1;
}

////////////////////////////////////////////////////////////////////////////////
/// Utility function for maping shape idx to digit idx
int REveDigitSet::GetShapeIdxFromAtomIdx(int iAtomIdx) const
{
   int atomIdx = 0;
   int shapeIdx = 0;
   for (Int_t c = 0; c < fPlex.VecSize(); ++c) {
      Char_t *a = fPlex.Chunk(c);
      Int_t n = fPlex.NAtoms(c);
      while (n--) {
         if (IsDigitVisible((DigitBase_t *)a)) {
            if (atomIdx == iAtomIdx) {
               return shapeIdx;
            }
            shapeIdx++;
         }
         atomIdx++;
         a += fPlex.S();
      }
   }

   printf("REveDigitSet::GetShapeIdxFromAtomIdx:: Atom with idx %d dose not have a visible shape \n", iAtomIdx);
   return -1;
}


////////////////////////////////////////////////////////////////////////////////
// Direct callback from EveElemenControl, function elementSelected/elementHighligted controll
//
void REveDigitSet::NewShapePicked(int shapeIdx, Int_t selectionId, bool multi)
{
   int digitId = GetAtomIdxFromShapeIdx(shapeIdx);
   auto digit = GetDigit(digitId);
   if (gDebug) printf("REveDigitSet::NewShapePicked elementId %d shape ID = %d, atom ID = %d, value = %d\n", GetElementId(), shapeIdx, digitId, digit->fValue);

   REveSelection *selection = dynamic_cast<REveSelection *>(ROOT::Experimental::gEve->FindElementById(selectionId));
   std::set<int> sset = {digitId};

   RefSelectedSet() = sset;
   selection->NewElementPicked(GetElementId(), multi, true, sset);
}

////////////////////////////////////////////////////////////////////////////////
// called from REveSelection to stream specific selection data
void REveDigitSet::FillExtraSelectionData(nlohmann::json &j, const std::set<int> &secondary_idcs) const
{
   j["shape_idcs"] = nlohmann::json::array();
   for (auto &i : secondary_idcs) {
      int shapeIdx =  GetShapeIdxFromAtomIdx(i);
      if (shapeIdx >= 0)
         j["shape_idcs"].push_back(GetShapeIdxFromAtomIdx(i));
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Fill core part of JSON representation.

Int_t REveDigitSet::WriteCoreJson(nlohmann::json &j, Int_t rnr_offset)
{
   Int_t ret = REveElement::WriteCoreJson(j, rnr_offset);

   j["fSingleColor"] = fSingleColor;
   j["fAntiFlick"] = GetAntiFlick();
   j["fSecondarySelect"] = fAlwaysSecSelect;
   j["fDetIdsAsSecondaryIndices"] = fDetIdsAsSecondaryIndices;

   return ret;
}
