/// \file TPad.cxx
/// \ingroup Gpad ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2017-07-07
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/TPad.hxx"

#include "ROOT/TLogger.hxx"
#include "ROOT/TPadExtent.hxx"
#include "ROOT/TPadPos.hxx"
#include <ROOT/TPadDisplayItem.hxx>
#include <ROOT/TPadPainter.hxx>
#include <ROOT/TCanvas.hxx>

#include <cassert>
#include <limits>

ROOT::Experimental::TPadBase::~TPadBase() = default;

void ROOT::Experimental::TPadBase::AssignUniqueID(std::shared_ptr<TDrawable> &ptr)
{
   if (!ptr)
      return;

   TCanvas *canv = GetCanvas();
   if (!canv) {
      R__ERROR_HERE("Gpad") << "Cannot access canvas when unique object id should be assigned";
      return;
   }

   ptr->fId = canv->GenerateUniqueId();
}

std::shared_ptr<ROOT::Experimental::TDrawable> ROOT::Experimental::TPadBase::FindDrawable(const std::string &id) const
{
   for (auto &&drawable : GetPrimitives()) {

      if (drawable->GetId() == id)
         return drawable;

      TPadDrawable *pad_draw = dynamic_cast<TPadDrawable *> (drawable.get());
      if (!pad_draw || !pad_draw->Get()) continue;

      auto subelem = pad_draw->Get()->FindDrawable(id);

      if (!subelem) continue;

      return subelem;
   }

   return nullptr;
}

std::vector<std::vector<ROOT::Experimental::TPad *>>
ROOT::Experimental::TPadBase::Divide(int nHoriz, int nVert, const TPadExtent &padding /*= {}*/)
{
   std::vector<std::vector<TPad *>> ret;
   if (!nHoriz)
      R__ERROR_HERE("Gpad") << "Cannot divide into 0 horizontal sub-pads!";
   if (!nVert)
      R__ERROR_HERE("Gpad") << "Cannot divide into 0 vertical sub-pads!";
   if (!nHoriz || !nVert)
      return ret;

   // Start with the whole (sub-)pad:
   TPadExtent offset{1._normal, 1._normal};
   /// We need n Pads plus n-1 padding. Thus each `(subPadSize + padding)` is `(parentPadSize + padding) / n`.
   offset = (offset + padding);
   offset *= {1. / nHoriz, 1. / nVert};
   const TPadExtent size = offset - padding;

   printf("SIZES %5.2f %5.2f\n", size.fHoriz.fNormal.fVal, size.fVert.fNormal.fVal);

   ret.resize(nHoriz);
   for (int iHoriz = 0; iHoriz < nHoriz; ++iHoriz) {
      ret[iHoriz].resize(nVert);
      for (int iVert = 0; iVert < nVert; ++iVert) {
         TPadPos subPos = offset;
         subPos *= {1. * iHoriz, 1. * iVert};
         auto uniqPad = std::make_unique<TPad>(*this, size);
         ret[iHoriz][iVert] = uniqPad.get();
         Draw(std::move(uniqPad), subPos);

         printf("Create subpad pos %5.2f %5.2f\n", subPos.fHoriz.fNormal.fVal, subPos.fVert.fNormal.fVal);
      }
   }
   return ret;
}

ROOT::Experimental::TFrame *ROOT::Experimental::TPadBase::GetOrCreateFrame()
{
   CreateFrameIfNeeded();
   return fFrame.get();
}

void ROOT::Experimental::TPadBase::CreateFrameIfNeeded()
{
   if (!fFrame) {
      fFrame = std::make_unique<ROOT::Experimental::TFrame>();
   }
}


/////////////////////////////////////////////////////////////////////////////////////////////////
/// Get a pad axis from the TFrame.
/// \param dimension - Index of the dimension of the TFrame user coordinate system.

ROOT::Experimental::TPadUserAxisBase* ROOT::Experimental::TPadBase::GetAxis(size_t dimension) const
{
   if (fFrame && dimension < fFrame->GetNDimensions())
      return &fFrame->GetUserAxis(dimension);
   return nullptr;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Get a pad axis from the TFrame.
/// \param dimension - Index of the dimension of the TFrame user coordinate system.

ROOT::Experimental::TPadUserAxisBase* ROOT::Experimental::TPadBase::GetOrCreateAxis(size_t dimension)
{
   GetOrCreateFrame()->GrowToDimensions(dimension);
   return &fFrame->GetUserAxis(dimension);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Set the range of an axis as begin, end.

void ROOT::Experimental::TPadBase::SetAxisBounds(int dimension, double begin, double end)
{
   GetOrCreateFrame()->GrowToDimensions(dimension);
   GetAxis(dimension)->SetBounds(begin, end);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Set the range of an axis as bound kind and bound (up or down).

void ROOT::Experimental::TPadBase::SetAxisBound(int dimension, TPadUserAxisBase::EAxisBoundsKind boundsKind, double bound)
{
   GetOrCreateFrame()->GrowToDimensions(dimension);
   GetAxis(dimension)->SetBound(boundsKind, bound);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Set the range of an axis as bound kind and bound (up or down).

void ROOT::Experimental::TPadBase::SetAxisAutoBounds(int dimension)
{
   GetOrCreateFrame()->GrowToDimensions(dimension);
   GetAxis(dimension)->SetAutoBounds();
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Set the range of an axis as bound kind and bound (up or down).

void ROOT::Experimental::TPadBase::SetAllAxisBounds(const std::vector<std::array<double, 2>> &vecBeginAndEnd)
{
   GetOrCreateFrame()->GrowToDimensions(vecBeginAndEnd.size());
   if (vecBeginAndEnd.size() != fFrame->GetNDimensions()) {
      R__ERROR_HERE("Gpadv7")
         << "Array of axis bound has wrong size " <<  vecBeginAndEnd.size()
         << " versus numer of axes in frame " << fFrame->GetNDimensions();
      return;
   }

   for (size_t i = 0, n = fFrame->GetNDimensions(); i < n; ++i)
      fFrame->GetUserAxis(i).SetBounds(vecBeginAndEnd[i][0], vecBeginAndEnd[i][1]);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Set the range of an axis as bound kind and bound (up or down).

void ROOT::Experimental::TPadBase::SetAllAxisBound(const std::vector<BoundKindAndValue> &vecBoundAndKind)
{
   GetOrCreateFrame()->GrowToDimensions(vecBoundAndKind.size());
   if (vecBoundAndKind.size() != fFrame->GetNDimensions()) {
      R__ERROR_HERE("Gpadv7")
         << "Array of axis bound has wrong size " <<  vecBoundAndKind.size()
         << " versus numer of axes in frame " << fFrame->GetNDimensions();
      return;
   }

   for (size_t i = 0, n = fFrame->GetNDimensions(); i < n; ++i)
      fFrame->GetUserAxis(i).SetBound(vecBoundAndKind[i].fKind, vecBoundAndKind[i].fBound);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Set the range of an axis as bound kind and bound (up or down).

void ROOT::Experimental::TPadBase::SetAllAxisAutoBounds()
{
   for (size_t i = 0, n = GetOrCreateFrame()->GetNDimensions(); i < n; ++i)
      fFrame->GetUserAxis(i).SetAutoBounds();
}


/////////////////////////////////////////////////////////////////////////////////////////////////


ROOT::Experimental::TPad::~TPad() = default;

/////////////////////////////////////////////////////////////////////////////////////////////////

ROOT::Experimental::TPadDrawable::TPadDrawable(std::shared_ptr<TPad> pPad, const TPadDrawingOpts &opts /*= {}*/)
   : fPad(std::move(pPad)), fOpts(opts)
{
}

/// Paint the pad.
void ROOT::Experimental::TPadDrawable::Paint(Internal::TPadPainter &toppad)
{
   Internal::TPadPainter painter;

   painter.PaintDrawables(*fPad.get());

   painter.fPadDisplayItem->SetDrawOpts(&GetOptions());

   painter.fPadDisplayItem->SetSize(&fPad->GetSize());

   toppad.AddDisplayItem(std::move(painter.fPadDisplayItem));
}
