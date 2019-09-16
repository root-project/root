/// \file RPad.cxx
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

#include "ROOT/RPad.hxx"

#include "ROOT/RLogger.hxx"
#include "ROOT/RPadExtent.hxx"
#include "ROOT/RPadPos.hxx"
#include <ROOT/RPadDisplayItem.hxx>
#include <ROOT/RPadPainter.hxx>
#include <ROOT/RCanvas.hxx>

#include <cassert>
#include <limits>

ROOT::Experimental::RPadBase::~RPadBase() = default;

void ROOT::Experimental::RPadBase::AssignUniqueID(std::shared_ptr<RDrawable> &ptr)
{
   if (!ptr)
      return;

   RCanvas *canv = GetCanvas();
   if (!canv) {
      R__ERROR_HERE("Gpad") << "Cannot access canvas when unique object id should be assigned";
      return;
   }

   ptr->fId = canv->GenerateUniqueId();
}

std::shared_ptr<ROOT::Experimental::RDrawable> ROOT::Experimental::RPadBase::FindDrawable(const std::string &id) const
{
   for (auto &drawable : GetPrimitives()) {

      if (drawable->GetId() == id)
         return drawable;

      RPadBase *pad_draw = dynamic_cast<RPadBase *> (drawable.get());

      if (pad_draw) {
         auto subelem = pad_draw->FindDrawable(id);

         if (subelem)
            return subelem;
      }
   }

   return nullptr;
}

std::vector<std::vector<std::shared_ptr<ROOT::Experimental::RPad>>>
ROOT::Experimental::RPadBase::Divide(int nHoriz, int nVert, const RPadExtent &padding)
{
   std::vector<std::vector<std::shared_ptr<RPad>>> ret;
   if (!nHoriz)
      R__ERROR_HERE("Gpad") << "Cannot divide into 0 horizontal sub-pads!";
   if (!nVert)
      R__ERROR_HERE("Gpad") << "Cannot divide into 0 vertical sub-pads!";
   if (!nHoriz || !nVert)
      return ret;

   // Start with the whole (sub-)pad:
   RPadExtent offset{1._normal, 1._normal};
   /// We need n Pads plus n-1 padding. Thus each `(subPadSize + padding)` is `(parentPadSize + padding) / n`.
   offset = (offset + padding);
   offset *= {1. / nHoriz, 1. / nVert};
   const RPadExtent size = offset - padding;

   for (int iHoriz = 0; iHoriz < nHoriz; ++iHoriz) {
      ret.emplace_back();
      for (int iVert = 0; iVert < nVert; ++iVert) {
         RPadPos subPos = offset;
         subPos *= {1. * iHoriz, 1. * iVert};

         auto subpad = Draw<RPad>(this, subPos, size);

         ret.back().emplace_back(subpad);
         // printf("Create subpad pos %5.2f %5.2f\n", subPos.fHoriz.fNormal.fVal, subPos.fVert.fNormal.fVal);
      }
   }
   return ret;
}

ROOT::Experimental::RFrame *ROOT::Experimental::RPadBase::GetOrCreateFrame()
{
   CreateFrameIfNeeded();
   return fFrame.get();
}

void ROOT::Experimental::RPadBase::CreateFrameIfNeeded()
{
   if (!fFrame) {
      fFrame = std::make_unique<ROOT::Experimental::RFrame>();
   }
}


/////////////////////////////////////////////////////////////////////////////////////////////////
/// Get a pad axis from the RFrame.
/// \param dimension - Index of the dimension of the RFrame user coordinate system.

ROOT::Experimental::RPadUserAxisBase* ROOT::Experimental::RPadBase::GetAxis(size_t dimension) const
{
   if (fFrame && dimension < fFrame->GetNDimensions())
      return &fFrame->GetUserAxis(dimension);
   return nullptr;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Get a pad axis from the RFrame.
/// \param dimension - Index of the dimension of the RFrame user coordinate system.

ROOT::Experimental::RPadUserAxisBase* ROOT::Experimental::RPadBase::GetOrCreateAxis(size_t dimension)
{
   GetOrCreateFrame()->GrowToDimensions(dimension);
   return &fFrame->GetUserAxis(dimension);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Set the range of an axis as begin, end.

void ROOT::Experimental::RPadBase::SetAxisBounds(int dimension, double begin, double end)
{
   GetOrCreateFrame()->GrowToDimensions(dimension);
   GetAxis(dimension)->SetBounds(begin, end);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Set the range of an axis as bound kind and bound (up or down).

void ROOT::Experimental::RPadBase::SetAxisBound(int dimension, RPadUserAxisBase::EAxisBoundsKind boundsKind, double bound)
{
   GetOrCreateFrame()->GrowToDimensions(dimension);
   GetAxis(dimension)->SetBound(boundsKind, bound);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Set the range of an axis as bound kind and bound (up or down).

void ROOT::Experimental::RPadBase::SetAxisAutoBounds(int dimension)
{
   GetOrCreateFrame()->GrowToDimensions(dimension);
   GetAxis(dimension)->SetAutoBounds();
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Set the range of an axis as bound kind and bound (up or down).

void ROOT::Experimental::RPadBase::SetAllAxisBounds(const std::vector<std::array<double, 2>> &vecBeginAndEnd)
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

void ROOT::Experimental::RPadBase::SetAllAxisBound(const std::vector<BoundKindAndValue> &vecBoundAndKind)
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
/// Collect all shared items to resolve shared_ptr after IO

void ROOT::Experimental::RPadBase::CollectShared(Internal::RIOSharedVector_t &vect)
{
   for (auto &dr : fPrimitives)
      dr->CollectShared(vect);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Set the range of an axis as bound kind and bound (up or down).

void ROOT::Experimental::RPadBase::SetAllAxisAutoBounds()
{
   for (size_t i = 0, n = GetOrCreateFrame()->GetNDimensions(); i < n; ++i)
      fFrame->GetUserAxis(i).SetAutoBounds();
}


/////////////////////////////////////////////////////////////////////////////////////////////////


ROOT::Experimental::RPad::~RPad() = default;

/////////////////////////////////////////////////////////////////////////////////////////////////


/// Paint the pad.
void ROOT::Experimental::RPad::Paint(Internal::RPadPainter &toppad)
{
   Internal::RPadPainter painter;

   painter.PaintDrawables(*this);

   painter.fPadDisplayItem->SetPadPosSize(&fPos, &fSize);

   painter.fPadDisplayItem->SetAttributes(&fAttr);

   toppad.AddDisplayItem(std::move(painter.fPadDisplayItem));
}


