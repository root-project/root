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

#include "ROOT/RPadBase.hxx"

#include "ROOT/RLogger.hxx"
#include "ROOT/RPadExtent.hxx"
#include "ROOT/RPadPos.hxx"
#include <ROOT/RPad.hxx>
#include <ROOT/RCanvas.hxx>
#include <ROOT/RPadDisplayItem.hxx>

#include <cassert>
#include <limits>

using namespace std::string_literals;

ROOT::Experimental::RPadBase::~RPadBase() = default;

///////////////////////////////////////////////////////////////////////////
/// Use provided style for pad and all primitives inside

void ROOT::Experimental::RPadBase::UseStyle(const std::shared_ptr<RStyle> &style)
{
   RDrawable::UseStyle(style);
   for (auto &drawable : fPrimitives)
      drawable->UseStyle(style);

}

///////////////////////////////////////////////////////////////////////////
/// Find primitive with specified id

std::shared_ptr<ROOT::Experimental::RDrawable> ROOT::Experimental::RPadBase::FindPrimitive(const std::string &id) const
{
   for (auto &drawable : fPrimitives) {

      if (drawable->GetId() == id)
         return drawable.get_shared();

      const RPadBase *pad_draw = dynamic_cast<const RPadBase *> (drawable.get());

      if (pad_draw) {
         auto subelem = pad_draw->FindPrimitive(id);

         if (subelem)
            return subelem;
      }
   }

   return nullptr;
}

///////////////////////////////////////////////////////////////////////////
/// Find primitive with unique id, produce for RDisplayItem
/// Such id used for client-server identification of objects

std::shared_ptr<ROOT::Experimental::RDrawable> ROOT::Experimental::RPadBase::FindPrimitiveByDisplayId(const std::string &id) const
{
   auto p = id.find("_");
   if (p == std::string::npos)
      return nullptr;

   auto prim = GetPrimitive(std::stoul(id.substr(0,p)));
   if (!prim)
      return nullptr;

   auto subid = id.substr(p+1);

   if (RDisplayItem::ObjectIDFromPtr(prim.get()) == subid)
      return prim;

   auto subpad = std::dynamic_pointer_cast<RPadBase>(prim);

   return subpad ? subpad->FindPrimitiveByDisplayId(subid) : nullptr;
}

///////////////////////////////////////////////////////////////////////////
/// Create display items for all primitives in the pad
/// Each display item gets its special id, which used later for client-server communication

void ROOT::Experimental::RPadBase::DisplayPrimitives(RPadBaseDisplayItem &paditem) const
{
   paditem.SetAttributes(&GetAttrMap());
   paditem.SetFrame(GetFrame());

   unsigned indx = 0;

   for (auto &drawable : fPrimitives) {
      auto item = drawable->Display();
      if (item) {
         item->SetObjectIDAsPtr(drawable.get());
         item->SetIndex(indx);
         paditem.Add(std::move(item));
      }
      ++indx;
   }
}

///////////////////////////////////////////////////////////////////////////
/// Divide pad on nHoriz X nVert subpads
/// Return array of array of pads

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
   if (!fFrame)
      fFrame = std::make_unique<ROOT::Experimental::RFrame>();
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
   for (auto &handle : fPrimitives) {
      vect.emplace_back(&handle);
      auto drawable = handle.get();
      if (drawable) drawable->CollectShared(vect);
   }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Set the range of an axis as bound kind and bound (up or down).

void ROOT::Experimental::RPadBase::SetAllAxisAutoBounds()
{
   for (size_t i = 0, n = GetOrCreateFrame()->GetNDimensions(); i < n; ++i)
      fFrame->GetUserAxis(i).SetAutoBounds();
}
