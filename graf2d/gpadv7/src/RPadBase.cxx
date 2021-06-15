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

using namespace ROOT::Experimental;

RPadBase::~RPadBase() = default;

///////////////////////////////////////////////////////////////////////////
/// Use provided style for pad and all primitives inside

void RPadBase::UseStyle(const std::shared_ptr<RStyle> &style)
{
   RDrawable::UseStyle(style);
   for (auto &drawable : fPrimitives)
      drawable->UseStyle(style);
}

///////////////////////////////////////////////////////////////////////////
/// Add primitive

void RPadBase::AddPrimitive(std::shared_ptr<RDrawable> drawable)
{
   if (drawable->GetCssType() == "pad") {
      auto pad = dynamic_cast<RPad *>(drawable.get());
      if (pad) pad->SetParent(this);
   }

   fPrimitives.emplace_back(drawable);
}

///////////////////////////////////////////////////////////////////////////
/// Find primitive with specified id

std::shared_ptr<RDrawable> RPadBase::FindPrimitive(const std::string &id) const
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

std::shared_ptr<RDrawable> RPadBase::FindPrimitiveByDisplayId(const std::string &id) const
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
/// Find subpad which contains primitive with given display id

const RPadBase *RPadBase::FindPadForPrimitiveWithDisplayId(const std::string &id) const
{
   auto p = id.find("_");
   if (p == std::string::npos)
      return nullptr;

   auto prim = GetPrimitive(std::stoul(id.substr(0,p)));
   if (!prim)
      return nullptr;

   auto subid = id.substr(p+1);

   if (RDisplayItem::ObjectIDFromPtr(prim.get()) == subid)
      return this;

   auto subpad = std::dynamic_pointer_cast<RPadBase>(prim);

   return subpad ? subpad->FindPadForPrimitiveWithDisplayId(subid) : nullptr;
}

///////////////////////////////////////////////////////////////////////////
/// Create display items for all primitives in the pad
/// Each display item gets its special id, which used later for client-server communication
/// Second parameter is version id which already delivered to the client

void RPadBase::DisplayPrimitives(RPadBaseDisplayItem &paditem, RDisplayContext &ctxt)
{
   paditem.SetAttributes(&GetAttrMap());
   paditem.SetPadStyle(fStyle.lock());

   unsigned indx = 0;

   for (auto &drawable : fPrimitives) {

      ctxt.SetDrawable(drawable.get(), indx++);

      auto item = drawable->Display(ctxt);

      if (!item)
         item = std::make_unique<RDisplayItem>(true);

      item->SetObjectIDAsPtr(drawable.get());
      item->SetIndex(ctxt.GetIndex());
      // add object with the style
      paditem.Add(std::move(item), drawable->fStyle.lock());
   }
}

///////////////////////////////////////////////////////////////////////////
/// Divide pad on nHoriz X nVert subpads
/// Return array of array of pads

std::vector<std::vector<std::shared_ptr<RPad>>>
RPadBase::Divide(int nHoriz, int nVert, const RPadExtent &padding)
{
   std::vector<std::vector<std::shared_ptr<RPad>>> ret;
   if (!nHoriz)
      R__LOG_ERROR(GPadLog()) << "Cannot divide into 0 horizontal sub-pads!";
   if (!nVert)
      R__LOG_ERROR(GPadLog()) << "Cannot divide into 0 vertical sub-pads!";
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

         auto subpad = Draw<RPad>(subPos, size);

         ret.back().emplace_back(subpad);
      }
   }
   return ret;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Get a frame object for the pad.
/// If frame not exists - creates and add to the end of primitives list

std::shared_ptr<RFrame> RPadBase::GetOrCreateFrame()
{
   auto frame = GetFrame();
   if (!frame) {
      frame.reset(new RFrame);
      fPrimitives.emplace_back(frame);
   }
   return frame;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Get a frame object if exists

const std::shared_ptr<RFrame> RPadBase::GetFrame() const
{
   for (auto &drawable : fPrimitives) {
      if (drawable->GetCssType() == "frame") {
         const std::shared_ptr<RFrame> frame = std::dynamic_pointer_cast<RFrame>(drawable.get_shared());
         if (frame) return frame;
      }
   }
   return nullptr;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Get a frame object if exists

std::shared_ptr<RFrame> RPadBase::GetFrame()
{
   for (auto &drawable : fPrimitives) {
      if (drawable->GetCssType() == "frame") {
         std::shared_ptr<RFrame> frame = std::dynamic_pointer_cast<RFrame>(drawable.get_shared());
         if (frame) return frame;
      }
   }
   return nullptr;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Collect all shared items to resolve shared_ptr after IO

void RPadBase::CollectShared(Internal::RIOSharedVector_t &vect)
{
   for (auto &handle : fPrimitives) {
      vect.emplace_back(&handle);
      auto drawable = handle.get();
      if (drawable) drawable->CollectShared(vect);
   }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Assign drawable version - for pad itself and all primitives

void RPadBase::SetDrawableVersion(Version_t vers)
{
   RDrawable::SetDrawableVersion(vers);

   for (auto &drawable : fPrimitives)
      drawable->SetDrawableVersion(vers);
}
