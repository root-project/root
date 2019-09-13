/// \file ROOT/RPad.hxx
/// \ingroup Gpad ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2017-07-06
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RPad
#define ROOT7_RPad

#include <memory>
#include <vector>

#include "ROOT/RDrawable.hxx"
#include "ROOT/RDrawingAttr.hxx"
#include "ROOT/RFrame.hxx"
#include "ROOT/RPadExtent.hxx"
#include "ROOT/RPadPos.hxx"
#include "ROOT/RPadUserAxis.hxx"
#include "ROOT/TypeTraits.hxx"

namespace ROOT {
namespace Experimental {

class RPad;
class RCanvas;

/** \class ROOT::Experimental::RPadBase
  Base class for graphic containers for `RDrawable`-s.
  */

class RPadBase : public RDrawable {
public:
   using Primitives_t = std::vector<std::shared_ptr<RDrawable>>;

private:
   /// Content of the pad.
   Primitives_t fPrimitives;

   /// RFrame with user coordinate system, if used by this pad.
   std::unique_ptr<RFrame> fFrame;

   /// Disable copy construction.
   RPadBase(const RPadBase &) = delete;

   /// Disable assignment.
   RPadBase &operator=(const RPadBase &) = delete;

   void AssignUniqueID(std::shared_ptr<RDrawable> &ptr);

protected:
   /// Allow derived classes to default construct a RPadBase.
   RPadBase() = default;

   RDrawableAttributes fAttr{"pad"};       ///< attributes

   void CollectShared(Internal::RIOSharedVector_t &) override;

public:
   virtual ~RPadBase();

   /// Divide this pad into a grid of subpads with padding in between.
   /// \param nHoriz Number of horizontal pads.
   /// \param nVert Number of vertical pads.
   /// \param padding Padding between pads.
   /// \returns vector of vector (ret[x][y]) of created pads.
   std::vector<std::vector<std::shared_ptr<RPad>>> Divide(int nHoriz, int nVert, const RPadExtent &padding = {});

   template<class T, class... ARGS>
   auto Draw(ARGS... args)
   {
      auto res = std::make_shared<T>(args...);

      fPrimitives.emplace_back(res);

      AssignUniqueID(fPrimitives.back());

      return res;
   }

   auto NumPrimitives() const
   {
      return fPrimitives.size();
   }

   /// Remove an object from the list of primitives.
   bool Remove(const std::string &id)
   {
      auto iter = std::find_if(fPrimitives.begin(), fPrimitives.end(),
         [&id](const std::shared_ptr<RDrawable>& drawable) { return drawable->GetId() == id; });
      if (iter == fPrimitives.end())
         return false;
      iter->reset();
      fPrimitives.erase(iter);
      return true;
   }

   bool Remove(const std::shared_ptr<RDrawable> &drawable)
   {
      auto iter = std::find_if(fPrimitives.begin(), fPrimitives.end(),
         [&drawable](const std::shared_ptr<RDrawable>& dr) { return drawable == dr; });
      if (iter == fPrimitives.end())
         return false;
      iter->reset();
      fPrimitives.erase(iter);
      return true;
   }

   bool RemoveAt(unsigned indx)
   {
      if (indx >= fPrimitives.size()) return false;
      fPrimitives[indx].reset();
      fPrimitives.erase(fPrimitives.begin() + indx);
      return true;

   }

   std::shared_ptr<RDrawable> FindDrawable(const std::string &id) const;

   /// Wipe the pad by clearing the list of primitives.
   void Wipe()
   {
      fPrimitives.clear();
   }

   void CreateFrameIfNeeded();

   RFrame *GetOrCreateFrame();
   const RFrame *GetFrame() const { return fFrame.get(); }

   RPadUserAxisBase* GetOrCreateAxis(size_t dimension);
   RPadUserAxisBase* GetAxis(size_t dimension) const;

   void SetAxisBounds(int dimension, double begin, double end);
   void SetAxisBound(int dimension, RPadUserAxisBase::EAxisBoundsKind boundsKind, double bound);
   void SetAxisAutoBounds(int dimension);

   void SetAllAxisBounds(const std::vector<std::array<double, 2>> &vecBeginAndEnd);

   /// Simple struct representing an axis bound.
   struct BoundKindAndValue {
      RPadUserAxisBase::EAxisBoundsKind fKind = RPadUserAxisBase::kAxisBoundsAuto;
      double fBound = 0.;
   };
   void SetAllAxisBound(const std::vector<BoundKindAndValue> &vecBoundAndKind);
   void SetAllAxisAutoBounds();

   /// Get the elements contained in the canvas.
   const Primitives_t &GetPrimitives() const { return fPrimitives; }

   /// Convert a `Pixel` position to Canvas-normalized positions.
//   virtual std::array<RPadLength::Normal, 2> PixelsToNormal(const std::array<RPadLength::Pixel, 2> &pos) const = 0;

   /// Access to the top-most canvas, if any (const version).
   virtual const RCanvas *GetCanvas() const = 0;

   /// Access to the top-most canvas, if any (non-const version).
   virtual RCanvas *GetCanvas() = 0;

   /// Convert user coordinates to normal coordinates.
//   std::array<RPadLength::Normal, 2> UserToNormal(const std::array<RPadLength::User, 2> &pos) const
//   {
//      return fFrame->UserToNormal(pos);
//   }
};


/** \class ROOT::Experimental::RPad
  Graphic container for `RDrawable`-s.
  */

class RPad: public RPadBase {

   /// Pad containing this pad as a sub-pad.
   RPadBase *fParent{nullptr};             /// The parent pad, if this pad has one.

   RPadPos fPos{fAttr, "pos_"};            ///<! pad position
   RPadExtent fSize{fAttr, "size_"};       ///<! pad size

   RAttrLine fLineAttr{fAttr, "border_"};   ///<! border attributes

public:
   /// Create a topmost, non-paintable pad.
   RPad() = default;

   /// Create a child pad.
   RPad(RPadBase *parent, const RPadPos &pos, const RPadExtent &size): fParent(parent) { fPos = pos; fSize = size; }

   /// Destructor to have a vtable.
   virtual ~RPad();

   /// Access to the parent pad (const version).
   const RPadBase *GetParent() const { return fParent; }

   /// Access to the parent pad (non-const version).
   RPadBase *GetParent() { return fParent; }

   /// Access to the top-most canvas (const version).
   const RCanvas *GetCanvas() const override { return fParent ? fParent->GetCanvas() : nullptr; }

   /// Access to the top-most canvas (non-const version).
   RCanvas *GetCanvas() override { return fParent ? fParent->GetCanvas() : nullptr; }

   /// Get the position of the pad in parent (!) coordinates.
   const RPadPos &GetPos() const { return fPos; }

   /// Get the size of the pad in parent (!) coordinates.
   const RPadExtent &GetSize() const { return fSize; }

   /// Set the size of the pad in parent (!) coordinates.
   void SetSize(const RPadExtent &sz) { fSize = sz; }

   /// Set position
   void SetPos(const RPadPos &p) const { (RPadExtent&) fPos = (const RPadExtent&) p; }

   RAttrLine &AttrLine() { return fLineAttr; }
   const RAttrLine &AttrLine() const { return fLineAttr; }

   void Paint(Internal::RPadPainter &) final;

/*

   /// Convert a `Pixel` position to Canvas-normalized positions.
   std::array<RPadLength::Normal, 2> PixelsToNormal(const std::array<RPadLength::Pixel, 2> &pos) const override
   {
      std::array<RPadLength::Normal, 2> posInParentNormal = fParent->PixelsToNormal(pos);
      std::array<RPadLength::Normal, 2> myPixelInNormal =
         fParent->PixelsToNormal({{fSize.fHoriz.fPixel, fSize.fVert.fPixel}});
      std::array<RPadLength::Normal, 2> myUserInNormal =
         fParent->UserToNormal({{fSize.fHoriz.fUser, fSize.fVert.fUser}});
      // If the parent says pos is at 0.6 in normal coords, and our size converted to normal is 0.2, then pos in our
      // coord system is 3.0!
      return {{posInParentNormal[0] / (fSize.fHoriz.fNormal + myPixelInNormal[0] + myUserInNormal[0]),
               posInParentNormal[1] / (fSize.fVert.fNormal + myPixelInNormal[1] + myUserInNormal[1])}};
   }

   /// Convert a RPadPos to [x, y] of normalized coordinates.
   std::array<RPadLength::Normal, 2> ToNormal(const Internal::RPadHorizVert &pos) const
   {
      std::array<RPadLength::Normal, 2> pixelsInNormal = PixelsToNormal({{pos.fHoriz.fPixel, pos.fVert.fPixel}});
      std::array<RPadLength::Normal, 2> userInNormal = UserToNormal({{pos.fHoriz.fUser, pos.fVert.fUser}});
      return {{pos.fHoriz.fNormal + pixelsInNormal[0] + userInNormal[0],
               pos.fVert.fNormal + pixelsInNormal[1] + userInNormal[1]}};
   }

*/

};

} // namespace Experimental
} // namespace ROOT

#endif
