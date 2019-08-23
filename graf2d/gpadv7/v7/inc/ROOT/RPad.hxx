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
#include "ROOT/RDrawingOptsBase.hxx"
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

class RPadBase {
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

   /// Adds a `DRAWABLE` to `fPrimitives`, returning a `shared_ptr` to `DRAWABLE::GetOptions()`.
   template <class DRAWABLE>
   auto AddDrawable(std::shared_ptr<DRAWABLE> &&uPtr)
   {
      fPrimitives.emplace_back(std::move(uPtr));

      AssignUniqueID(fPrimitives.back());

      using Options_t = typename std::remove_reference<decltype(uPtr->GetOptions())>::type;
      auto spDrawable = std::static_pointer_cast<DRAWABLE>(fPrimitives.back());
      // Return a shared_ptr to the GetOptions() sub-object of the drawable inserted into fPrimitives,
      // where the entry in fPrimitives defines the lifetime.
      return std::shared_ptr<Options_t>(spDrawable, &spDrawable->GetOptions());
   }

protected:
   /// Allow derived classes to default construct a RPadBase.
   RPadBase() = default;

public:
   virtual ~RPadBase();

   /// Divide this pad into a grid of subpads with padding in between.
   /// \param nHoriz Number of horizontal pads.
   /// \param nVert Number of vertical pads.
   /// \param padding Padding between pads.
   /// \returns vector of vector (ret[x][y]) of created pads.
   std::vector<std::vector<RPad *>> Divide(int nHoriz, int nVert, const RPadExtent &padding = {});

   /// Add something to be painted.
   /// The pad observes what's lifetime through a weak pointer.
   /// Drawing options will be constructed through `args`, which can be empty for default-constructed options.
   template <class T, class... ARGS>
   auto Draw(const std::shared_ptr<T> &what, ARGS... args)
   {
      // Requires GetDrawable(what) to be known!
      return AddDrawable(GetDrawable(what, args...));
   }

   /// Add something to be painted. The pad claims ownership.
   /// Drawing options will be constructed through `args`, which can be empty for default-constructed options.
   template <class T, class... ARGS>
   auto Draw(std::unique_ptr<T> &&what, ARGS... args)
   {
      // Requires GetDrawable(what) to be known!
      return AddDrawable(GetDrawable(std::move(what), args...));
   }

   /// Add a copy of something to be painted.
   /// Drawing options will be constructed through `args`, which can be empty for default-constructed options.
   template <class T, class... ARGS, class = typename std::enable_if<!ROOT::TypeTraits::IsSmartOrDumbPtr<T>::value>::type>
   auto Draw(const T &what, ARGS... args)
   {
      // Requires GetDrawable(what) to be known!
      return Draw(std::make_unique<T>(what), args...);
   }

   /// Remove an object from the list of primitives.
   bool Remove(RDrawingOptsBase& opts) {
      auto iter = std::find_if(fPrimitives.begin(), fPrimitives.end(),
         [&opts](const std::shared_ptr<RDrawable>& drawable) { return &drawable->GetOptionsBase() == &opts; });
      if (iter == fPrimitives.end())
         return false;
      iter->reset();
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
   virtual std::array<RPadLength::Normal, 2> PixelsToNormal(const std::array<RPadLength::Pixel, 2> &pos) const = 0;

   /// Access to the top-most canvas, if any (const version).
   virtual const RCanvas *GetCanvas() const = 0;

   /// Access to the top-most canvas, if any (non-const version).
   virtual RCanvas *GetCanvas() = 0;

   /// Convert user coordinates to normal coordinates.
   std::array<RPadLength::Normal, 2> UserToNormal(const std::array<RPadLength::User, 2> &pos) const
   {
      return fFrame->UserToNormal(pos);
   }
};

class RPadDrawable;

/** \class ROOT::Experimental::RPad
  Graphic container for `RDrawable`-s.
  */

class RPad: public RPadBase {
public:
   /** \class DrawingOpts
      Drawing options for a RPad
   */

   class DrawingOpts: public RDrawingOptsBase, public RDrawingAttrBase {
   public:
      DrawingOpts() = default;

      DrawingOpts(const RPadPos &pos, const RPadExtent &size):
         DrawingOpts()
      {
         SetPos(pos);
         SetSize(size);
      }

      RAttrBox Border() { return {FromOption, "border", *this}; }

      /// The position (offset) of the pad.
      DrawingOpts &SetPos(const RPadPos &pos) { Set("pos", pos); return *this; }
      RPadPos GetPos() const { return Get<RPadPos>("pos"); }

      /// The size of the pad.
      DrawingOpts &SetSize(const RPadExtent &size) { Set("size", size); return *this; }
      RPadExtent GetSize() const { return Get<RPadExtent>("size"); }
   };

private:
   /// Pad containing this pad as a sub-pad.
   RPadBase *fParent = nullptr; /// The parent pad, if this pad has one.

   /// Drawing options, containing the size (in parent coordinates!)
   DrawingOpts fOpts;

   /// Position of the pad in the parent's (!) coordinate system.
   RPadPos fPos = fOpts.GetPos();

   /// Size of the pad in the parent's (!) coordinate system.
   RPadExtent fSize = fOpts.GetSize(); // {640_px, 400_px};

public:
   friend std::unique_ptr<RPadDrawable> GetDrawable(std::unique_ptr<RPad> &&pad);

   /// Create a topmost, non-paintable pad.
   RPad() = default;

   /// Create a child pad.
   RPad(RPadBase &parent, const RPadPos &pos, const RPadExtent &size): fParent(&parent), fPos(pos), fSize(size) {}

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

   /// Drawing options.
   DrawingOpts &GetDrawingOpts() { return fOpts; }

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
};

/** \class RPadDrawable
   Draw a RPad, by drawing its contained graphical elements at the pad offset in the parent pad.'
   */
class RPadDrawable: public RDrawableBase<RPadDrawable> {
private:
   const std::shared_ptr<RPad> fPad; ///< The pad to be painted

public:
   /// Move a sub-pad into this (i.e. parent's) list of drawables.
   RPadDrawable(const std::shared_ptr<RPad> &pPad, const RPad::DrawingOpts& opts = {});

   /// Paint primitives from the pad.
   void Paint(Internal::RPadPainter &) final;

   RPad *Get() const { return fPad.get(); }

   /// Drawing options.
   RPad::DrawingOpts &GetOptions() { return fPad->GetDrawingOpts(); }
};

template <class... ARGS>
inline std::shared_ptr<RPadDrawable> GetDrawable(std::unique_ptr<RPad> &&pad, ARGS... args)
{
   return std::make_shared<RPadDrawable>(std::move(pad), RPad::DrawingOpts(args...));
}

template <class... ARGS>
inline std::shared_ptr<RPadDrawable> GetDrawable(const std::shared_ptr<RPad> &pad, ARGS... args)
{
   return std::make_shared<RPadDrawable>(pad, RPad::DrawingOpts(args...));
}

} // namespace Experimental
} // namespace ROOT

#endif
