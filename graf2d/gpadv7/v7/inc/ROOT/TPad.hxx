/// \file ROOT/TPad.hxx
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

#ifndef ROOT7_TPad
#define ROOT7_TPad

#include <memory>
#include <vector>

#include "ROOT/TDrawable.hxx"
#include "ROOT/TDrawingAttr.hxx"
#include "ROOT/TDrawingOptsBase.hxx"
#include "ROOT/TFrame.hxx"
#include "ROOT/TPadExtent.hxx"
#include "ROOT/TPadPos.hxx"
#include "ROOT/TypeTraits.hxx"

namespace ROOT {
namespace Experimental {

class TPad;
class TCanvas;

/** \class ROOT::Experimental::TPadBase
  Base class for graphic containers for `TDrawable`-s.
  */

class TPadBase {
public:
   using Primitives_t = std::vector<std::shared_ptr<TDrawable>>;

private:
   /// Content of the pad.
   Primitives_t fPrimitives;

   /// TFrame with user coordinate system, if used by this pad.
   std::unique_ptr<TFrame> fFrame;

   /// Disable copy construction.
   TPadBase(const TPadBase &) = delete;

   /// Disable assignment.
   TPadBase &operator=(const TPadBase &) = delete;

   void AssignUniqueID(std::shared_ptr<TDrawable> &ptr);

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
   /// Allow derived classes to default construct a TPadBase.
   TPadBase() = default;

public:
   virtual ~TPadBase();

   /// Divide this pad into a grid of subpads with padding in between.
   /// \param nHoriz Number of horizontal pads.
   /// \param nVert Number of vertical pads.
   /// \param padding Padding between pads.
   /// \returns vector of vector (ret[x][y]) of created pads.
   std::vector<std::vector<TPad *>> Divide(int nHoriz, int nVert, const TPadExtent &padding = {});

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
   bool Remove(TDrawingOptsBase& opts) {
      auto iter = std::find_if(fPrimitives.begin(), fPrimitives.end(),
         [&opts](const std::shared_ptr<TDrawable>& drawable) { return &drawable->GetOptionsBase() == &opts; });
      if (iter == fPrimitives.end())
         return false;
      iter->reset();
      return true;
   }

   std::shared_ptr<TDrawable> FindDrawable(const std::string &id) const;

   /// Wipe the pad by clearing the list of primitives.
   void Wipe()
   {
      fPrimitives.clear();
   }

   const TFrame *GetFrame() const { return fFrame.get(); }

   /// Get the elements contained in the canvas.
   const Primitives_t &GetPrimitives() const { return fPrimitives; }

   /// Convert a `Pixel` position to Canvas-normalized positions.
   virtual std::array<TPadLength::Normal, 2> PixelsToNormal(const std::array<TPadLength::Pixel, 2> &pos) const = 0;

   /// Access to the top-most canvas, if any (const version).
   virtual const TCanvas *GetCanvas() const = 0;

   /// Access to the top-most canvas, if any (non-const version).
   virtual TCanvas *GetCanvas() = 0;

   /// Convert user coordinates to normal coordinates.
   std::array<TPadLength::Normal, 2> UserToNormal(const std::array<TPadLength::User, 2> &pos) const
   {
      return fFrame->UserToNormal(pos);
   }
};

class TPadDrawable;

/** \class ROOT::Experimental::TPad
  Graphic container for `TDrawable`-s.
  */

class TPad: public TPadBase {
private:
   /// Pad containing this pad as a sub-pad.
   TPadBase *fParent = nullptr; /// The parent pad, if this pad has one.

   /// Size of the pad in the parent's (!) coordinate system.
   TPadExtent fSize = {1._normal, 1._normal}; // {640_px, 400_px};

public:
   friend std::unique_ptr<TPadDrawable> GetDrawable(std::unique_ptr<TPad> &&pad);

   /// Create a topmost, non-paintable pad.
   TPad() = default;

   /// Create a child pad.
   TPad(TPadBase &parent, const TPadExtent &size): fParent(&parent), fSize(size) {}

   /// Destructor to have a vtable.
   virtual ~TPad();

   /// Access to the parent pad (const version).
   const TPadBase *GetParent() const { return fParent; }

   /// Access to the parent pad (non-const version).
   TPadBase *GetParent() { return fParent; }

   /// Access to the top-most canvas (const version).
   const TCanvas *GetCanvas() const override { return fParent ? fParent->GetCanvas() : nullptr; }

   /// Access to the top-most canvas (non-const version).
   TCanvas *GetCanvas() override { return fParent ? fParent->GetCanvas() : nullptr; }

   /// Get the size of the pad in parent (!) coordinates.
   const TPadExtent &GetSize() const { return fSize; }

   /// Convert a `Pixel` position to Canvas-normalized positions.
   std::array<TPadLength::Normal, 2> PixelsToNormal(const std::array<TPadLength::Pixel, 2> &pos) const override
   {
      std::array<TPadLength::Normal, 2> posInParentNormal = fParent->PixelsToNormal(pos);
      std::array<TPadLength::Normal, 2> myPixelInNormal =
         fParent->PixelsToNormal({{fSize.fHoriz.fPixel, fSize.fVert.fPixel}});
      std::array<TPadLength::Normal, 2> myUserInNormal =
         fParent->UserToNormal({{fSize.fHoriz.fUser, fSize.fVert.fUser}});
      // If the parent says pos is at 0.6 in normal coords, and our size converted to normal is 0.2, then pos in our
      // coord system is 3.0!
      return {{posInParentNormal[0] / (fSize.fHoriz.fNormal + myPixelInNormal[0] + myUserInNormal[0]),
               posInParentNormal[1] / (fSize.fVert.fNormal + myPixelInNormal[1] + myUserInNormal[1])}};
   }

   /// Convert a TPadPos to [x, y] of normalized coordinates.
   std::array<TPadLength::Normal, 2> ToNormal(const Internal::TPadHorizVert &pos) const
   {
      std::array<TPadLength::Normal, 2> pixelsInNormal = PixelsToNormal({{pos.fHoriz.fPixel, pos.fVert.fPixel}});
      std::array<TPadLength::Normal, 2> userInNormal = UserToNormal({{pos.fHoriz.fUser, pos.fVert.fUser}});
      return {{pos.fHoriz.fNormal + pixelsInNormal[0] + userInNormal[0],
               pos.fVert.fNormal + pixelsInNormal[1] + userInNormal[1]}};
   }
};

/** \class TPadDrawingOpts
 Drawing options for a TPad
 */

class TPadDrawingOpts: public TDrawingOptsBase {
   TDrawingAttr<TPadPos> fPos{*this, "PadOffset"}; ///< Offset with respect to parent TPad.

public:
   TPadDrawingOpts() = default;

   /// Construct the drawing options.
   TPadDrawingOpts(const TPadPos& pos): fPos(*this, "PadOffset", pos) {}

   /// Set the position of this pad with respect to the parent pad.
   TPadDrawingOpts &At(const TPadPos &pos)
   {
      fPos = pos;
      return *this;
   }

   TDrawingAttr<TPadPos> &GetOffset() { return fPos; }
   const TDrawingAttr<TPadPos> &GetOffset() const { return fPos; }
};

/** \class TPadDrawable
   Draw a TPad, by drawing its contained graphical elements at the pad offset in the parent pad.'
   */
class TPadDrawable: public TDrawableBase<TPadDrawable> {
private:
   const std::unique_ptr<TPad> fPad; ///< The pad to be painted
   TPadDrawingOpts fOpts;            ///< The drawing options.

public:
   /// Move a sub-pad into this (i.e. parent's) list of drawables.
   TPadDrawable(std::unique_ptr<TPad> &&pPad, const TPadDrawingOpts& opts = {});

   /// Paint primitives from the pad.
   void Paint(Internal::TPadPainter &) final;

   TPad *Get() const { return fPad.get(); }

   /// Drawing options.
   TPadDrawingOpts &GetOptions() { return fOpts; }
};

template <class... ARGS>
inline std::shared_ptr<TPadDrawable> GetDrawable(std::unique_ptr<TPad> &&pad, ARGS... args)
{
   return std::make_shared<TPadDrawable>(std::move(pad), TPadDrawingOpts(args...));
}

} // namespace Experimental
} // namespace ROOT

#endif
