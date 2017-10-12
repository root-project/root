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
#include "ROOT/TFrame.hxx"
#include "ROOT/TPadExtent.hxx"
#include "ROOT/TPadPos.hxx"
#include "ROOT/TypeTraits.hxx"

namespace ROOT {
namespace Experimental {

class TPad;

namespace Internal {
class TVirtualCanvasPainter;
}

/** \class ROOT::Experimental::TPadBase
  Base class for graphic containers for `TDrawable`-s.
  */

class TPadBase {
public:
   using Primitives_t = std::vector<std::unique_ptr<TDrawable>>;

private:
   /// Content of the pad.
   Primitives_t fPrimitives;

   /// TFrame with user coordinate system, if used by this pad.
   std::unique_ptr<TFrame> fFrame;

   /// Disable copy construction.
   TPadBase(const TPadBase &) = delete;

   /// Disable assignment.
   TPadBase &operator=(const TPadBase &) = delete;

   /// Adds a `DRAWABLE` to `fPrimitives`, returning the drawing options as given by `DRAWABLE::Options()`.
   template <class DRAWABLE>
   auto &AddDrawable(std::unique_ptr<DRAWABLE> &&uPtr)
   {
      DRAWABLE &drw = *uPtr;
      fPrimitives.emplace_back(std::move(uPtr));
      return drw.GetOptions();
   }

protected:
   /// Allow derived classes to default construct a TPadBase.
   TPadBase() = default;

public:
   virtual ~TPadBase();

   /// Divide this pad into a grid of subpad with padding in between.
   /// \param nHoriz Number of horizontal pads.
   /// \param nVert Number of vertical pads.
   /// \param padding Padding between pads.
   /// \returns vector of vector (ret[x][y]) of created pads.
   std::vector<std::vector<TPad *>> Divide(int nHoriz, int nVert, const TPadExtent &padding = {});

   /// Add something to be painted.
   /// The pad observes what's lifetime through a weak pointer.
   template <class T>
   auto &Draw(const std::shared_ptr<T> &what)
   {
      // Requires GetDrawable(what) to be known!
      return AddDrawable(GetDrawable(what, *this));
   }

   /// Add something to be painted. The pad claims ownership.
   template <class T>
   auto &Draw(std::unique_ptr<T> &&what)
   {
      // Requires GetDrawable(what) to be known!
      return AddDrawable(GetDrawable(std::move(what), *this));
   }

   /// Add a copy of something to be painted.
   template <class T, class = typename std::enable_if<!ROOT::TypeTraits::IsSmartOrDumbPtr<T>::value>::type>
   auto &Draw(const T &what)
   {
      // Requires GetDrawable(what) to be known!
      return Draw(std::make_unique<T>(what));
   }

   /// Remove an object from the list of primitives.
   // TODO: void Wipe(???);

   /// Wipe the pad by clearing the list of primitives.
   void Wipe()
   {
      fPrimitives.clear();
   }

   /// Get the elements contained in the canvas.
   const Primitives_t &GetPrimitives() const { return fPrimitives; }

   /// Convert a `Pixel` position to Canvas-normalized positions.
   virtual std::array<TPadCoord::Normal, 2> PixelsToNormal(const std::array<TPadCoord::Pixel, 2> &pos) const = 0;

   /// Access to the top-most canvas, if any (const version).
   virtual const TCanvas &GetCanvas() const = 0;

   /// Access to the top-most canvas, if any (non-const version).
   virtual TCanvas &GetCanvas() = 0;

   /// Convert user coordinates to normal coordinates.
   std::array<TPadCoord::Normal, 2> UserToNormal(const std::array<TPadCoord::User, 2> &pos) const
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
   TPadBase *fParent = nullptr; //-> This must never be nullptr!

   /// Size of the pad in the parent's (!) coordinate system.
   TPadExtent fSize;

public:
   friend std::unique_ptr<TPadDrawable> GetDrawable(std::unique_ptr<TPad> &&pad, const TPadBase &parent);

   /// Create a child pad.
   TPad(TPadBase &parent, const TPadExtent &size): fParent(&parent), fSize(size) {}

   /// Destructor to have a vtable.
   virtual ~TPad();

   /// Access to the parent pad (const version).
   const TPadBase &GetParent() const { return *fParent; }

   /// Access to the parent pad (non-const version).
   TPadBase &GetParent() { return *fParent; }

   /// Access to the top-most canvas (const version).
   const TCanvas &GetCanvas() const override { return fParent->GetCanvas(); }

   /// Access to the top-most canvas (non-const version).
   TCanvas &GetCanvas() override { return fParent->GetCanvas(); }

   /// Get the size of the pad in parent (!) coordinates.
   const TPadExtent &GetSize() const { return fSize; }

   /// Convert a `Pixel` position to Canvas-normalized positions.
   std::array<TPadCoord::Normal, 2> PixelsToNormal(const std::array<TPadCoord::Pixel, 2> &pos) const override
   {
      std::array<TPadCoord::Normal, 2> posInParentNormal = fParent->PixelsToNormal(pos);
      std::array<TPadCoord::Normal, 2> myPixelInNormal =
         fParent->PixelsToNormal({{fSize.fHoriz.fPixel, fSize.fVert.fPixel}});
      std::array<TPadCoord::Normal, 2> myUserInNormal =
         fParent->UserToNormal({{fSize.fHoriz.fUser, fSize.fVert.fUser}});
      // If the parent says pos is at 0.6 in normal coords, and our size converted to normal is 0.2, then pos in our
      // coord system is 3.0!
      return {{posInParentNormal[0] / (fSize.fHoriz.fNormal + myPixelInNormal[0] + myUserInNormal[0]),
               posInParentNormal[1] / (fSize.fVert.fNormal + myPixelInNormal[1] + myUserInNormal[1])}};
   }

   /// Convert a TPadPos to [x, y] of normalized coordinates.
   std::array<TPadCoord::Normal, 2> ToNormal(const Internal::TPadHorizVert &pos) const
   {
      std::array<TPadCoord::Normal, 2> pixelsInNormal = PixelsToNormal({{pos.fHoriz.fPixel, pos.fVert.fPixel}});
      std::array<TPadCoord::Normal, 2> userInNormal = UserToNormal({{pos.fHoriz.fUser, pos.fVert.fUser}});
      return {{pos.fHoriz.fNormal + pixelsInNormal[0] + userInNormal[0],
               pos.fVert.fNormal + pixelsInNormal[1] + userInNormal[1]}};
   }
};

/** \class TPadDrawingOpts
 Drawing options for a TPad
 */

class TPadDrawingOpts: public TDrawingOptsBase<TPadDrawingOpts> {
   TPadPos fPos; ///< Offset with respect to parent TPad.
public:
   TPadDrawingOpts(TPadBase &parent): TDrawingOptsBase<TPadDrawingOpts>(parent, "Pad") {}

   /// Set the position of this pad with respect to the parent pad.
   TPadDrawingOpts &At(const TPadPos &pos)
   {
      fPos = pos;
      return *this;
   }
};

/** \class TPadDrawable
   Draw a TPad, by drawing its contained graphical elements at the pad offset in the parent pad.'
   */
class TPadDrawable: public TDrawable {
private:
   const std::unique_ptr<TPad> fPad; ///< The pad to be painted
   TPadDrawingOpts fOpts;            ///< The drawing options.

public:
   /// Move a sub-pad into this (i.e. parent's) list of drawables.
   TPadDrawable(std::unique_ptr<TPad> &&pPad, TPadBase &parent);

   /// Paint the pad.
   void Paint(Internal::TVirtualCanvasPainter & /*canv*/) final
   {
      // FIXME: and then what? Something with fPad.GetListOfPrimitives()?
   }

   TPad *Get() const { return fPad.get(); }

   /// Drawing options.
   TPadDrawingOpts &GetOptions() { return fOpts; }
};

inline std::unique_ptr<TPadDrawable> GetDrawable(std::unique_ptr<TPad> &&pad, TPadBase &parent)
{
   return std::make_unique<TPadDrawable>(std::move(pad), parent);
}

} // namespace Experimental
} // namespace ROOT

#endif
