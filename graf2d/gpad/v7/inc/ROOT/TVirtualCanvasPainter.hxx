/// \file ROOT/TVirtualCanvasPainter.hxx
/// \ingroup Gpad ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2017-05-31
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_TVirtualCanvasPainter
#define ROOT7_TVirtualCanvasPainter

#include <memory>

#include "ROOT/TDisplayItem.hxx"

namespace ROOT {
namespace Experimental {

class TCanvas;

namespace Internal {

/** \class ROOT::Experimental::Internal::TVirtualCanvasPainter
  Abstract interface for painting a canvas.
  */

class TVirtualCanvasPainter {
protected:
   class Generator {
   public:
      /// Abstract interface to create a TVirtualCanvasPainter implementation.
      virtual std::unique_ptr<TVirtualCanvasPainter> Create(const TCanvas &canv, bool batch_mode) const = 0;
      /// Default destructor.
      virtual ~Generator();
   };
   /// The generator for implementations.
   static std::unique_ptr<Generator> fgGenerator;

public:
   /// Default destructor.
   virtual ~TVirtualCanvasPainter();

   /// returns true is canvas used in batch mode
   virtual bool IsBatchMode() const { return true; }

   /// add display item to the canvas
   virtual void AddDisplayItem(TDisplayItem *item) = 0;

   /// perform special action when drawing is ready
   virtual void DoWhenReady(const std::string &, const std::string &) = 0;

   /// Loads the plugin that implements this class.
   static std::unique_ptr<TVirtualCanvasPainter> Create(const TCanvas &canv, bool batch_mode = false);
};
} // namespace Internal
} // namespace Experimental
} // namespace ROOT

#endif // ROOT7_TVirtualCanvasPainter
