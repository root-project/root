/// \file TCanvasPainter.cxx
/// \ingroup CanvasPainter ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2017-05-31
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!


/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/TVirtualCanvasPainter.hxx"
#include "ROOT/TCanvas.hxx"
#include <ROOT/TLogger.hxx>

#include <memory>
#include <string>
#include <vector>

namespace {

/** \class TCanvasPainter
  Handles TCanvas communication with THttpServer.
  */

class TCanvasPainter: public ROOT::Experimental::Internal::TVirtualCanvasPainter /*, THttpSocketListener*/ {
private:
  /// The canvas we are painting. It might go out of existance while painting.
  const ROOT::Experimental::TCanvas& fCanvas;
  /// Disable copy construction.
  TCanvasPainter(const TCanvasPainter&) = delete;

  /// Disable assignment.
  TCanvasPainter& operator=(const TCanvasPainter&) = delete;

  /// Send the canvas primitives to the THttpServer.
  void SendCanvas();

public:
  /// Create a TVirtualCanvasPainter for the given canvas.
  /// The painter observes it; it needs to know should the TCanvas be deleted.
  TCanvasPainter(const ROOT::Experimental::TCanvas& canv): fCanvas(canv) {}

  // void ReactToSocketNews(...) override { SendCanvas(); }

/** \class CanvasPainterGenerator
    Creates TCanvasPainter objects.
  */

  class GeneratorImpl: public Generator {
  public:

    /// Create a new TCanvasPainter to paint the given TCanvas.
    std::unique_ptr<TVirtualCanvasPainter> Create(const ROOT::Experimental::TCanvas& canv) const override {
      return std::make_unique<TCanvasPainter>(canv);
    }
    ~GeneratorImpl() = default;

    /// Set TVirtualCanvasPainter::fgGenerator to a new GeneratorImpl object.
    static void SetGlobalPainter() {
      if (TVirtualCanvasPainter::fgGenerator) {
        R__ERROR_HERE("CanvasPainter") << "Generator is already set! Skipping second initialization.";
        return;
      }
      TVirtualCanvasPainter::fgGenerator.reset(new GeneratorImpl());
    }

    /// Release the GeneratorImpl object.
    static void ResetGlobalPainter() {
      TVirtualCanvasPainter::fgGenerator.reset();
    }
  };
};



/** \class TCanvasPainterReg
  Registers TCanvasPainterGenerator as generator with ROOT::Experimental::Internal::TVirtualCanvasPainter.
  */
struct TCanvasPainterReg {
  TCanvasPainterReg() {
    TCanvasPainter::GeneratorImpl::SetGlobalPainter();
  }
  ~TCanvasPainterReg() {
    TCanvasPainter::GeneratorImpl::ResetGlobalPainter();
  }
} canvasPainterReg;

/// \}

} // unnamed namespace



void TCanvasPainter::SendCanvas() {
  for (auto &&drawable: fCanvas.GetPrimitives()) {
    drawable->Paint(*this);
  }
}
