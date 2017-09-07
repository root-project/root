/// \file TVirtualCanvasPainter.cxx
/// \ingroup Gpad ROOT7
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

#include <ROOT/TVirtualCanvasPainter.hxx>

#include <ROOT/TLogger.hxx>
#include <TSystem.h> // TSystem::Load

#include <exception>

namespace {
static void LoadCanvasPainterLibrary() {
  if (gSystem->Load("libROOTCanvasPainter") != 0)
    R__ERROR_HERE("Gpad") << "Loading of libROOTCanvasPainter failed!";
}
} // unnamed namespace

std::unique_ptr<ROOT::Experimental::Internal::TVirtualCanvasPainter::Generator>
  ROOT::Experimental::Internal::TVirtualCanvasPainter::fgGenerator;


/// The implementation is here to pin the vtable.
ROOT::Experimental::Internal::TVirtualCanvasPainter::~TVirtualCanvasPainter() = default;

std::unique_ptr<ROOT::Experimental::Internal::TVirtualCanvasPainter> ROOT::Experimental::Internal::
   TVirtualCanvasPainter::Create(const TCanvas &canv, bool batch_mode)
{
   if (!fgGenerator) {
      LoadCanvasPainterLibrary();
      if (!fgGenerator) {
         R__ERROR_HERE("Gpad") << "TVirtualCanvasPainter::Generator failed to register!";
         throw std::runtime_error("TVirtualCanvasPainter::Generator failed to initialize");
      }
   }
   return fgGenerator->Create(canv, batch_mode);
}

/// The implementation is here to pin the vtable.
ROOT::Experimental::Internal::TVirtualCanvasPainter::Generator::~Generator() = default;

