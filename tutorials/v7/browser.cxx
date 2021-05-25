/// \file
/// \ingroup tutorial_v7
///
/// \macro_code
///
/// \date 2019-05-29
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
/// \author Bertrand Bellenot <Bertrand.Bellenot@cern.ch>
/// \author Sergey Linev <S.Linev@gsi.de>

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

// macro must be here to let macro work on Windows
R__LOAD_LIBRARY(libROOTBrowserv7)

#include <ROOT/RBrowser.hxx>
#include <ROOT/RDirectory.hxx>

using namespace ROOT::Experimental;

void browser()
{
   // create browser
   auto br = std::make_shared<RBrowser>();

   // add to global list - avoid auto deletion
   RDirectory::Heap().Add("browser", br);
}

