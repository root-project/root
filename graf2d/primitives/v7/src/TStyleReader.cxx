/// \file TDrawingOptsReader.cxx
/// \ingroup Gpad ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2017-09-26
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TStyleReader.hxx" // in src/

#include <ROOT/TColor.hxx>
#include <ROOT/TLogger.hxx>

#include <TROOT.h>
#include <TSystem.h>

using namespace ROOT::Experimental;
using namespace ROOT::Experimental::Internal;

TDrawingOptsReader::Attrs_t TDrawingOptsReader::ReadDefaults()
{
   Attrs_t ret;
   TDrawingOptsReader reader(ret);
   reader.AddFromStyleFile(std::string(TROOT::GetEtcDir().Data()) + "/system.rootstylerc");
   reader.AddFromStyleFile(gSystem->GetHomeDirectory() + "/.rootstylerc");
   reader.AddFromStyleFile(".rootstylerc");
   return ret;
}

bool TDrawingOptsReader::AddFromStyleFile(std::string_view filename)
{
   R__WARNING_HERE("Gpad") << "Not implemented yet, while reading style file " << filename;
   return false;
}
