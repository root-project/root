/// \file
/// \ingroup tutorial_v7
///
/// \macro_code
///
/// \date 2019-11-01
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
/// \author Sergey Linev <S.Linev@gsi.de>

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RFileDialog.hxx>
#include <ROOT/RDirectory.hxx>

using namespace ROOT::Experimental;

void filedialog(int kind = 0)
{
   std::string fileName;

   // example of sync methods, blocks until name is selected
   switch (kind) {
      case 1: fileName = RFileDialog::OpenFile("Open file title"); break;
      case 2: fileName = RFileDialog::SaveAsFile("Save as title"); break;
      case 3: fileName = RFileDialog::NewFile("New File title"); break;
   }

   if (kind > 0) {
      printf("fileName %s\n", fileName.c_str());
      return;
   }

   auto panel = std::make_shared<RFileDialog>(RFileDialog::kOpenFile, "Open file (async) title");
   // add to global list
   RDirectory::Heap().Add("filedialog", panel);

   panel->SetCallback([](const std::string &res) {
      printf("Selected %s\n", res.c_str());
      // remove from global list
      RDirectory::Heap().Remove("filedialog");
   });

   panel->Show();
}

