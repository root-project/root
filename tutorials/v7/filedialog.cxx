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

// Show how RFileDialog can be used in sync and async modes
// Normally file dialogs will be used inside other widgets as ui5 dialogs
// By default, dialog starts in async mode - means macro immediately returns to command line
// To start OpenFile dialog in sync mode, call `root "filedialog.cxx(1)" -q`.
// Once file is selected, root execution will be stopped


using namespace ROOT::Experimental;

void filedialog(int kind = 0)
{
   std::string fileName;

   // example of sync methods, blocks until name is selected
   switch (kind) {
      case 1: fileName = RFileDialog::OpenFile("OpenFile title"); break;
      case 2: fileName = RFileDialog::SaveAs("SaveAs title", "newfile.xml"); break;
      case 3: fileName = RFileDialog::NewFile("NewFile title", "test.txt"); break;
   }

   if (kind > 0) {
      printf("Selected file: %s\n", fileName.c_str());
      return;
   }

   auto dialog = std::make_shared<RFileDialog>(RFileDialog::kOpenFile, "OpenFile dialog in async mode");

   dialog->SetNameFilters({ "C++ files (*.cxx *.cpp *.c *.C)", "Image files (*.png *.jpg *.jpeg)", "Text files (*.txt)", "Any files (*)" });

   dialog->SetSelectedFilter("C++ files");

   // use dialog capture to keep reference until file name is selected
   dialog->SetCallback([dialog](const std::string &res) mutable {
      printf("Selected file: %s\n", res.c_str());

      // cleanup dialog - actually not needed, lambda is cleaned up after that call anyway
      // dialog.reset();
   });

   dialog->Show();
}

