/// \file ROOT/RBrowseVisitor.cxx
/// \ingroup NTupleBrowse ROOT7
/// \author Simon Leisibach <simon.satoshi.rene.leisibach@cern.ch>
/// \date 2019-07-30
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RBrowseVisitor.hxx>

//------------------------ RBrowseVisitor -------------------------------

void ROOT::Experimental::RBrowseVisitor::VisitField(const ROOT::Experimental::Detail::RFieldBase &field, int level)
{
   // only print direct subfields (direct subfield <=> (level == 1)).
   if (level != 1) {
      return;
   }
   
   // if the field has no children/subfields a RNTupleBrowseLeaf is created, which displays a histrogram upon double-click. If the field has children/subfields a RNTupleBrowseFolder is created, which displays it's subfields when double-clicked.
   if (field.GetLevelInfo().GetNumChildren() == 0) {
      ROOT::Experimental::RNTupleBrowseLeaf* f = new ROOT::Experimental::RNTupleBrowseLeaf(fNTupleBrowserPtr, &field);
      f->AddBrowse(fBrowser);
      fNTupleBrowserPtr->fNTupleBrowsePtrVec.emplace_back(f);
   } else {
      ROOT::Experimental::RNTupleBrowseFolder* f = new ROOT::Experimental::RNTupleBrowseFolder(fNTupleBrowserPtr, &field);
      f->AddBrowse(fBrowser);
      fNTupleBrowserPtr->fNTupleBrowsePtrVec.emplace_back(f);
   }
}


