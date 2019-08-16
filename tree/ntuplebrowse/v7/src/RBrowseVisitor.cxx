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
#include <ROOT/RNTupleBrowser.hxx>

//------------------------ RBrowseVisitor -------------------------------

void ROOT::Experimental::RBrowseVisitor::VisitField(const ROOT::Experimental::Detail::RFieldBase &field, int level)
{
   // only print direct subfields. fType is reset to its default value.
   if (level != 1) {
      fType = fieldDatatype_noHist;
      return;
   }
   
   // only leaf-fields should display a histogram. fType = fieldDatatype_notkLeaf ensures no histogram is displayed.
   if (field.GetStructure() != kLeaf)
      fType = fieldDatatype_notkLeaf;
   
   // Currently subfield of a vector field shouldn't display a histogram. TODO (lesimon): think if it should be displayed and if so how.
   if (std::string(field.GetParent()->GetType(), 0, 12).compare("std::vector<") == 0) { fType = fieldDatatype_parentIsVec;
   }
   
   // if the field has no children/subfields a RNTupleFielElement is created, which displays a histrogram upon double-click. If the field has children/subfields a RNTupleElementFolder is created, which displays it's subfields when double-clicked.
   if (field.GetLevelInfo().GetNumChildren() == 0) {
      ROOT::Experimental::RNTupleFieldElement* f = new ROOT::Experimental::RNTupleFieldElement(field.GetName(), fNTupleBrowserPtr, fType);
      f->AddBrowse(fBrowser);
      fNTupleBrowserPtr->fNTupleBrowsePtrVec.emplace_back(f);
   } else {
      ROOT::Experimental::RNTupleFieldElementFolder* f = new ROOT::Experimental::RNTupleFieldElementFolder(field.GetName(), &field, fNTupleBrowserPtr);
      f->AddBrowse(fBrowser);
      fNTupleBrowserPtr->fNTupleBrowsePtrVec.emplace_back(f);
   }
   
   // fType is set to its default value, or else it's possible that e.g. std::string field retains the previous fType fieldDatatype_float. fType is only changed when it passes a field with numerical value, non-kLeaf, or field which has a std::vector parent.
   fType = fieldDatatype_noHist;
}
