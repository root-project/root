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
   if (level != 1) { fType = numericDatatype_noHist; return; }
   if (field.GetStructure() != kLeaf) fType = numericDatatype_notkLeaf;
   if (std::string(field.GetParent()->GetType(), 0, 12).compare("std::vector<") == 0) { fType = numericDatatype_parentIsVec; }
   if (field.GetLevelInfo().fNumChildren == 0) { ROOT::Experimental::RNTupleFieldElement* f = new ROOT::Experimental::RNTupleFieldElement(field.GetName(), ntplb, fType);
      f->AddBrowse(b);
      ntplb->fNTupleBrowsePtrVec.emplace_back(f);
      // fType is only changed when it passes a field with numerical value. If fType is not changed here, the fType from the previous field is conserved, which can lead to a RField<std::string> trying to draw a histogramm in NTupleFieldElement::Browse
   }
   else { ROOT::Experimental::RNTupleFieldElementFolder* f = new ROOT::Experimental::RNTupleFieldElementFolder(field.GetName());
      f->SetRNTupleBrowser(ntplb);
      f->SetField(field);
      f->AddBrowse(b);
      ntplb->fNTupleBrowsePtrVec.emplace_back(f);
   }
   fType = numericDatatype_noHist;
}
