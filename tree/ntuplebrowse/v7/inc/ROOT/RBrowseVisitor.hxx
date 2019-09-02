/// \file ROOT/RBrowseVisitor.hxx
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

#ifndef ROOT7_RBrowseVisitor
#define ROOT7_RBrowseVisitor

#include <ROOT/RField.hxx>
#include <ROOT/RFieldVisitor.hxx>
#include <ROOT/RNTupleBrowser.hxx>

#include <TH1.h>

#include <cassert>
#include <cmath>
#include <limits.h>

class TBrowser;

namespace ROOT {
namespace Experimental {

// clang-format off
/**
\class ROOT::Experimental::RBrowseVisitor
\ingroup NTupleBrowse
\brief Visitor class which traverses fields to display them on the TBrowser.
    
RBrowseVisitor uses information about a field and creates an instance of RNTupleBrowseLeaf or RNTupleBrowseFolder.
*/
// clang-format on
class RBrowseVisitor : public Detail::RNTupleVisitor {
private:
   /// Is passed down to RNTupleBrowseLeaf or RNTupleBrowseFolder.
   TBrowser *fBrowser;
   /// Used to save created instance of RNTupleBrowseLeaf or RNTupleBrowseFolder in RNTupleBrowser and also passed down
   /// to RNTupleBrowseLeaf and RNTupleBrowseFolder.
   RNTupleBrowser *fNTupleBrowserPtr;

public:
   RBrowseVisitor(TBrowser *parb, RNTupleBrowser *parntplb) : fBrowser{parb}, fNTupleBrowserPtr{parntplb} {}

   /// Creates instance of RNTupleBrowseLeaf or RNTupleBrowseFolder and displays it in TBrowser.
   void VisitField(const Detail::RFieldBase &field, int level) final;
   // Do nothing for RootField
   void VisitRootField(const RFieldRoot & /*field*/, int /*level*/) final {}
};

// clang-format off
/**
\class ROOT::Experimental::RDisplayHistVisitor
\ingroup NTupleBrowse
\brief Visitor class which draws a histogram for fields with numerical data.
    
Visits fields displayed in TBrowser and draws a histogram for appropriate fields. Instances of this class are created
 when a field without subfields is double-clicked in TBrowser. (called by RNTupleBrowseLeaf::Browse(TBrowser* b))
*/
// clang-format on
class RDisplayHistVisitor : public Detail::RNTupleVisitor {
private:
   /// Allows to access RNTupleBrowser::fCurrentTH1F.
   RNTupleBrowser *fNTupleBrowserPtr;
   /// Allows to get entries of a field which will be displayed in the histogram.
   RNTupleReader *fNTupleReaderPtr;
   // Note: fNTupleBrowserPtr->GetReaderPtr() returns the last created RNTupleReader. This can be another RNTupleReader
   // than the one which contains information about the visited field. Therefore a separate member had to be created.

public:
   RDisplayHistVisitor(RNTupleBrowser *parntplb, RNTupleReader *readerPtr)
      : fNTupleBrowserPtr{parntplb}, fNTupleReaderPtr{readerPtr}
   {
   }

   void VisitField(const Detail::RFieldBase & /*field*/, int /*level*/) final {}
   void VisitRootField(const RFieldRoot & /*field*/, int /*level*/) final {}
   void VisitFloatField(const RField<float> &field, int /*level*/) { DrawHistogram<float>(field, false); }
   void VisitDoubleField(const RField<double> &field, int /*level*/) { DrawHistogram<double>(field, false); }
   void VisitInt32Field(const RField<std::int32_t> &field, int /*level*/) { DrawHistogram<std::int32_t>(field, true); }
   void VisitUInt32Field(const RField<std::uint32_t> &field, int /*level*/)
   {
      DrawHistogram<std::uint32_t>(field, true);
   }
   void VisitUInt64Field(const RField<std::uint64_t> &field, int /*level*/)
   {
      DrawHistogram<std::uint64_t>(field, true);
   }

   template <typename T>
   void DrawHistogram(const Detail::RFieldBase &field, bool isIntegraltype)
   {
      // only leaf-fields should display a histogram.
      if (field.GetStructure() != kLeaf)
         return;
      // Currently subfield of a vector/collection shouldn't display a histogram. TODO (lesimon): think if it should be
      // displayed and if so how.
      if (field.GetParent()->GetStructure() == kCollection)
         return;

      // for now only print fields directly attached to RootField, until RNTupleView is fixed. TODO(lesimon): Remove
      // this later.
      if (field.GetLevelInfo().GetLevel() != 1)
         return;

      auto ntupleView = fNTupleReaderPtr->GetView<T>(field.GetName());

      // if min = 3 and max = 10, a histogram with a x-axis range of 3 to 10 is created with 8 bins (3, 4, 5, 6, 7, 8,
      // 9, 10)
      double max{LONG_MIN}, min{LONG_MAX};
      // TODO(lesimon): Think how RNTupleView can be interated only once.
      for (auto i : fNTupleReaderPtr->GetViewRange()) {
         max = std::max(max, static_cast<double>(ntupleView(i)));
         min = std::min(min, static_cast<double>(ntupleView(i)));
      }
      if (min > max)
         return; // no histogram for empty field.

      // It doesn't make sense to create 100 bins if only integers from 3 to 10 are used to fill the histogram.
      int nbins = isIntegraltype ? std::min(100, static_cast<int>(std::round(max - min) + 1)) : 100;
      // deleting the old TH1-histogram after creating a new one makes cling complain if both histograms have the same
      // name.
      delete fNTupleBrowserPtr->fCurrentTH1F;
      auto h1 = new TH1F(field.GetName().c_str(), field.GetName().c_str(), nbins, min - 0.5, max + 0.5);
      for (auto i : fNTupleReaderPtr->GetViewRange()) {
         h1->Fill(ntupleView(i));
      }
      h1->Draw();
      fNTupleBrowserPtr->fCurrentTH1F = h1;
   }
};

} // namespace Experimental
} // namespace ROOT

#endif /* ROOT7_RBrowseVisitor */
