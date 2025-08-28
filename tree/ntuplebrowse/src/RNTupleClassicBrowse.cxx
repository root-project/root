/// \file RNTupleClassicBrowse.cxx
/// \ingroup NTuple
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2025-06-30

/*************************************************************************
 * Copyright (C) 1995-2025, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleBrowseUtils.hxx>
#include <ROOT/RNTupleClassicBrowse.hxx>
#include <ROOT/RNTupleDrawVisitor.hxx>
#include <ROOT/RNTupleDescriptor.hxx>
#include <ROOT/RNTupleReader.hxx>

#include <TBrowser.h>
#include <TObject.h>
#include <TPad.h>
#include <TText.h>

#include <memory>
#include <string>

namespace {

class RFieldBrowsable final : public TObject {
private:
   std::shared_ptr<ROOT::RNTupleReader> fReader;
   ROOT::DescriptorId_t fFieldId = ROOT::kInvalidDescriptorId;
   ROOT::DescriptorId_t fBrowsableFieldId = ROOT::kInvalidDescriptorId;
   bool fIsLeaf = false;
   std::unique_ptr<TH1> fHistogram;
   std::string fFieldName;
   std::string fTypeName;

public:
   RFieldBrowsable(std::shared_ptr<ROOT::RNTupleReader> reader, ROOT::DescriptorId_t fieldId)
      : fReader(reader), fFieldId(fieldId)
   {
      const auto &desc = fReader->GetDescriptor();
      fBrowsableFieldId = ROOT::Internal::GetNextBrowsableField(fFieldId, desc);
      fIsLeaf = desc.GetFieldDescriptor(fBrowsableFieldId).GetLinkIds().empty();
      fFieldName = desc.GetFieldDescriptor(fFieldId).GetFieldName();
      fTypeName = desc.GetFieldDescriptor(fFieldId).GetTypeName();
   }

   void Browse(TBrowser *b) final
   {
      if (!b)
         return;

      const auto &desc = fReader->GetDescriptor();

      if (fIsLeaf) {
         if (!gPad)
            return;

         auto view = fReader->GetView<void>(desc.GetQualifiedFieldName(fBrowsableFieldId));

         ROOT::Internal::RNTupleDrawVisitor drawVisitor(fReader, desc.GetFieldDescriptor(fFieldId).GetFieldName());
         view.GetField().AcceptVisitor(drawVisitor);
         fHistogram = std::unique_ptr<TH1>(drawVisitor.MoveHist());
         if (fHistogram->GetEntries() == 0) {
            gPad->DrawFrame(-1., -1., 1., 1.);
            TText *textEmpty = new TText(0., 0., "Empty");
            textEmpty->SetTextAlign(22);
            textEmpty->SetTextFont(42);
            textEmpty->SetTextSize(0.1);
            textEmpty->SetTextColor(1);
            textEmpty->Draw();
         } else {
            fHistogram->Draw();
         }
         gPad->Update();
      } else {
         for (const auto &f : desc.GetFieldIterable(fBrowsableFieldId)) {
            b->Add(new RFieldBrowsable(fReader, f.GetId()), f.GetFieldName().c_str());
         }
      }
   }

   bool IsFolder() const final { return !fIsLeaf; }
   const char *GetIconName() const final { return IsFolder() ? "RNTuple-folder" : "RNTuple-leaf"; }

   const char *GetName() const final { return fFieldName.c_str(); }
   const char *GetTitle() const final { return fTypeName.c_str(); }
};

} // anonymous namespace

void ROOT::Internal::BrowseRNTuple(const void *ntuple, TBrowser *b)
{
   if (!b)
      return;

   std::shared_ptr<ROOT::RNTupleReader> reader = RNTupleReader::Open(*static_cast<const ROOT::RNTuple *>(ntuple));
   const auto &desc = reader->GetDescriptor();
   for (const auto &f : desc.GetTopLevelFields()) {
      b->Add(new RFieldBrowsable(reader, f.GetId()), f.GetFieldName().c_str());
   }
}
