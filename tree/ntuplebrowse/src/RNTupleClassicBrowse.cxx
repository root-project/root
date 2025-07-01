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
#include <ROOT/RNTupleClassicBrowse.hxx>
#include <ROOT/RNTupleDescriptor.hxx>
#include <ROOT/RNTupleReader.hxx>

#include <TBrowser.h>
#include <TObject.h>

#include <memory>
#include <string>

namespace {

class RFieldBrowsable final : public TObject {
private:
   std::shared_ptr<ROOT::RNTupleReader> fReader;
   ROOT::DescriptorId_t fFieldId = ROOT::kInvalidDescriptorId;

public:
   RFieldBrowsable(std::shared_ptr<ROOT::RNTupleReader> reader, ROOT::DescriptorId_t fieldId)
      : fReader(reader), fFieldId(fieldId)
   {
   }

   void Browse(TBrowser *b) final
   {
      if (!b)
         return;

      const auto &desc = fReader->GetDescriptor();
      for (const auto &f : desc.GetFieldIterable(fFieldId)) {
         b->Add(new RFieldBrowsable(fReader, f.GetId()), f.GetFieldName().c_str());
      }
   }
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
