#include "ntuple_test.hxx"

#include <TMemFile.h>

TEST(RField, Unsplit)
{
   FileRaii fileGuard("test_ntuple_rfield_unsplit.root");
   {
      auto model = RNTupleModel::Create();
      model->AddField(std::make_unique<ROOT::Experimental::RUnsplitField>("pt", "std::vector<float>"));
      auto ptrPt = model->GetDefaultEntry().GetPtr<std::vector<float>>("pt");
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      ptrPt->push_back(1.0);
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   auto ptrPt = reader->GetModel().GetDefaultEntry().GetPtr<std::vector<float>>("pt");

   ASSERT_EQ(1U, reader->GetNEntries());
   reader->LoadEntry(0);
   EXPECT_EQ(1u, ptrPt->size());
   EXPECT_FLOAT_EQ(1.0, ptrPt->at(0));
}
