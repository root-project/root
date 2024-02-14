#include "ntuple_test.hxx"

#include "Unsplit.hxx"

TEST(RField, UnsplitDirect)
{
   FileRaii fileGuard("test_ntuple_rfield_unsplit_direct.root");
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

TEST(RField, UnsplitMember)
{
   FileRaii fileGuard("test_ntuple_rfield_unsplit_member.root");
   {
      auto model = RNTupleModel::Create();
      auto ptrUnsplitMember = model->MakeField<UnsplitMember>("event");
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      ptrUnsplitMember->a = 1.0;
      UnsplitMember inner;
      inner.a = 2.0;
      ptrUnsplitMember->v.push_back(inner);
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   auto ptrUnsplitMember = reader->GetModel().GetDefaultEntry().GetPtr<UnsplitMember>("event");

   ASSERT_EQ(1U, reader->GetNEntries());
   reader->LoadEntry(0);
   EXPECT_FLOAT_EQ(1.0, ptrUnsplitMember->a);
   EXPECT_EQ(1u, ptrUnsplitMember->v.size());
   EXPECT_FLOAT_EQ(2.0, ptrUnsplitMember->v.at(0).a);
}
