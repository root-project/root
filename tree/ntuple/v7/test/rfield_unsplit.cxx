#include "ntuple_test.hxx"

#include <TDictAttributeMap.h>

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
   auto cl = TClass::GetClass("CyclicMember");
   cl->CreateAttributeMap();
   cl->GetAttributeMap()->AddProperty("rntuple.split", "false");

   FileRaii fileGuard("test_ntuple_rfield_unsplit_member.root");
   {
      auto model = RNTupleModel::Create();
      auto ptrClassWithUnsplitMember = model->MakeField<ClassWithUnsplitMember>("event");
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      ptrClassWithUnsplitMember->fA = 1.0;
      CyclicMember inner;
      inner.fB = 3.0;
      ptrClassWithUnsplitMember->fUnsplit.fB = 2.0;
      ptrClassWithUnsplitMember->fUnsplit.fV.push_back(inner);
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   auto ptrClassWithUnsplitMember = reader->GetModel().GetDefaultEntry().GetPtr<ClassWithUnsplitMember>("event");

   ASSERT_EQ(1U, reader->GetNEntries());
   reader->LoadEntry(0);
   EXPECT_FLOAT_EQ(1.0, ptrClassWithUnsplitMember->fA);
   EXPECT_FLOAT_EQ(2.0, ptrClassWithUnsplitMember->fUnsplit.fB);
   EXPECT_EQ(1u, ptrClassWithUnsplitMember->fUnsplit.fV.size());
   EXPECT_FLOAT_EQ(3.0, ptrClassWithUnsplitMember->fUnsplit.fV.at(0).fB);
   EXPECT_EQ(0u, ptrClassWithUnsplitMember->fUnsplit.fV.at(0).fV.size());
}
