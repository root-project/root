#include <ROOT/RFieldBase.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleReader.hxx>
#include <ROOT/RNTupleWriter.hxx>

#include <ROOT/TestSupport.hxx>

#include <cstdio>
#include <string>
#include <utility>

#include "StreamerBeyond.hxx"
#include "gtest/gtest.h"

TEST(RField, StreamerBeyond)
{
   ROOT::TestSupport::FileRaii fileGuard("test_ntuple_rfield_streamer_beyond.root");

   {
      auto model = ROOT::RNTupleModel::Create();
      auto f = ROOT::RFieldBase::Create("f", "StreamerBeyond").Unwrap();
      EXPECT_TRUE(dynamic_cast<ROOT::RStreamerField *>(f.get()));
      model->AddField(std::move(f));
      auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());

      auto ptr = writer->GetModel().GetDefaultEntry().GetPtr<StreamerBeyond>("f");
      ptr->fOne = std::vector<std::int64_t>(100000000, -1);
      ptr->fTwo = std::vector<std::int64_t>(100000000, -2);

      writer->Fill();
   }

   auto reader = ROOT::RNTupleReader::Open("ntpl", fileGuard.GetPath());
   ASSERT_EQ(1u, reader->GetNEntries());
   StreamerBeyond sb;
   auto view = reader->GetView("f", &sb, "StreamerBeyond");

   view(0);

   auto ptr = view.GetValue().GetPtr<StreamerBeyond>();
   EXPECT_EQ(100000000u, ptr->fOne.size());
   EXPECT_EQ(-1, ptr->fOne.at(1000));
   EXPECT_EQ(100000000u, ptr->fTwo.size());
   EXPECT_EQ(-2, ptr->fTwo.at(2000));
   EXPECT_EQ(-2, ptr->fTwo.at(99999999u));
}
