#include "ntuple_test.hxx"
#include "CustomStruct.hxx"

TEST(RNTupleIndex, Basic)
{
   FileRaii fileGuard("test_ntuple_index_basic.root");
   {
      auto model = RNTupleModel::Create();
      auto fld = model->MakeField<std::uint64_t>("fld");

      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath());

      for (int i = 0; i < 10; ++i) {
         *fld = i * 2;
         ntuple->Fill();
      }
   }

   auto pageSource = RPageSource::Create("ntuple", fileGuard.GetPath());
   auto index = ROOT::Experimental::Internal::RNTupleIndex::Create({"fld"}, *pageSource);

   auto ntuple = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   auto fld = ntuple->GetView<std::uint64_t>("fld");

   for (unsigned i = 0; i < ntuple->GetNEntries(); ++i) {
      auto fldValue = fld(i);
      EXPECT_EQ(fldValue, i * 2);
      EXPECT_EQ(index->GetFirstEntryNumber({&fldValue}), i);
   }
}

TEST(RNTupleIndex, SparseSecondary)
{
   FileRaii fileGuardMain("test_ntuple_index_sparse_secondary1.root");
   {
      auto model = RNTupleModel::Create();
      auto fldEvent = model->MakeField<std::uint64_t>("event");

      auto ntuple = RNTupleWriter::Recreate(std::move(model), "primary", fileGuardMain.GetPath());

      for (int i = 0; i < 10; ++i) {
         *fldEvent = i;
         ntuple->Fill();
      }
   }

   FileRaii fileGuardSecondary("test_ntuple_index_sparse_secondary2.root");
   {
      auto model = RNTupleModel::Create();
      auto fldEvent = model->MakeField<std::uint64_t>("event");
      auto fldX = model->MakeField<float>("x");

      auto ntuple = RNTupleWriter::Recreate(std::move(model), "secondary", fileGuardSecondary.GetPath());

      for (int i = 0; i < 5; ++i) {
         *fldEvent = i * 2;
         *fldX = static_cast<float>(i) / 3.14;
         ntuple->Fill();
      }
   }

   auto mainNtuple = RNTupleReader::Open("primary", fileGuardMain.GetPath());
   auto fldEvent = mainNtuple->GetView<std::uint64_t>("event");

   auto secondaryPageSource = RPageSource::Create("secondary", fileGuardSecondary.GetPath());
   auto index = ROOT::Experimental::Internal::RNTupleIndex::Create({"event"}, *secondaryPageSource);
   auto secondaryNTuple = RNTupleReader::Open("secondary", fileGuardSecondary.GetPath());
   auto fldX = secondaryNTuple->GetView<float>("x");

   for (unsigned i = 0; i < mainNtuple->GetNEntries(); ++i) {
      auto event = fldEvent(i);

      if (i % 2 == 1) {
         EXPECT_EQ(index->GetFirstEntryNumber<std::uint64_t>(event), ROOT::Experimental::kInvalidNTupleIndex)
            << "entry should not be present in the index";
      } else {
         auto idx = index->GetFirstEntryNumber<std::uint64_t>(event);
         EXPECT_EQ(idx, i / 2);
         EXPECT_FLOAT_EQ(fldX(idx), static_cast<float>(idx) / 3.14);
      }
   }
}

TEST(RNTupleIndex, MultipleFields)
{
   FileRaii fileGuard("test_ntuple_index_multiple_fields.root");
   {
      auto model = RNTupleModel::Create();
      auto fldRun = model->MakeField<std::uint64_t>("run");
      auto fldEvent = model->MakeField<std::uint64_t>("event");
      auto fldX = model->MakeField<float>("x");

      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath());

      for (int i = 0; i < 3; ++i) {
         *fldRun = i;
         for (int j = 0; j < 5; ++j) {
            *fldEvent = j;
            *fldX = static_cast<float>(i + j) / 3.14;
            ntuple->Fill();
         }
      }
   }

   auto pageSource = RPageSource::Create("ntuple", fileGuard.GetPath());
   pageSource->Attach();

   auto index = ROOT::Experimental::Internal::RNTupleIndex::Create({"run", "event"}, *pageSource);

   auto ntuple = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   auto fld = ntuple->GetView<float>("x");

   std::uint64_t event, run;
   for (std::uint64_t i = 0; i < pageSource->GetNEntries(); ++i) {
      run = i / 5;
      event = i % 5;
      auto entryIdx = index->GetFirstEntryNumber<std::uint64_t, std::uint64_t>(run, event);
      EXPECT_EQ(fld(entryIdx), fld(i));
   }

   auto idx1 = index->GetFirstEntryNumber<std::uint64_t, std::uint64_t>(2, 1);
   auto idx2 = index->GetFirstEntryNumber<std::uint64_t, std::uint64_t>(1, 2);
   EXPECT_NE(idx1, idx2);
}

TEST(RNTupleIndex, MultipleMatches)
{
   FileRaii fileGuard("test_ntuple_index_multiple_fields.root");
   {
      auto model = RNTupleModel::Create();
      auto fldRun = model->MakeField<std::uint64_t>("run");

      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath());

      *fldRun = 1;
      for (int i = 0; i < 10; ++i) {
         if (i > 4)
            *fldRun = 2;
         if (i > 7)
            *fldRun = 3;
         ntuple->Fill();
      }
   }

   auto pageSource = RPageSource::Create("ntuple", fileGuard.GetPath());
   pageSource->Attach();

   auto index = ROOT::Experimental::Internal::RNTupleIndex::Create({"run"}, *pageSource);

   auto entryIdxs = index->GetAllEntryNumbers<std::uint64_t>(1);
   auto expected = std::vector<std::uint64_t>{0, 1, 2, 3, 4};
   EXPECT_EQ(expected, *entryIdxs);
   entryIdxs = index->GetAllEntryNumbers<std::uint64_t>(2);
   expected = {5, 6, 7};
   EXPECT_EQ(expected, *entryIdxs);
   entryIdxs = index->GetAllEntryNumbers<std::uint64_t>(3);
   expected = {8, 9};
   EXPECT_EQ(expected, *entryIdxs);
   entryIdxs = index->GetAllEntryNumbers<std::uint64_t>(4);
   EXPECT_EQ(nullptr, entryIdxs);
}
