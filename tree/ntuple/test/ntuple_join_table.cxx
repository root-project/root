#include "ntuple_test.hxx"
#include "CustomStruct.hxx"

TEST(RNTupleJoinTable, Basic)
{
   FileRaii fileGuard("test_ntuple_join_table_basic.root");
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
   auto joinTable = RNTupleJoinTable::Create({"fld"});

   std::uint64_t fldValue = 0;

   // No entry mappings have been added to the join table yet
   EXPECT_EQ(ROOT::kInvalidNTupleIndex, joinTable->GetEntryIndex({&fldValue}));

   // Now add the entry mapping for the page source
   joinTable->Add(*pageSource);

   auto ntuple = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   auto fld = ntuple->GetView<std::uint64_t>("fld");

   for (unsigned i = 0; i < ntuple->GetNEntries(); ++i) {
      fldValue = fld(i);
      EXPECT_EQ(fldValue, i * 2);
      EXPECT_EQ(joinTable->GetEntryIndex({&fldValue}), i);
   }
}

TEST(RNTupleJoinTable, InvalidTypes)
{
   FileRaii fileGuard("test_ntuple_join_table_invalid_types.root");
   {
      auto model = RNTupleModel::Create();
      *model->MakeField<std::int8_t>("fldInt") = 99;
      *model->MakeField<float>("fldFloat") = 2.5;
      *model->MakeField<std::string>("fldString") = "foo";
      *model->MakeField<CustomStruct>("fldStruct") =
         CustomStruct{0.1, {2.f, 4.f}, {{1.f}, {3.f, 5.f}}, "bar", std::byte(8)};

      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath());
      ntuple->Fill();
   }

   auto pageSource = RPageSource::Create("ntuple", fileGuard.GetPath());

   try {
      RNTupleJoinTable::Create({"fldFloat"})->Add(*pageSource);
      FAIL() << "non-integral-type field should not be allowed as join fields";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(
         err.what(),
         testing::HasSubstr(
            "cannot use field \"fldFloat\" with type \"float\" in join table: only integral types are allowed"));
   }

   try {
      RNTupleJoinTable::Create({"fldString"})->Add(*pageSource);
      FAIL() << "non-integral-type field should not be allowed as join fields";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(
         err.what(),
         testing::HasSubstr(
            "cannot use field \"fldString\" with type \"std::string\" in join table: only integral types are allowed"));
   }

   try {
      RNTupleJoinTable::Create({"fldStruct"})->Add(*pageSource);
      FAIL() << "non-integral-type field should not be allowed as join fields";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("cannot use field \"fldStruct\" with type \"CustomStruct\" in "
                                                 "join table: only integral types are allowed"));
   }
}

TEST(RNTupleJoinTable, SparseSecondary)
{
   FileRaii fileGuardMain("test_ntuple_join_table_sparse_secondary1.root");
   {
      auto model = RNTupleModel::Create();
      auto fldEvent = model->MakeField<std::uint64_t>("event");

      auto ntuple = RNTupleWriter::Recreate(std::move(model), "primary", fileGuardMain.GetPath());

      for (int i = 0; i < 10; ++i) {
         *fldEvent = i;
         ntuple->Fill();
      }
   }

   FileRaii fileGuardSecondary("test_ntuple_join_table_sparse_secondary2.root");
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
   auto joinTable = RNTupleJoinTable::Create({"event"});
   joinTable->Add(*secondaryPageSource);

   auto secondaryNTuple = RNTupleReader::Open("secondary", fileGuardSecondary.GetPath());
   auto fldX = secondaryNTuple->GetView<float>("x");

   for (unsigned i = 0; i < mainNtuple->GetNEntries(); ++i) {
      auto event = fldEvent(i);

      if (i % 2 == 1) {
         EXPECT_EQ(joinTable->GetEntryIndex({&event}), ROOT::kInvalidNTupleIndex)
            << "entry should not be present in the join table";
      } else {
         auto entryIdx = joinTable->GetEntryIndex({&event});
         EXPECT_EQ(entryIdx, i / 2);
         EXPECT_FLOAT_EQ(fldX(entryIdx), static_cast<float>(entryIdx) / 3.14);
      }
   }
}

TEST(RNTupleJoinTable, MultipleFields)
{
   FileRaii fileGuard("test_ntuple_join_table_multiple_fields.root");
   {
      auto model = RNTupleModel::Create();
      auto fldRun = model->MakeField<std::int16_t>("run");
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
   auto joinTable = RNTupleJoinTable::Create({"run", "event"});
   joinTable->Add(*pageSource);

   auto ntuple = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   auto fld = ntuple->GetView<float>("x");

   std::int16_t run;
   std::uint64_t event;
   for (std::uint64_t i = 0; i < pageSource->GetNEntries(); ++i) {
      run = i / 5;
      event = i % 5;
      auto entryIdx = joinTable->GetEntryIndex({&run, &event});
      EXPECT_EQ(fld(entryIdx), fld(i));
   }

   run = 1;
   event = 2;
   auto idx1 = joinTable->GetEntryIndex({&run, &event});
   auto idx2 = joinTable->GetEntryIndex({&event, &run});
   EXPECT_NE(idx1, idx2);

   try {
      joinTable->GetEntryIndex({&run, &event, &event});
      FAIL() << "querying the join table with more values than join field values should not be possible";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("number of value pointers must match number of join fields"));
   }

   try {
      joinTable->GetEntryIndex({&run});
      FAIL() << "querying the join table with fewer values than join field values should not be possible";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("number of value pointers must match number of join fields"));
   }
}
