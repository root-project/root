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
   EXPECT_FALSE(joinTable->IsBuilt());

   try {
      joinTable->GetEntryIndexes({nullptr});
      FAIL() << "querying an unbuilt join table should not be possible";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("join table has not been built yet"));
   }

   joinTable->Build(*pageSource);
   EXPECT_TRUE(joinTable->IsBuilt());

   EXPECT_EQ(10UL, joinTable->GetSize());

   auto ntuple = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   auto fld = ntuple->GetView<std::uint64_t>("fld");

   for (unsigned i = 0; i < ntuple->GetNEntries(); ++i) {
      auto fldValue = fld(i);
      EXPECT_EQ(fldValue, i * 2);
      EXPECT_EQ(joinTable->GetEntryIndexes({&fldValue}), std::vector{static_cast<ROOT::NTupleSize_t>(i)});
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

   auto joinTable = RNTupleJoinTable::Create({"fldInt"});
   joinTable->Build(*pageSource);
   EXPECT_EQ(1UL, joinTable->GetSize());

   try {
      RNTupleJoinTable::Create({"fldFloat"})->Build(*pageSource);
      FAIL() << "non-integral-type field should not be allowed as join fields";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(
         err.what(),
         testing::HasSubstr(
            "cannot use field \"fldFloat\" with type \"float\" in join table: only integral types are allowed"));
   }

   try {
      RNTupleJoinTable::Create({"fldString"})->Build(*pageSource);
      FAIL() << "non-integral-type field should not be allowed as join fields";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(
         err.what(),
         testing::HasSubstr(
            "cannot use field \"fldString\" with type \"std::string\" in join table: only integral types are allowed"));
   }

   try {
      RNTupleJoinTable::Create({"fldStruct"})->Build(*pageSource);
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
   joinTable->Build(*secondaryPageSource);

   auto secondaryNTuple = RNTupleReader::Open("secondary", fileGuardSecondary.GetPath());
   auto fldX = secondaryNTuple->GetView<float>("x");

   for (unsigned i = 0; i < mainNtuple->GetNEntries(); ++i) {
      auto event = fldEvent(i);

      if (i % 2 == 1) {
         EXPECT_EQ(joinTable->GetEntryIndexes({&event}), std::vector<ROOT::NTupleSize_t>{})
            << "entry should not be present in the join table";
      } else {
         auto entryIdxs = joinTable->GetEntryIndexes({&event});
         ASSERT_EQ(1ul, entryIdxs.size());
         EXPECT_EQ(entryIdxs[0], i / 2);
         EXPECT_FLOAT_EQ(fldX(entryIdxs[0]), static_cast<float>(entryIdxs[0]) / 3.14);
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
   joinTable->Build(*pageSource);

   EXPECT_EQ(15ULL, joinTable->GetSize());

   auto ntuple = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   auto fld = ntuple->GetView<float>("x");

   std::int16_t run;
   std::uint64_t event;
   for (std::uint64_t i = 0; i < pageSource->GetNEntries(); ++i) {
      run = i / 5;
      event = i % 5;
      auto entryIdxs = joinTable->GetEntryIndexes({&run, &event});
      ASSERT_EQ(1ul, entryIdxs.size());
      EXPECT_EQ(fld(entryIdxs[0]), fld(i));
   }

   run = 1;
   event = 2;
   auto idx1 = joinTable->GetEntryIndexes({&run, &event});
   auto idx2 = joinTable->GetEntryIndexes({&event, &run});
   EXPECT_NE(idx1, idx2);

   try {
      joinTable->GetEntryIndexes({&run, &event, &event});
      FAIL() << "querying the join table with more values than join field values should not be possible";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("number of value pointers must match number of join fields"));
   }

   try {
      joinTable->GetEntryIndexes({&run});
      FAIL() << "querying the join table with fewer values than join field values should not be possible";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("number of value pointers must match number of join fields"));
   }
}

TEST(RNTupleJoinTable, MultipleMatches)
{
   FileRaii fileGuard("test_ntuple_join_table_multiple_matches.root");
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
   auto joinTable = RNTupleJoinTable::Create({"run"});
   joinTable->Build(*pageSource);

   EXPECT_EQ(3ULL, joinTable->GetSize());

   std::uint64_t run = 1;
   auto entryIdxs = joinTable->GetEntryIndexes({&run});
   auto expected = std::vector<ROOT::NTupleSize_t>{0, 1, 2, 3, 4};
   EXPECT_EQ(expected, entryIdxs);
   entryIdxs = joinTable->GetEntryIndexes({&(++run)});
   expected = {5, 6, 7};
   EXPECT_EQ(expected, entryIdxs);
   entryIdxs = joinTable->GetEntryIndexes({&(++run)});
   expected = {8, 9};
   EXPECT_EQ(expected, entryIdxs);
   entryIdxs = joinTable->GetEntryIndexes({&(++run)});
   EXPECT_EQ(std::vector<ROOT::NTupleSize_t>{}, entryIdxs);
}
