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
   EXPECT_EQ(std::vector<ROOT::NTupleSize_t>{}, joinTable->GetEntryIndexes({&fldValue}));

   // Now add the entry mapping for the page source
   joinTable->Add(*pageSource);

   auto ntuple = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   auto fld = ntuple->GetView<std::uint64_t>("fld");

   for (unsigned i = 0; i < ntuple->GetNEntries(); ++i) {
      fldValue = fld(i);
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
   joinTable->Add(*pageSource);

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
   joinTable->Add(*pageSource);

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

TEST(RNTupleJoinTable, Partitions)
{
   auto fnWriteNTuple = [](const FileRaii &fileGuard, std::uint16_t run) {
      auto model = RNTupleModel::Create();
      *model->MakeField<std::int16_t>("run") = run;
      auto fldI = model->MakeField<std::uint32_t>("i");

      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath());

      for (int i = 0; i < 5; ++i) {
         *fldI = i;
         ntuple->Fill();
      }
   };

   auto joinTable = RNTupleJoinTable::Create({"i"});

   std::vector<FileRaii> fileGuards;
   fileGuards.emplace_back("test_ntuple_join_partition1.root");
   fileGuards.emplace_back("test_ntuple_join_partition2.root");
   fileGuards.emplace_back("test_ntuple_join_partition3.root");
   fileGuards.emplace_back("test_ntuple_join_partition4.root");

   std::int16_t runNumbers[4] = {1, 2, 3, 3};
   std::vector<std::unique_ptr<RPageSource>> pageSources;

   // Create four ntuples where with their corresponding run numbers and add them to the join table using this run
   // number as the partition key.
   for (unsigned i = 0; i < fileGuards.size(); ++i) {
      fnWriteNTuple(fileGuards[i], runNumbers[i]);
      pageSources.emplace_back(RPageSource::Create("ntuple", fileGuards[i].GetPath()));
      joinTable->Add(*pageSources.back(), runNumbers[i]);
   }

   std::vector<RNTupleOpenSpec> openSpec;
   for (const auto &fileGuard : fileGuards) {
      openSpec.emplace_back("ntuple", fileGuard.GetPath());
   }
   auto proc = RNTupleProcessor::CreateChain(openSpec);

   auto i = proc->GetEntry().GetPtr<std::uint32_t>("i");
   auto run = proc->GetEntry().GetPtr<int16_t>("run");

   // When getting the entry indexes for all partitions, we expect multiple resulting entry indexes (i.e., one entry
   // index for each ntuple in the chain).
   *i = 0;
   std::unordered_map<RNTupleJoinTable::PartitionKey_t, std::vector<ROOT::NTupleSize_t>> expectedEntryIdxMap = {
      {1, {0}},
      {2, {0}},
      {3, {0, 0}},
   };
   EXPECT_EQ(expectedEntryIdxMap, joinTable->GetPartitionedEntryIndexes({i.get()}));
   EXPECT_EQ(expectedEntryIdxMap, joinTable->GetPartitionedEntryIndexes({i.get()}, {1, 2, 3}));

   expectedEntryIdxMap = {
      {1, {0}},
      {3, {0, 0}},
   };
   EXPECT_EQ(expectedEntryIdxMap, joinTable->GetPartitionedEntryIndexes({i.get()}, {1, 3}));

   // Calling GetEntryIndexes with a partition key not present in the join table shouldn't fail; it should just return
   // an empty vector.
   EXPECT_EQ(std::vector<ROOT::NTupleSize_t>{}, joinTable->GetEntryIndexes({i.get()}, 4));

   // Similarly, calling GetEntryIndexes with a partition key that is present in the join table but a join value that
   // isn't shouldn't fail; it should just return an empty vector.
   *i = 99;
   EXPECT_EQ(std::vector<ROOT::NTupleSize_t>{}, joinTable->GetEntryIndexes({i.get()}, 3));

   expectedEntryIdxMap.clear();
   EXPECT_EQ(expectedEntryIdxMap, joinTable->GetPartitionedEntryIndexes({i.get()}));
   EXPECT_EQ(expectedEntryIdxMap, joinTable->GetPartitionedEntryIndexes({i.get()}, {1, 2, 3}));
   EXPECT_EQ(expectedEntryIdxMap, joinTable->GetPartitionedEntryIndexes({i.get()}, {1, 3}));

   for (auto it = proc->begin(); it != proc->end(); it++) {
      auto entryIdxs = joinTable->GetEntryIndexes({i.get()}, *run);

      // Because two ntuples store their events under run number 3 and their entries for `i` are identical, two entry
      // indexes are expected. For the other case (run == 1 and run == 2), only one entry index is expected.
      if (*run == 3)
         EXPECT_EQ(entryIdxs.size(), 2ul);
      else
         EXPECT_EQ(entryIdxs.size(), 1ul);
   }
}
