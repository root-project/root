#include "ntuple_test.hxx"
#include <ROOT/RNTupleAttributes.hxx>

static std::size_t Count(ROOT::Experimental::RNTupleAttrEntryIterable iterable)
{
   std::size_t n = 0;
   for ([[maybe_unused]] auto _ : iterable)
      ++n;
   return n;
}

TEST(RNTupleAttributes, AttributeBasics)
{
   FileRaii fileGuard("test_ntuple_attrs_basics.root");

   // WRITING
   // ----------------------------------------------
   {
      // Create a RNTuple
      auto model = RNTupleModel::Create();
      auto pInt = model->MakeField<int>("int");
      auto file = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
      auto writer = RNTupleWriter::Append(std::move(model), "ntpl", *file);

      // Step 1: create model for the attribute set
      auto attrModel = RNTupleModel::Create();
      attrModel->SetDescription("My description");
      auto pMyAttr = attrModel->MakeField<std::string>("myAttr");

      // Step 2: create the attribute set from the writer
      auto attrSet = writer->CreateAttributeSet("MyAttrSet", std::move(attrModel));

      // Step 3: open attribute range. attrEntry has basically the same interface as REntry
      auto attrRange = attrSet->BeginRange();

      // Step 4: assign attribute values.
      // Values can be assigned anywhere between BeginRange() and CommitRange().
      *pMyAttr = "This is a custom attribute";
      for (int i = 0; i < 100; ++i) {
         *pInt = i;
         writer->Fill();
      }

      // Step 5: close attribute range
      attrSet->CommitRange(std::move(attrRange));
   }

   // READING
   // ----------------------------------------------
   {
      auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());

      // Read back the list of available attribute sets
      const auto attrSets = reader->GetDescriptor().GetAttributeSetNames();
      EXPECT_EQ(attrSets.size(), 1);
      for (const auto &name : attrSets) {
         EXPECT_EQ(name, "MyAttrSet");
      }

      // Fetch a specific attribute set
      auto attrSet = reader->OpenAttributeSet("MyAttrSet");
      for (int i = 0; i < 100; ++i) {
         int nAttrs = 0;
         for (const auto idx : attrSet->GetAttributes(i)) {
            attrSet->LoadAttrEntry(idx);
            auto pAttr = attrSet->GetModel().GetDefaultEntry().GetPtr<std::string>("myAttr");
            EXPECT_EQ(*pAttr, "This is a custom attribute");
            ++nAttrs;
         }
         EXPECT_EQ(nAttrs, 1);
      }
      EXPECT_EQ(attrSet->GetDescriptor().GetDescription(), "My description");
   }
}

TEST(RNTupleAttributes, AttributeBasicsExplicitEntry)
{
   FileRaii fileGuard("test_ntuple_attrs_basics_explentry.root");

   {
      auto model = RNTupleModel::Create();
      auto pInt = model->MakeField<int>("int");
      auto file = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
      auto writer = RNTupleWriter::Append(std::move(model), "ntpl", *file);

      auto attrModel = RNTupleModel::CreateBare();
      attrModel->SetDescription("My description");
      attrModel->MakeField<std::string>("myAttr");

      auto attrSet = writer->CreateAttributeSet("MyAttrSet", std::move(attrModel));
      auto attrEntry = attrSet->CreateEntry();
      auto attrRange = attrSet->BeginRange();
      auto pMyAttr = attrEntry->GetPtr<std::string>("myAttr");
      *pMyAttr = "This is a custom attribute";
      for (int i = 0; i < 100; ++i) {
         *pInt = i;
         writer->Fill();
      }
      attrSet->CommitRange(std::move(attrRange), *attrEntry);
   }
   {
      auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());

      const auto attrSets = reader->GetDescriptor().GetAttributeSetNames();
      EXPECT_EQ(attrSets.size(), 1);
      for (const auto &name : attrSets) {
         EXPECT_EQ(name, "MyAttrSet");
      }

      auto attrSet = reader->OpenAttributeSet("MyAttrSet");
      auto attrEntry = attrSet->CreateEntry();
      for (int i = 0; i < 100; ++i) {
         int nAttrs = 0;
         for (const auto idx : attrSet->GetAttributes(i)) {
            attrSet->LoadAttrEntry(idx, *attrEntry);
            auto pAttr = attrEntry->GetPtr<std::string>("myAttr");
            EXPECT_EQ(*pAttr, "This is a custom attribute");
            ++nAttrs;
         }
         EXPECT_EQ(nAttrs, 1);
      }
      EXPECT_EQ(attrSet->GetDescriptor().GetDescription(), "My description");
   }
}

TEST(RNTupleAttributes, AttributeDuplicateName)
{
   FileRaii fileGuard("test_ntuple_attrs_duplicate_name.root");

   // Create a RNTuple
   auto model = RNTupleModel::Create();
   auto pInt = model->MakeField<int>("int");
   auto file = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
   auto writer = RNTupleWriter::Append(std::move(model), "ntpl", *file);

   auto attrModel = RNTupleModel::Create();
   writer->CreateAttributeSet("MyAttrSet", attrModel->Clone());
   try {
      writer->CreateAttributeSet("MyAttrSet", std::move(attrModel));
      FAIL() << "Trying to create duplicate attribute sets should fail";
   } catch (const ROOT::RException &ex) {
      EXPECT_THAT(ex.what(), testing::HasSubstr("already exists"));
   }
}

TEST(RNTupleAttributes, AttributeInvalidModel)
{
   FileRaii fileGuard("test_ntuple_attrs_invalid_model.root");

   // Create a RNTuple
   auto model = RNTupleModel::Create();
   auto pInt = model->MakeField<int>("int");
   auto file = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
   auto writer = RNTupleWriter::Append(std::move(model), "ntpl", *file);

   auto attrModel = RNTupleModel::Create();
   attrModel->MakeField<int>("foo");
   auto projField = std::make_unique<ROOT::RField<int>>("proj");
   attrModel->AddProjectedField(std::move(projField), [](const auto &) { return "foo"; });
   try {
      writer->CreateAttributeSet("MyAttrSet", attrModel->Clone());
      FAIL() << "Trying to create an attribute model with projected fields should fail";
   } catch (const ROOT::RException &ex) {
      EXPECT_THAT(ex.what(), testing::HasSubstr("cannot contain projected fields"));
   }
}

TEST(RNTupleAttributes, ReservedAttributeSetName)
{
   FileRaii fileGuard("test_ntuple_attrs_reserved_name.root");

   // Create a RNTuple
   auto model = RNTupleModel::Create();
   auto pInt = model->MakeField<int>("int");
   auto file = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
   auto writer = RNTupleWriter::Append(std::move(model), "ntpl", *file);

   auto attrModel = RNTupleModel::Create();
   try {
      writer->CreateAttributeSet("ROOT", std::move(attrModel));
      FAIL() << "Trying to create an attribute set using a reserved name should fail";
   } catch (const ROOT::RException &ex) {
      EXPECT_THAT(ex.what(), testing::HasSubstr("reserved"));
   }
   try {
      writer->CreateAttributeSet("ROOT.MyAttrSet", std::move(attrModel));
      FAIL() << "Trying to create an attribute set using a reserved name should fail";
   } catch (const ROOT::RException &ex) {
      EXPECT_THAT(ex.what(), testing::HasSubstr("reserved"));
   }
   try {
      writer->CreateAttributeSet("ROOT.", std::move(attrModel));
      FAIL() << "Trying to create an attribute set using a reserved name should fail";
   } catch (const ROOT::RException &ex) {
      EXPECT_THAT(ex.what(), testing::HasSubstr("reserved"));
   }
}

TEST(RNTupleAttributes, InterleavingRanges)
{
   // Calling BeginRange multiple times without calling CommitRange in between is valid:
   // the user gets separate AttributeEntries that they can commit at their will.

   FileRaii fileGuard("test_ntuple_attrs_multiplebegin.root");

   {
      auto model = RNTupleModel::Create();
      auto pInt = model->MakeField<int>("int");
      auto file = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
      auto wopts = RNTupleWriteOptions();
      wopts.SetCompression(0);
      auto writer = RNTupleWriter::Append(std::move(model), "ntpl", *file, wopts);

      auto attrModel = RNTupleModel::Create();
      attrModel->MakeField<int>("attrInt");
      auto attrSet = writer->CreateAttributeSet("MyAttrSet", attrModel->Clone());

      auto attrEntry1 = attrSet->CreateEntry();
      auto attrRange1 = attrSet->BeginRange();
      auto attrEntry2 = attrSet->CreateEntry();
      auto attrRange2 = attrSet->BeginRange();
      int i1 = 0, i2 = 0;
      *attrEntry1->GetPtr<int>("attrInt") = i1++;
      *attrEntry2->GetPtr<int>("attrInt") = i2++;
      for (int i = 0; i < 30; ++i) {
         if (i > 0 && (i % 5) == 0) {
            attrSet->CommitRange(std::move(attrRange1), *attrEntry1);
            attrRange1 = attrSet->BeginRange();
            *attrEntry1->GetPtr<int>("attrInt") = i1++;
         }
         if (i > 0 && (i % 11) == 0) {
            attrSet->CommitRange(std::move(attrRange2), *attrEntry2);
            attrRange2 = attrSet->BeginRange();
            *attrEntry2->GetPtr<int>("attrInt") = i2++;
         }
         *pInt = i;
         writer->Fill();
      }
      attrSet->CommitRange(std::move(attrRange1), *attrEntry1);
      attrSet->CommitRange(std::move(attrRange2), *attrEntry2);
   }

   // read back
   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   auto attrSet = reader->OpenAttributeSet("MyAttrSet");
   auto attrEntry = attrSet->CreateEntry();
   for (auto i : reader->GetEntryRange()) {
      auto attrs = attrSet->GetAttributes(i);
      const auto nAttrs = Count(attrs);
      EXPECT_EQ(nAttrs, 2);
      int totVal = 0;
      for (auto idx : attrs) {
         attrSet->LoadAttrEntry(idx, *attrEntry);
         totVal += *attrEntry->GetPtr<int>("attrInt");
      }
      int expected = (i / 5) + (i / 11);
      EXPECT_EQ(totVal, expected);
   }
}

TEST(RNTupleAttributes, MultipleCommitRange)
{
   // Calling CommitRange multiple times on the same handle is an error (technically it cannot be on the "same handle"
   // since CommitRange requires you to move the handle - meaning the second time you are passing an invalid handle.)

   FileRaii fileGuard("test_ntuple_attrs_multipleend.root");

   auto model = RNTupleModel::Create();
   auto pInt = model->MakeField<int>("int");
   auto file = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
   auto writer = RNTupleWriter::Append(std::move(model), "ntpl", *file);

   auto attrModel = RNTupleModel::Create();
   attrModel->MakeField<std::string>("string");
   auto attrSet = writer->CreateAttributeSet("MyAttrSet", attrModel->Clone());

   auto &wModel = writer->GetModel();

   auto attrRange = attrSet->BeginRange();
   auto pMyAttr = attrSet->GetModel().GetDefaultEntry().GetPtr<std::string>("string");
   *pMyAttr = "Run 1";
   for (int i = 0; i < 100; ++i) {
      auto entry = wModel.CreateEntry();
      *pInt = i;
      writer->Fill(*entry);
   }
   attrSet->CommitRange(std::move(attrRange));
   EXPECT_THROW(attrSet->CommitRange(std::move(attrRange)), ROOT::RException);
}

TEST(RNTupleAttributes, AccessPastCommitRange)
{
   FileRaii fileGuard("test_ntuple_attrs_pastendrange.root");

   auto model = RNTupleModel::Create();
   auto pInt = model->MakeField<int>("int");
   auto file = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
   auto writer = RNTupleWriter::Append(std::move(model), "ntpl", *file);

   auto attrModel = RNTupleModel::Create();
   auto pMyAttr = attrModel->MakeField<std::string>("string");
   auto attrSet = writer->CreateAttributeSet("MyAttrSet", attrModel->Clone());

   auto &wModel = writer->GetModel();

   auto attrRange = attrSet->BeginRange();
   *pMyAttr = "Run 1";
   for (int i = 0; i < 100; ++i) {
      auto entry = wModel.CreateEntry();
      *pInt = i;
      writer->Fill(*entry);
   }
   attrSet->CommitRange(std::move(attrRange));
   // Cannot access attrRange after CommitRange()
   EXPECT_THROW(attrRange.Start(), ROOT::RException);
}

TEST(RNTupleAttributes, AssignMetadataAfterData)
{
   // Assigning the attribute range's value to the pointer can be done either before or after filling
   // the corresponding rows - provided it's done between a BeginRange() and an CommitRange().

   FileRaii fileGuard("test_ntuple_attrs_assign_after.root");

   {
      auto model = RNTupleModel::Create();
      auto pInt = model->MakeField<int>("int");
      auto file = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
      auto writer = RNTupleWriter::Append(std::move(model), "ntpl", *file);

      auto attrModel = RNTupleModel::Create();
      auto pAttrString = attrModel->MakeField<std::string>("string");
      auto pAttrInt = attrModel->MakeField<int>("int");
      auto attrSet = writer->CreateAttributeSet("MyAttrSet", std::move(attrModel));

      // First attribute entry
      {
         auto attrRange = attrSet->BeginRange();
         *pAttrString = "Run 1";
         *pAttrInt = 1;
         for (int i = 0; i < 10; ++i) {
            *pInt = i;
            writer->Fill();
         }
         attrSet->CommitRange(std::move(attrRange));
      }

      // Second attribute entry
      {
         auto attrRange = attrSet->BeginRange();
         for (int i = 0; i < 10; ++i) {
            *pInt = i;
            writer->Fill();
         }
         *pAttrString = "Run 2";
         *pAttrInt = 2;
         attrSet->CommitRange(std::move(attrRange));
      }
   }

   // Read back the attributes
   {
      auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
      // Fetch a specific attribute set
      auto attrSet = reader->OpenAttributeSet("MyAttrSet");
      auto nAttrs = 0;
      auto attrEntry = attrSet->CreateEntry();
      for (const auto idx : attrSet->GetAttributes()) {
         const auto range = attrSet->LoadAttrEntry(idx, *attrEntry);
         auto pAttrStr = attrEntry->GetPtr<std::string>("string");
         auto pAttrInt = attrEntry->GetPtr<int>("int");
         EXPECT_EQ(range.Start(), nAttrs * 10);
         EXPECT_EQ(range.End(), (1 + nAttrs) * 10);
         EXPECT_EQ(*pAttrStr, nAttrs == 0 ? "Run 1" : "Run 2");
         EXPECT_EQ(*pAttrInt, nAttrs == 0 ? 1 : 2);
         ++nAttrs;
      }
      EXPECT_EQ(nAttrs, 2);
   }
}

TEST(RNTupleAttributes, NoImplicitCommitRange)
{
   // CommitRange doesn't get called automatically when a AttributeRangeHandle goes out of scope:
   // forgetting to call CommitRange will cause the attribute range not to be saved.

   FileRaii fileGuard("test_ntuple_attrs_auto_end_range.root");

   {
      auto model = RNTupleModel::Create();
      auto pInt = model->MakeField<int>("int");
      auto file = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
      auto writer = RNTupleWriter::Append(std::move(model), "ntpl", *file);

      auto attrModel = RNTupleModel::Create();
      auto pMyAttr = attrModel->MakeField<std::string>("string");
      auto attrSet = writer->CreateAttributeSet("MyAttrSet", attrModel->Clone());

      [[maybe_unused]] auto attrRange = attrSet->BeginRange();
      *pMyAttr = "Run 1";
      for (int i = 0; i < 10; ++i) {
         *pInt = i;
         writer->Fill();
      }
      // Not calling CommitRange, so the attributes are not written.
   }

   // Read back the attributes
   {
      auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
      // Fetch a specific attribute set
      auto attrSet = reader->OpenAttributeSet("MyAttrSet");
      EXPECT_EQ(Count(attrSet->GetAttributes()), 0);
   }
}

TEST(RNTupleAttributes, AttributeMultipleSets)
{
   // Create multiple sets and interleave attribute ranges

   FileRaii fileGuard("test_ntuple_attrs_multiplesets.root");

   auto model = RNTupleModel::Create();
   auto pInt = model->MakeField<int>("int");
   auto file = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
   auto writer = RNTupleWriter::Append(std::move(model), "ntpl", *file);

   auto attrModel1 = RNTupleModel::Create();
   auto pInt1 = attrModel1->MakeField<int>("int");
   auto attrSet1 = writer->CreateAttributeSet("MyAttrSet1", attrModel1->Clone());

   auto attrModel2 = RNTupleModel::Create();
   auto pString2 = attrModel2->MakeField<std::string>("string");
   auto attrSet2 = writer->CreateAttributeSet("MyAttrSet2", attrModel2->Clone());

   auto attrRange2 = attrSet2->BeginRange();
   for (int i = 0; i < 100; ++i) {
      auto attrRange1 = attrSet1->BeginRange();
      *pInt1 = i;
      *pInt = i;
      writer->Fill();
      attrSet1->CommitRange(std::move(attrRange1));
   }
   *pString2 = "Run 1";
   attrSet2->CommitRange(std::move(attrRange2));
}

TEST(RNTupleAttributes, AttributeInvalidCommitRange)
{
   // Same as AttributeMultipleSets but try to pass the wrong range to a Set and verify it fails.

   FileRaii fileGuard("test_ntuple_attrs_invalid_endrange.root");

   // Create a RNTuple
   auto model = RNTupleModel::Create();
   auto pInt = model->MakeField<int>("int");
   auto file = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
   auto writer = RNTupleWriter::Append(std::move(model), "ntpl", *file);

   auto attrModel1 = RNTupleModel::Create();
   auto pInt1 = attrModel1->MakeField<int>("int");
   auto attrSet1 = writer->CreateAttributeSet("MyAttrSet1", attrModel1->Clone());

   auto attrModel2 = RNTupleModel::Create();
   auto pString2 = attrModel2->MakeField<std::string>("string");
   auto attrSet2 = writer->CreateAttributeSet("MyAttrSet2", attrModel2->Clone());

   [[maybe_unused]] auto attrRange2 = attrSet2->BeginRange();

   for (int i = 0; i < 100; ++i) {
      auto attrRange1 = attrSet1->BeginRange();
      *pInt1 = i;
      *pInt = i;
      writer->Fill();
      attrSet1->CommitRange(std::move(attrRange1));
   }
   *pString2 = "Run 1";

   // Oops! Calling CommitRange on the wrong set!
   EXPECT_THROW(attrSet1->CommitRange(std::move(attrRange2)), ROOT::RException);
}

TEST(RNTupleAttributes, ReadRanges)
{
   // Testing reading attribute ranges with a [startIdx, endIdx].

   FileRaii fileGuard("test_ntuple_attrs_readranges.root");

   {
      auto model = RNTupleModel::Create();
      auto pInt = model->MakeField<int>("int");
      auto file = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
      auto writer = RNTupleWriter::Append(std::move(model), "ntpl", *file);

      auto attrModelRuns = RNTupleModel::Create();
      auto pRun = attrModelRuns->MakeField<std::string>("run");
      auto attrSetRuns = writer->CreateAttributeSet("Attr_Runs", std::move(attrModelRuns));
      auto attrModelEpochs = RNTupleModel::Create();
      auto pEpoch = attrModelEpochs->MakeField<std::string>("epoch");
      auto attrSetEpochs = writer->CreateAttributeSet("Attr_Epochs", std::move(attrModelEpochs));

      {
         auto attrRangeEpoch = attrSetEpochs->BeginRange();
         *pEpoch = "Epoch 1";

         {
            auto attrRange = attrSetRuns->BeginRange();
            *pRun = "Run 1";
            for (int i = 0; i < 10; ++i) {
               *pInt = i;
               writer->Fill();
            }
            attrSetRuns->CommitRange(std::move(attrRange));
         }

         // Second attribute entry
         {
            auto attrRange = attrSetRuns->BeginRange();
            for (int i = 0; i < 10; ++i) {
               *pInt = i;
               writer->Fill();
            }
            *pRun = "Run 2";
            attrSetRuns->CommitRange(std::move(attrRange));
         }
         attrSetEpochs->CommitRange(std::move(attrRangeEpoch));
      }
      {
         auto attrRangeEpoch = attrSetEpochs->BeginRange();
         *pEpoch = "Epoch 2";

         {
            auto attrRange = attrSetRuns->BeginRange();
            *pRun = "Run 3";
            for (int i = 0; i < 10; ++i) {
               *pInt = i;
               writer->Fill();
            }
            attrSetRuns->CommitRange(std::move(attrRange));
         }
         attrSetEpochs->CommitRange(std::move(attrRangeEpoch));
      }
   }

   // Read back the attributes
   {
      auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
      // Fetch a specific attribute set
      auto attrSetRuns = reader->OpenAttributeSet("Attr_Runs");
      EXPECT_EQ(Count(attrSetRuns->GetAttributes()), 3);
      EXPECT_EQ(Count(attrSetRuns->GetAttributes(4)), 1);
      EXPECT_EQ(Count(attrSetRuns->GetAttributesContainingRange(4, 5)), 1);
      EXPECT_EQ(Count(attrSetRuns->GetAttributesContainingRange(0, 10)), 1);
      EXPECT_EQ(Count(attrSetRuns->GetAttributesContainingRange(0, 11)), 0);
      EXPECT_EQ(Count(attrSetRuns->GetAttributesContainingRange(0, 100)), 0);
      EXPECT_EQ(Count(attrSetRuns->GetAttributesInRange(4, 5)), 0);
      EXPECT_EQ(Count(attrSetRuns->GetAttributesInRange(0, 100)), 3);
      EXPECT_EQ(Count(attrSetRuns->GetAttributes(12)), 1);
      EXPECT_EQ(Count(attrSetRuns->GetAttributes(120)), 0);

      ROOT_EXPECT_WARNING_PARTIAL(EXPECT_EQ(Count(attrSetRuns->GetAttributesInRange(4, 3)), 0), "ROOT.NTuple",
                                  "empty range");
      ROOT_EXPECT_WARNING_PARTIAL(EXPECT_EQ(Count(attrSetRuns->GetAttributesContainingRange(4, 3)), 0), "ROOT.NTuple",
                                  "empty range");
      ROOT_EXPECT_WARNING_PARTIAL(EXPECT_EQ(Count(attrSetRuns->GetAttributesInRange(4, 4)), 0), "ROOT.NTuple",
                                  "empty range");
      ROOT_EXPECT_WARNING_PARTIAL(EXPECT_EQ(Count(attrSetRuns->GetAttributesContainingRange(4, 4)), 0), "ROOT.NTuple",
                                  "empty range");

      auto attrSetEpochs = reader->OpenAttributeSet("Attr_Epochs");
      EXPECT_EQ(Count(attrSetEpochs->GetAttributes()), 2);
   }
}

TEST(RNTupleAttributes, EmptyAttrRange)
{
   FileRaii fileGuard("test_ntuple_attrs_empty.root");

   {
      auto model = RNTupleModel::Create();
      auto pInt = model->MakeField<int>("int");
      auto file = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
      auto writer = RNTupleWriter::Append(std::move(model), "ntpl", *file);

      auto attrModel = RNTupleModel::Create();
      auto pAttr = attrModel->MakeField<std::string>("myAttr");
      auto attrSet = writer->CreateAttributeSet("MyAttrSet", std::move(attrModel));

      auto attrRange = attrSet->BeginRange();
      *pAttr = "This is range is empty.";
      // No values written to the main RNTuple...
      attrSet->CommitRange(std::move(attrRange));
   }

   {
      auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());

      // Read back the list of available attribute sets
      const auto attrSets = reader->GetDescriptor().GetAttributeSetNames();
      EXPECT_EQ(attrSets.size(), 1);
      for (const auto &name : attrSets) {
         EXPECT_EQ(name, "MyAttrSet");
      }

      // Fetch a specific attribute set
      auto attrSet = reader->OpenAttributeSet("MyAttrSet");
      EXPECT_EQ(Count(attrSet->GetAttributes()), 1);
      EXPECT_EQ(Count(attrSet->GetAttributes(0)), 0);
      ROOT_EXPECT_WARNING_PARTIAL(EXPECT_EQ(Count(attrSet->GetAttributesInRange(0, reader->GetNEntries())), 0),
                                  "ROOT.NTuple", "empty range");
      ROOT_EXPECT_WARNING_PARTIAL(EXPECT_EQ(Count(attrSet->GetAttributesContainingRange(0, reader->GetNEntries())), 0),
                                  "ROOT.NTuple", "empty range");
   }
}

TEST(RNTupleAttributes, AccessAttrSetReaderAfterClosingMainReader)
{
   FileRaii fileGuard("test_ntuple_attrs_readerafterclosing.root");

   {
      auto model = RNTupleModel::Create();
      auto pInt = model->MakeField<int>("int");
      auto file = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
      auto writer = RNTupleWriter::Append(std::move(model), "ntpl", *file);

      auto attrModel = RNTupleModel::Create();
      auto pAttr = attrModel->MakeField<std::string>("myAttr");
      auto attrSet = writer->CreateAttributeSet("MyAttrSet", std::move(attrModel));

      auto attrRange = attrSet->BeginRange();
      *pAttr = "This is an attribute";

      for (int i = 0; i < 10; ++i) {
         *pInt = i;
         writer->Fill();
      }
      attrSet->CommitRange(std::move(attrRange));
   }

   {
      auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());

      // Fetch an attribute set
      auto attrSet = reader->OpenAttributeSet("MyAttrSet");

      const auto nEntries = reader->GetNEntries();

      // Close the main reader
      reader.reset();

      // Access the attribute set reader
      EXPECT_EQ(Count(attrSet->GetAttributes()), 1);
      EXPECT_EQ(Count(attrSet->GetAttributes(0)), 1);
      EXPECT_EQ(Count(attrSet->GetAttributesInRange(0, nEntries)), 1);
      EXPECT_EQ(Count(attrSet->GetAttributesContainingRange(0, nEntries)), 1);
   }
}
