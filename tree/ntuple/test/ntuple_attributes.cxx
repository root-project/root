#include "ntuple_test.hxx"
#include <ROOT/RNTupleAttributes.hxx>

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
      auto pAttr = attrModel->MakeField<std::string>("myAttr");

      // Step 2: create the attribute set from the writer
      auto attrSet = writer->CreateAttributeSet("MyAttrSet", std::move(attrModel));

      // Step 3: open attribute range. attrRange has basically the same interface as REntry
      auto attrRange = attrSet->BeginRange();

      auto &wModel = writer->GetModel();

      // Step 4: assign attribute values.
      // Values can be assigned anywhere between BeginRange() and CommitRange().
      auto pMyAttr = attrRange.GetPtr<std::string>("myAttr");
      *pMyAttr = "This is a custom attribute";
      for (int i = 0; i < 100; ++i) {
         auto entry = wModel.CreateEntry();
         *pInt = i;
         writer->Fill(*entry);
      }

      // Step 5: close attribute range
      attrSet->CommitRange(std::move(attrRange));
   }

   // READING
   // ----------------------------------------------
   {
      auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());

      // Read back the list of available attribute sets
      const auto attrSets = reader->GetDescriptor().GetAttributeSets();
      EXPECT_EQ(attrSets.size(), 1);
      for (const auto &[name, _] : attrSets) {
         EXPECT_EQ(name, "MyAttrSet");
      }

      // Fetch a specific attribute set
      auto attrSet = reader->OpenAttributeSet("MyAttrSet");
      for (int i = 0; i < 100; ++i) {
         int nAttrs = 0;
         for (const auto &attrEntry : attrSet->GetAttributes(i)) {
            auto pAttr = attrEntry.GetPtr<std::string>("myAttr");
            EXPECT_EQ(*pAttr, "This is a custom attribute");
            ++nAttrs;
         }
         EXPECT_EQ(nAttrs, 1);
      }
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

TEST(RNTupleAttributes, MultipleBeginRange)
{
   // Calling BeginRange multiple times without calling CommitRange is an error.

   FileRaii fileGuard("test_ntuple_attrs_multiplebegin.root");

   auto model = RNTupleModel::Create();
   auto pInt = model->MakeField<int>("int");
   auto file = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
   auto writer = RNTupleWriter::Append(std::move(model), "ntpl", *file);

   auto attrModel = RNTupleModel::Create();
   attrModel->MakeField<std::string>("string");
   auto attrSet = writer->CreateAttributeSet("MyAttrSet", attrModel->Clone());

   attrSet->BeginRange();
   EXPECT_THROW(attrSet->BeginRange(), ROOT::RException);
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
   auto pMyAttr = attrRange.GetPtr<std::string>("string");
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
   attrModel->MakeField<std::string>("string");
   auto attrSet = writer->CreateAttributeSet("MyAttrSet", attrModel->Clone());

   auto &wModel = writer->GetModel();

   auto attrRange = attrSet->BeginRange();
   auto pMyAttr = attrRange.GetPtr<std::string>("string");
   *pMyAttr = "Run 1";
   for (int i = 0; i < 100; ++i) {
      auto entry = wModel.CreateEntry();
      *pInt = i;
      writer->Fill(*entry);
   }
   attrSet->CommitRange(std::move(attrRange));
   // Cannot access attrRange after CommitRange()
   EXPECT_THROW(attrRange.GetPtr<std::string>("string"), ROOT::RException);
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
      attrModel->MakeField<std::string>("string");
      attrModel->MakeField<int>("int");
      auto attrSet = writer->CreateAttributeSet("MyAttrSet", attrModel->Clone());

      // First attribute entry
      {
         auto attrRange = attrSet->BeginRange();
         *attrRange.GetPtr<std::string>("string") = "Run 1";
         *attrRange.GetPtr<int>("int") = 1;
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
         *attrRange.GetPtr<std::string>("string") = "Run 2";
         *attrRange.GetPtr<int>("int") = 2;
         attrSet->CommitRange(std::move(attrRange));
      }
   }

   // Read back the attributes
   {
      auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
      // Fetch a specific attribute set
      auto attrSet = reader->OpenAttributeSet("MyAttrSet");
      auto nAttrs = 0;
      for (const auto &attrEntry : attrSet->GetAttributes()) {
         auto pAttrStr = attrEntry.GetPtr<std::string>("string");
         auto pAttrInt = attrEntry.GetPtr<int>("int");
         EXPECT_EQ(attrEntry.GetRange().Start(), nAttrs * 10);
         EXPECT_EQ(attrEntry.GetRange().End(), (1 + nAttrs) * 10);
         EXPECT_EQ(*pAttrStr, nAttrs == 0 ? "Run 1" : "Run 2");
         EXPECT_EQ(*pAttrInt, nAttrs == 0 ? 1 : 2);
         ++nAttrs;
      }
      EXPECT_EQ(nAttrs, 2);
   }
}

TEST(RNTupleAttributes, ImplicitCommitRange)
{
   // CommitRange gets called automatically when a AttributeRangeHandle goes out of scope.

   FileRaii fileGuard("test_ntuple_attrs_auto_end_range.root");

   {
      auto model = RNTupleModel::Create();
      auto pInt = model->MakeField<int>("int");
      auto file = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
      auto writer = RNTupleWriter::Append(std::move(model), "ntpl", *file);

      auto attrModel = RNTupleModel::Create();
      attrModel->MakeField<std::string>("string");
      auto attrSet = writer->CreateAttributeSet("MyAttrSet", attrModel->Clone());

      auto attrRange = attrSet->BeginRange();
      auto pMyAttr = attrRange.GetPtr<std::string>("string");
      *pMyAttr = "Run 1";
      for (int i = 0; i < 10; ++i) {
         *pInt = i;
         writer->Fill();
      }
      // Calling CommitRange implicitly on scope exit
   }

   // Read back the attributes
   {
      auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
      // Fetch a specific attribute set
      auto attrSet = reader->OpenAttributeSet("MyAttrSet");
      auto nAttrs = 0;
      for (const auto &attrEntry : attrSet->GetAttributes()) {
         auto pAttr = attrEntry.GetPtr<std::string>("string");
         EXPECT_EQ(attrEntry.GetRange().Start(), 0);
         EXPECT_EQ(attrEntry.GetRange().End(), 10);
         EXPECT_EQ(*pAttr, "Run 1");
         ++nAttrs;
      }
      EXPECT_EQ(nAttrs, 1);
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
   attrModel1->MakeField<int>("int");
   auto attrSet1 = writer->CreateAttributeSet("MyAttrSet1", attrModel1->Clone());
   auto attrModel2 = RNTupleModel::Create();
   attrModel2->MakeField<std::string>("string");
   auto attrSet2 = writer->CreateAttributeSet("MyAttrSet2", attrModel2->Clone());

   auto &wModel = writer->GetModel();

   auto attrRange2 = attrSet2->BeginRange();
   auto pMyAttr2 = attrRange2.GetPtr<std::string>("string");
   for (int i = 0; i < 100; ++i) {
      auto attrRange1 = attrSet1->BeginRange();
      auto pMyAttr1 = attrRange1.GetPtr<int>("int");
      *pMyAttr1 = i;
      auto entry = wModel.CreateEntry();
      *pInt = i;
      writer->Fill(*entry);
      attrSet1->CommitRange(std::move(attrRange1));
   }
   *pMyAttr2 = "Run 1";
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
   attrModel1->MakeField<int>("int");
   auto attrSet1 = writer->CreateAttributeSet("MyAttrSet1", attrModel1->Clone());
   auto attrModel2 = RNTupleModel::Create();
   attrModel2->MakeField<std::string>("string");
   auto attrSet2 = writer->CreateAttributeSet("MyAttrSet2", attrModel2->Clone());

   auto attrRange2 = attrSet2->BeginRange();

   auto &wModel = writer->GetModel();

   auto pMyAttr2 = attrRange2.GetPtr<std::string>("string");
   for (int i = 0; i < 100; ++i) {
      auto attrRange1 = attrSet1->BeginRange();
      auto pMyAttr1 = attrRange1.GetPtr<int>("int");
      *pMyAttr1 = i;
      auto entry = wModel.CreateEntry();
      *pInt = i;
      writer->Fill(*entry);
      attrSet1->CommitRange(std::move(attrRange1));
   }
   *pMyAttr2 = "Run 1";

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
      attrModelRuns->MakeField<std::string>("run");
      auto attrSetRuns = writer->CreateAttributeSet("Attr_Runs", std::move(attrModelRuns));
      auto attrModelEpochs = RNTupleModel::Create();
      attrModelEpochs->MakeField<std::string>("epoch");
      auto attrSetEpochs = writer->CreateAttributeSet("Attr_Epochs", std::move(attrModelEpochs));

      {
         auto attrRangeEpoch = attrSetEpochs->BeginRange();
         auto pEpoch = attrRangeEpoch.GetPtr<std::string>("epoch");
         *pEpoch = "Epoch 1";

         {
            auto attrRange = attrSetRuns->BeginRange();
            auto pMyAttr = attrRange.GetPtr<std::string>("run");
            *pMyAttr = "Run 1";
            for (int i = 0; i < 10; ++i) {
               *pInt = i;
               writer->Fill();
            }
            attrSetRuns->CommitRange(std::move(attrRange));
         }

         // Second attribute entry
         {
            auto attrRange = attrSetRuns->BeginRange();
            auto pMyAttr = attrRange.GetPtr<std::string>("run");
            for (int i = 0; i < 10; ++i) {
               *pInt = i;
               writer->Fill();
            }
            *pMyAttr = "Run 2";
            attrSetRuns->CommitRange(std::move(attrRange));
         }
         attrSetEpochs->CommitRange(std::move(attrRangeEpoch));
      }
      {
         auto attrRangeEpoch = attrSetEpochs->BeginRange();
         auto pMyAttrEpoch = attrRangeEpoch.GetPtr<std::string>("epoch");
         *pMyAttrEpoch = "Epoch 2";

         {
            auto attrRange = attrSetRuns->BeginRange();
            auto pMyAttr = attrRange.GetPtr<std::string>("run");
            *pMyAttr = "Run 3";
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
      EXPECT_EQ(attrSetRuns->GetAttributes().size(), 3);
      EXPECT_EQ(attrSetRuns->GetAttributes(4).size(), 1);
      EXPECT_EQ(attrSetRuns->GetAttributesContainingRange(4, 5).size(), 1);
      EXPECT_EQ(attrSetRuns->GetAttributesContainingRange(4, 4).size(), 1);
      EXPECT_EQ(attrSetRuns->GetAttributesContainingRange(0, 10).size(), 1);
      EXPECT_EQ(attrSetRuns->GetAttributesContainingRange(0, 11).size(), 0);
      EXPECT_EQ(attrSetRuns->GetAttributesContainingRange(0, 100).size(), 0);
      EXPECT_EQ(attrSetRuns->GetAttributesInRange(4, 5).size(), 0);
      EXPECT_EQ(attrSetRuns->GetAttributesInRange(4, 4).size(), 0);
      EXPECT_EQ(attrSetRuns->GetAttributesInRange(0, 100).size(), 3);
      EXPECT_EQ(attrSetRuns->GetAttributes(12).size(), 1);
      EXPECT_EQ(attrSetRuns->GetAttributes(120).size(), 0);

      {
         ROOT::TestSupport::CheckDiagsRAII diags;
         diags.requiredDiag(kWarning, "ROOT.NTuple", "end < start", false);
         EXPECT_EQ(attrSetRuns->GetAttributesInRange(4, 3).size(), 0);
         EXPECT_EQ(attrSetRuns->GetAttributesContainingRange(4, 3).size(), 0);
      }

      auto attrSetEpochs = reader->OpenAttributeSet("Attr_Epochs");
      EXPECT_EQ(attrSetEpochs->GetAttributes().size(), 2);
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

      auto pMyAttr = attrRange.GetPtr<std::string>("myAttr");
      *pMyAttr = "This is range is empty.";
      // No values written to the main RNTuple...
      attrSet->CommitRange(std::move(attrRange));
   }

   {
      auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());

      // Read back the list of available attribute sets
      const auto attrSets = reader->GetDescriptor().GetAttributeSets();
      EXPECT_EQ(attrSets.size(), 1);
      for (const auto &[name, _] : attrSets) {
         EXPECT_EQ(name, "MyAttrSet");
      }

      // Fetch a specific attribute set
      auto attrSet = reader->OpenAttributeSet("MyAttrSet");
      EXPECT_EQ(attrSet->GetAttributes().size(), 1);
      EXPECT_EQ(attrSet->GetAttributes(0).size(), 0);
      EXPECT_EQ(attrSet->GetAttributesInRange(0, reader->GetNEntries()).size(), 0);
      EXPECT_EQ(attrSet->GetAttributesContainingRange(0, reader->GetNEntries()).size(), 0);
   }
}
