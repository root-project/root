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
      auto attrSet = writer->CreateAttributeSet("MyAttrSet", std::move(attrModel)).Unwrap();

      // Step 3: open attribute range. attrRange has basically the same interface as REntry
      auto attrRange = attrSet->BeginRange();

      auto &wModel = writer->GetModel();

      // Step 4: assign attribute values.
      // Values can be assigned anywhere between BeginRange() and EndRange().
      auto pMyAttr = attrRange.GetPtr<std::string>("myAttr");
      *pMyAttr = "This is a custom attribute";
      for (int i = 0; i < 100; ++i) {
         auto entry = wModel.CreateEntry();
         *pInt = i;
         writer->Fill(*entry);
      }

      // Step 5: close attribute range
      attrSet->EndRange(std::move(attrRange));
   }

   // READING
   // ----------------------------------------------
   {
      auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());

      // Read back the list of available attribute sets
      const auto attrSetNames = reader->GetDescriptor().GetAttributeSetNames();
      EXPECT_EQ(attrSetNames.size(), 1);
      for (const auto &name : attrSetNames) {
         EXPECT_EQ(name, "MyAttrSet");
      }

      // Fetch a specific attribute set
      auto attrSet = reader->GetAttributeSet("MyAttrSet").Unwrap();
      for (int i = 0; i < 100; ++i) {
         int nAttrs = 0;
         for (const auto &attrEntry : attrSet.GetAttributes(i)) {
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
   writer->CreateAttributeSet("MyAttrSet", attrModel->Clone()).Unwrap();
   auto res = writer->CreateAttributeSet("MyAttrSet", std::move(attrModel));
   ASSERT_FALSE(res);
   EXPECT_THAT(res.GetError()->GetReport(), testing::HasSubstr("already exists"));
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
   auto res = writer->CreateAttributeSet("MyAttrSet", attrModel->Clone());
   ASSERT_FALSE(res);
   EXPECT_THAT(res.GetError()->GetReport(), testing::HasSubstr("cannot contain projected fields"));
}

TEST(RNTupleAttributes, MultipleBeginRange)
{
   // Calling BeginRange multiple times without calling EndRange is an error.

   FileRaii fileGuard("test_ntuple_attrs_multiplebegin.root");

   auto model = RNTupleModel::Create();
   auto pInt = model->MakeField<int>("int");
   auto file = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
   auto writer = RNTupleWriter::Append(std::move(model), "ntpl", *file);

   auto attrModel = RNTupleModel::Create();
   attrModel->MakeField<std::string>("string");
   auto attrSet = writer->CreateAttributeSet("MyAttrSet", attrModel->Clone()).Unwrap();

   attrSet->BeginRange();
   EXPECT_THROW(attrSet->BeginRange(), ROOT::RException);
}

TEST(RNTupleAttributes, MultipleEndRange)
{
   // Calling EndRange multiple times on the same handle is an error (technically it cannot be on the "same handle"
   // since EndRange requires you to move the handle - meaning the second time you are passing an invalid handle.)

   FileRaii fileGuard("test_ntuple_attrs_multipleend.root");

   auto model = RNTupleModel::Create();
   auto pInt = model->MakeField<int>("int");
   auto file = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
   auto writer = RNTupleWriter::Append(std::move(model), "ntpl", *file);

   auto attrModel = RNTupleModel::Create();
   attrModel->MakeField<std::string>("string");
   auto attrSet = writer->CreateAttributeSet("MyAttrSet", attrModel->Clone()).Unwrap();

   auto &wModel = writer->GetModel();

   auto attrRange = attrSet->BeginRange();
   auto pMyAttr = attrRange.GetPtr<std::string>("string");
   *pMyAttr = "Run 1";
   for (int i = 0; i < 100; ++i) {
      auto entry = wModel.CreateEntry();
      *pInt = i;
      writer->Fill(*entry);
   }
   attrSet->EndRange(std::move(attrRange));
   EXPECT_THROW(attrSet->EndRange(std::move(attrRange)), ROOT::RException);
}

TEST(RNTupleAttributes, AccessPastEndRange)
{
   FileRaii fileGuard("test_ntuple_attrs_pastendrange.root");

   auto model = RNTupleModel::Create();
   auto pInt = model->MakeField<int>("int");
   auto file = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
   auto writer = RNTupleWriter::Append(std::move(model), "ntpl", *file);

   auto attrModel = RNTupleModel::Create();
   attrModel->MakeField<std::string>("string");
   auto attrSet = writer->CreateAttributeSet("MyAttrSet", attrModel->Clone()).Unwrap();

   auto &wModel = writer->GetModel();

   auto attrRange = attrSet->BeginRange();
   auto pMyAttr = attrRange.GetPtr<std::string>("string");
   *pMyAttr = "Run 1";
   for (int i = 0; i < 100; ++i) {
      auto entry = wModel.CreateEntry();
      *pInt = i;
      writer->Fill(*entry);
   }
   attrSet->EndRange(std::move(attrRange));
   // Cannot access attrRange after EndRange()
   EXPECT_THROW(attrRange.GetPtr<std::string>("string"), ROOT::RException);
}

TEST(RNTupleAttributes, AssignMetadataAfterData)
{
   // Assigning the attribute range's value to the pointer can be done either before or after filling
   // the corresponding rows - provided it's done between a BeginRange() and an EndRange().

   FileRaii fileGuard("test_ntuple_attrs_assign_after.root");

   {
      auto model = RNTupleModel::Create();
      auto pInt = model->MakeField<int>("int");
      auto file = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
      auto writer = RNTupleWriter::Append(std::move(model), "ntpl", *file);

      auto attrModel = RNTupleModel::Create();
      attrModel->MakeField<std::string>("string");
      auto attrSet = writer->CreateAttributeSet("MyAttrSet", attrModel->Clone()).Unwrap();

      // First attribute entry
      {
         auto attrRange = attrSet->BeginRange();
         auto pMyAttr = attrRange.GetPtr<std::string>("string");
         *pMyAttr = "Run 1";
         for (int i = 0; i < 10; ++i) {
            *pInt = i;
            writer->Fill();
         }
         attrSet->EndRange(std::move(attrRange));
      }

      // Second attribute entry
      {
         auto attrRange = attrSet->BeginRange();
         auto pMyAttr = attrRange.GetPtr<std::string>("string");
         for (int i = 0; i < 10; ++i) {
            *pInt = i;
            writer->Fill();
         }
         *pMyAttr = "Run 2";
         attrSet->EndRange(std::move(attrRange));
      }
   }

   // Read back the attributes
   {
      auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
      // Fetch a specific attribute set
      auto attrSet = reader->GetAttributeSet("MyAttrSet").Unwrap();
      auto nAttrs = 0;
      for (const auto &attrEntry : attrSet.GetAttributes()) {
         auto pAttr = attrEntry.GetPtr<std::string>("string");
         EXPECT_EQ(attrEntry.GetRange().first, nAttrs * 10);
         EXPECT_EQ(attrEntry.GetRange().second, (1 + nAttrs) * 10);
         EXPECT_EQ(*pAttr, nAttrs == 0 ? "Run 1" : "Run 2");
         ++nAttrs;
      }
      EXPECT_EQ(nAttrs, 2);
   }
}

TEST(RNTupleAttributes, ImplicitEndRange)
{
   // EndRange gets called automatically when a AttributeRangeHandle goes out of scope.

   FileRaii fileGuard("test_ntuple_attrs_auto_end_range.root");

   {
      auto model = RNTupleModel::Create();
      auto pInt = model->MakeField<int>("int");
      auto file = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
      auto writer = RNTupleWriter::Append(std::move(model), "ntpl", *file);

      auto attrModel = RNTupleModel::Create();
      attrModel->MakeField<std::string>("string");
      auto attrSet = writer->CreateAttributeSet("MyAttrSet", attrModel->Clone()).Unwrap();

      auto attrRange = attrSet->BeginRange();
      auto pMyAttr = attrRange.GetPtr<std::string>("string");
      *pMyAttr = "Run 1";
      for (int i = 0; i < 10; ++i) {
         *pInt = i;
         writer->Fill();
      }
      // Calling EndRange implicitly on scope exit
   }

   // Read back the attributes
   {
      auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
      // Fetch a specific attribute set
      auto attrSet = reader->GetAttributeSet("MyAttrSet").Unwrap();
      auto nAttrs = 0;
      for (const auto &attrEntry : attrSet.GetAttributes()) {
         auto pAttr = attrEntry.GetPtr<std::string>("string");
         EXPECT_EQ(attrEntry.GetRange().first, 0);
         EXPECT_EQ(attrEntry.GetRange().second, 10);
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
   auto attrSet1 = writer->CreateAttributeSet("MyAttrSet1", attrModel1->Clone()).Unwrap();
   auto attrModel2 = RNTupleModel::Create();
   attrModel2->MakeField<std::string>("string");
   auto attrSet2 = writer->CreateAttributeSet("MyAttrSet2", attrModel2->Clone()).Unwrap();

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
      attrSet1->EndRange(std::move(attrRange1));
   }
   *pMyAttr2 = "Run 1";
   attrSet2->EndRange(std::move(attrRange2));
}

TEST(RNTupleAttributes, AttributeInvalidEndRange)
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
   auto attrSet1 = writer->CreateAttributeSet("MyAttrSet1", attrModel1->Clone()).Unwrap();
   auto attrModel2 = RNTupleModel::Create();
   attrModel2->MakeField<std::string>("string");
   auto attrSet2 = writer->CreateAttributeSet("MyAttrSet2", attrModel2->Clone()).Unwrap();

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
      attrSet1->EndRange(std::move(attrRange1));
   }
   *pMyAttr2 = "Run 1";

   // Oops! Calling EndRange on the wrong set!
   EXPECT_THROW(attrSet1->EndRange(std::move(attrRange2)), ROOT::RException);
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
      auto attrSetRuns = writer->CreateAttributeSet("Attr_Runs", std::move(attrModelRuns)).Unwrap();
      auto attrModelEpochs = RNTupleModel::Create();
      attrModelEpochs->MakeField<std::string>("epoch");
      auto attrSetEpochs = writer->CreateAttributeSet("Attr_Epochs", std::move(attrModelEpochs)).Unwrap();

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
            attrSetRuns->EndRange(std::move(attrRange));
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
            attrSetRuns->EndRange(std::move(attrRange));
         }
         attrSetEpochs->EndRange(std::move(attrRangeEpoch));
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
            attrSetRuns->EndRange(std::move(attrRange));
         }
         attrSetEpochs->EndRange(std::move(attrRangeEpoch));
      }
   }

   // Read back the attributes
   {
      auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
      // Fetch a specific attribute set
      auto attrSetRuns = reader->GetAttributeSet("Attr_Runs").Unwrap();
      EXPECT_EQ(attrSetRuns.GetAttributes().size(), 3);
      EXPECT_EQ(attrSetRuns.GetAttributes(4).size(), 1);
      EXPECT_EQ(attrSetRuns.GetAttributesContainingRange(4, 5).size(), 1);
      EXPECT_EQ(attrSetRuns.GetAttributesContainingRange(4, 4).size(), 1);
      EXPECT_EQ(attrSetRuns.GetAttributesContainingRange(0, 100).size(), 0);
      EXPECT_EQ(attrSetRuns.GetAttributesInRange(4, 5).size(), 0);
      EXPECT_EQ(attrSetRuns.GetAttributesInRange(4, 4).size(), 0);
      EXPECT_EQ(attrSetRuns.GetAttributesInRange(0, 100).size(), 3);
      EXPECT_EQ(attrSetRuns.GetAttributes(12).size(), 1);
      EXPECT_EQ(attrSetRuns.GetAttributes(120).size(), 0);

      {
         ROOT::TestSupport::CheckDiagsRAII diags;
         diags.requiredDiag(kWarning, "ROOT.NTuple", "end < start", false);
         EXPECT_EQ(attrSetRuns.GetAttributesInRange(4, 3).size(), 0);
         EXPECT_EQ(attrSetRuns.GetAttributesContainingRange(4, 3).size(), 0);
      }

      auto attrSetEpochs = reader->GetAttributeSet("Attr_Epochs").Unwrap();
      EXPECT_EQ(attrSetEpochs.GetAttributes().size(), 2);
   }
}
