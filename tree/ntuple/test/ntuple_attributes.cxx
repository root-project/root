#include "ntuple_test.hxx"
#include <ROOT/RNTupleAttributes.hxx>

TEST(RNTupleAttributes, AttributeBasics)
{
   FileRaii fileGuard("test_ntuple_attrs_basics.root");
   fileGuard.PreserveFile();

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
      auto *attrSet = writer->CreateAttributeSet("MyAttrSet", std::move(attrModel)).Unwrap();

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
      // const auto *attrSet = reader->GetAttributeSet("MyAttrSet");
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

   // TODO: read back RNTuple and check if Attributes are correct.
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
