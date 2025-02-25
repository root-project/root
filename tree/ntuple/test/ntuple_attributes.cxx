#include "ntuple_test.hxx"
#include <ROOT/RNTupleAttributes.hxx>

TEST(RNTupleAttributes, AttributeBasics)
{
   FileRaii fileGuard("test_ntuple_attrs_basics.root");
   
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
   
   fileGuard.PreserveFile();
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
   attrModel->AddProjectedField(std::move(projField), [] (const auto &) { return "foo"; });
   auto res = writer->CreateAttributeSet("MyAttrSet", attrModel->Clone());
   ASSERT_FALSE(res);
   EXPECT_THAT(res.GetError()->GetReport(), testing::HasSubstr("cannot contain projected fields"));
}
