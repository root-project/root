#include "ntuple_test.hxx"

TEST(RNTupleModel, Merge)
{
   auto model1 = RNTupleModel::Create();
   model1->MakeField<int>("x");

   auto model2 = RNTupleModel::Create();
   model2->MakeField<std::vector<float>>("y");

   try {
      ROOT::Experimental::Internal::MergeModels(*model1, *model2);
      FAIL() << "cannot merge unfrozen models";
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("invalid attempt to merge unfrozen models"));
   }

   model1->Freeze();
   model2->Freeze();

   auto mergedModel = ROOT::Experimental::Internal::MergeModels(*model1, *model2);

   EXPECT_EQ(mergedModel->GetField("x").GetFieldName(), "x");
   EXPECT_EQ(mergedModel->GetField("y").GetFieldName(), "y");
}

TEST(RNTupleModel, MergeWithPrefix)
{
   auto model1 = RNTupleModel::Create();
   model1->MakeField<int>("x");
   model1->Freeze();

   auto model2 = RNTupleModel::Create();
   model2->MakeField<int>("x");
   model2->Freeze();

   try {
      ROOT::Experimental::Internal::MergeModels(*model1, *model2);
      FAIL() << "cannot merge models with fields containing the same name without providing a prefix";
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("field name 'x' already exists in NTuple model"));
   }

   auto mergedModel = ROOT::Experimental::Internal::MergeModels(*model1, *model2, "n");

   EXPECT_EQ(mergedModel->GetField("x").GetFieldName(), "x");
   EXPECT_EQ(mergedModel->GetField("n:x").GetFieldName(), "n:x");
}
