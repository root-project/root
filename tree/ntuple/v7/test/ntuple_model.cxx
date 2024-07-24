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

   EXPECT_EQ(mergedModel->GetField("x").GetQualifiedFieldName(), "x");
   EXPECT_EQ(mergedModel->GetField("y").GetQualifiedFieldName(), "y");
}

TEST(RNTupleModel, MergeWithPrefix)
{
   auto model1 = RNTupleModel::Create();
   model1->MakeField<int>("x");
   model1->Freeze();

   auto model2 = RNTupleModel::Create();
   model2->MakeField<int>("x");
   model2->MakeField<std::vector<float>>("y");
   model2->Freeze();

   try {
      ROOT::Experimental::Internal::MergeModels(*model1, *model2);
      FAIL() << "cannot merge models with fields containing the same name without providing a prefix";
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("field name 'x' already exists in NTuple model"));
   }

   auto mergedModel = ROOT::Experimental::Internal::MergeModels(*model1, *model2, "n");

   EXPECT_EQ(mergedModel->GetField("x").GetQualifiedFieldName(), "x");
   EXPECT_EQ(mergedModel->GetField("n.x").GetQualifiedFieldName(), "n.x");
}

TEST(RNTupleModel, EstimateWriteMemoryUsage)
{
   auto model = RNTupleModel::CreateBare();
   auto customStructVec = model->MakeField<std::vector<CustomStruct>>("CustomStructVec");

   static constexpr std::size_t NumColumns = 10;
   static constexpr std::size_t PageSize = 1234;
   static constexpr std::size_t ClusterSize = 6789;
   RNTupleWriteOptions options;
   options.SetApproxUnzippedPageSize(PageSize);
   options.SetApproxZippedClusterSize(ClusterSize);

   // Tail page optimization and buffered writing on, IMT not disabled.
   static constexpr std::size_t Expected1 = NumColumns * 3 * PageSize * 2 + 3 * ClusterSize;
   EXPECT_EQ(model->EstimateWriteMemoryUsage(options), Expected1);

   // Disable IMT.
   options.SetUseImplicitMT(RNTupleWriteOptions::EImplicitMT::kOff);
   static constexpr std::size_t Expected2 = NumColumns * 3 * PageSize * 2 + ClusterSize;
   EXPECT_EQ(model->EstimateWriteMemoryUsage(options), Expected2);

   // Disable buffered writing.
   options.SetUseBufferedWrite(false);
   static constexpr std::size_t Expected3 = NumColumns * 3 * PageSize;
   EXPECT_EQ(model->EstimateWriteMemoryUsage(options), Expected3);

   // Disable tail page optimization.
   options.SetUseTailPageOptimization(false);
   static constexpr std::size_t Expected4 = NumColumns * PageSize;
   EXPECT_EQ(model->EstimateWriteMemoryUsage(options), Expected4);

   // Enable buffered writing again.
   options.SetUseBufferedWrite(true);
   static constexpr std::size_t Expected5 = NumColumns * PageSize * 2 + ClusterSize;
   EXPECT_EQ(model->EstimateWriteMemoryUsage(options), Expected5);
}
