#include "ntuple_test.hxx"

TEST(RNTupleModel, EstimateWriteMemoryUsage)
{
   auto model = RNTupleModel::CreateBare();
   auto customStructVec = model->MakeField<std::vector<CustomStruct>>("CustomStructVec");

   static constexpr std::size_t NumColumns = 10;
   static constexpr std::size_t ColumnElementsSize = 8 + 4 + 8 + 4 + 8 + 8 + 4 + 8 + 1 + 1;
   static constexpr std::size_t InitialNElementsPerPage = 1;
   static constexpr std::size_t MaxPageSize = 100;
   static constexpr std::size_t ClusterSize = 6789;
   RNTupleWriteOptions options;
   options.SetInitialNElementsPerPage(InitialNElementsPerPage);
   options.SetMaxUnzippedPageSize(MaxPageSize);
   options.SetApproxZippedClusterSize(ClusterSize);

   // Tail page optimization and buffered writing on, IMT not disabled.
   static constexpr std::size_t Expected1 = NumColumns * MaxPageSize + ColumnElementsSize + 3 * ClusterSize;
   EXPECT_EQ(model->EstimateWriteMemoryUsage(options), Expected1);

   static constexpr std::size_t PageBufferBudget = 800;
   options.SetPageBufferBudget(PageBufferBudget);
   static constexpr std::size_t Expected2 = PageBufferBudget + ColumnElementsSize + 3 * ClusterSize;
   EXPECT_EQ(model->EstimateWriteMemoryUsage(options), Expected2);

   // Disable IMT.
   options.SetUseImplicitMT(RNTupleWriteOptions::EImplicitMT::kOff);
   static constexpr std::size_t Expected3 = PageBufferBudget + ColumnElementsSize + ClusterSize;
   EXPECT_EQ(model->EstimateWriteMemoryUsage(options), Expected3);

   // Disable buffered writing.
   options.SetUseBufferedWrite(false);
   static constexpr std::size_t Expected4 = PageBufferBudget;
   EXPECT_EQ(model->EstimateWriteMemoryUsage(options), Expected4);
}
