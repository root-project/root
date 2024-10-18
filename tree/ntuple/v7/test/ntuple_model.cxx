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

TEST(RNTupleModel, Clone)
{
   auto model = RNTupleModel::Create();
   model->MakeField<float>("f");
   model->MakeField<std::vector<float>>("vec");
   model->MakeField<CustomStruct>("struct");
   model->MakeField<TObject>("obj");
   model->Freeze();

   for (auto &f : model->GetFieldZero()) {
      if (f.GetTypeName() == "float") {
         f.SetColumnRepresentatives({{EColumnType::kReal32}});
      }
      if (f.GetTypeName() == "std::uint32_t") {
         f.SetColumnRepresentatives({{EColumnType::kUInt32}});
      }
   }

   auto clone = model->Clone();

   for (const auto &f : clone->GetFieldZero()) {
      if (f.GetTypeName() == "float") {
         EXPECT_EQ(EColumnType::kReal32, f.GetColumnRepresentatives()[0][0]);
      }
      if (f.GetTypeName() == "std::uint32_t") {
         EXPECT_EQ(EColumnType::kUInt32, f.GetColumnRepresentatives()[0][0]);
      }
   }
   EXPECT_TRUE(clone->GetField("struct").GetTraits() & RFieldBase::kTraitTypeChecksum);
   EXPECT_TRUE(clone->GetField("obj").GetTraits() & RFieldBase::kTraitTypeChecksum);
}
