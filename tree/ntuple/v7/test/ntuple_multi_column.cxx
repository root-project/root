#include "ntuple_test.hxx"

TEST(RNTuple, MultiColumnRepresentationSimple)
{
   FileRaii fileGuard("test_ntuple_multi_column_representation_simple.root");

   {
      auto model = RNTupleModel::Create();
      auto fldPx = RFieldBase::Create("px", "float").Unwrap();
      fldPx->SetColumnRepresentatives({{ROOT::ENTupleColumnType::kReal32}, {ROOT::ENTupleColumnType::kReal16}});
      model->AddField(std::move(fldPx));
      auto ptrPx = model->GetDefaultEntry().GetPtr<float>("px");
      RNTupleWriteOptions options;
      options.SetCompression(0);
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath(), options);
      *ptrPx = 1.0;
      writer->Fill();
      writer->CommitCluster();
      ROOT::Experimental::Internal::RFieldRepresentationModifier::SetPrimaryColumnRepresentation(
         const_cast<RFieldBase &>(writer->GetModel().GetConstField("px")), 1);
      *ptrPx = 2.0;
      writer->Fill();
      writer->CommitCluster();
      ROOT::Experimental::Internal::RFieldRepresentationModifier::SetPrimaryColumnRepresentation(
         const_cast<RFieldBase &>(writer->GetModel().GetConstField("px")), 0);
      *ptrPx = 3.0;
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   EXPECT_EQ(3u, reader->GetView<float>("px").GetFieldRange().size());

   const auto &desc = reader->GetDescriptor();
   EXPECT_EQ(3u, desc.GetNClusters());

   const auto &fieldDesc = desc.GetFieldDescriptor(desc.FindFieldId("px"));
   EXPECT_EQ(1u, fieldDesc.GetColumnCardinality());
   EXPECT_EQ(2u, fieldDesc.GetLogicalColumnIds().size());

   const auto &colDesc32 = desc.GetColumnDescriptor(fieldDesc.GetLogicalColumnIds()[0]);
   const auto &colDesc16 = desc.GetColumnDescriptor(fieldDesc.GetLogicalColumnIds()[1]);

   EXPECT_EQ(ROOT::ENTupleColumnType::kReal32, colDesc32.GetType());
   EXPECT_EQ(0u, colDesc32.GetRepresentationIndex());
   EXPECT_EQ(ROOT::ENTupleColumnType::kReal16, colDesc16.GetType());
   EXPECT_EQ(1u, colDesc16.GetRepresentationIndex());

   const auto &clusterDesc0 = desc.GetClusterDescriptor(0);
   EXPECT_FALSE(clusterDesc0.GetColumnRange(colDesc32.GetPhysicalId()).IsSuppressed());
   EXPECT_EQ(1u, clusterDesc0.GetColumnRange(colDesc32.GetPhysicalId()).GetNElements());
   EXPECT_EQ(0u, clusterDesc0.GetColumnRange(colDesc32.GetPhysicalId()).GetFirstElementIndex());
   EXPECT_TRUE(clusterDesc0.GetColumnRange(colDesc16.GetPhysicalId()).IsSuppressed());
   EXPECT_EQ(1u, clusterDesc0.GetColumnRange(colDesc16.GetPhysicalId()).GetNElements());
   EXPECT_EQ(0u, clusterDesc0.GetColumnRange(colDesc16.GetPhysicalId()).GetFirstElementIndex());
   EXPECT_FALSE(clusterDesc0.GetColumnRange(colDesc16.GetPhysicalId()).GetCompressionSettings());

   const auto &clusterDesc1 = desc.GetClusterDescriptor(1);
   EXPECT_FALSE(clusterDesc1.GetColumnRange(colDesc16.GetPhysicalId()).IsSuppressed());
   EXPECT_EQ(1u, clusterDesc1.GetColumnRange(colDesc16.GetPhysicalId()).GetNElements());
   EXPECT_EQ(1u, clusterDesc1.GetColumnRange(colDesc16.GetPhysicalId()).GetFirstElementIndex());
   EXPECT_TRUE(clusterDesc1.GetColumnRange(colDesc32.GetPhysicalId()).IsSuppressed());
   EXPECT_EQ(1u, clusterDesc1.GetColumnRange(colDesc32.GetPhysicalId()).GetNElements());
   EXPECT_EQ(1u, clusterDesc1.GetColumnRange(colDesc32.GetPhysicalId()).GetFirstElementIndex());
   EXPECT_FALSE(clusterDesc1.GetColumnRange(colDesc32.GetPhysicalId()).GetCompressionSettings());

   const auto &clusterDesc2 = desc.GetClusterDescriptor(2);
   EXPECT_FALSE(clusterDesc2.GetColumnRange(colDesc32.GetPhysicalId()).IsSuppressed());
   EXPECT_EQ(1u, clusterDesc2.GetColumnRange(colDesc32.GetPhysicalId()).GetNElements());
   EXPECT_EQ(2u, clusterDesc2.GetColumnRange(colDesc32.GetPhysicalId()).GetFirstElementIndex());
   EXPECT_TRUE(clusterDesc2.GetColumnRange(colDesc16.GetPhysicalId()).IsSuppressed());
   EXPECT_EQ(1u, clusterDesc2.GetColumnRange(colDesc16.GetPhysicalId()).GetNElements());
   EXPECT_EQ(2u, clusterDesc2.GetColumnRange(colDesc16.GetPhysicalId()).GetFirstElementIndex());
   EXPECT_FALSE(clusterDesc2.GetColumnRange(colDesc16.GetPhysicalId()).GetCompressionSettings());

   auto ptrPx = reader->GetModel().GetDefaultEntry().GetPtr<float>("px");
   reader->LoadEntry(0);
   EXPECT_FLOAT_EQ(1.0, *ptrPx);
   reader->LoadEntry(1);
   EXPECT_FLOAT_EQ(2.0, *ptrPx);
   reader->LoadEntry(2);
   EXPECT_FLOAT_EQ(3.0, *ptrPx);

   auto viewPx = reader->GetView<float>("px");
   EXPECT_FLOAT_EQ(1.0, viewPx(0));
   EXPECT_FLOAT_EQ(2.0, viewPx(1));
   EXPECT_FLOAT_EQ(3.0, viewPx(2));

   std::ostringstream osDetails;
   reader->PrintInfo(ROOT::Experimental::ENTupleInfo::kStorageDetails, osDetails);
   const std::string reference = std::string("") + "============================================================\n"
                                                   "NTUPLE:      ntpl\n"
                                                   "Compression: 0\n"
                                                   "------------------------------------------------------------\n"
                                                   "  # Entries:        3\n"
                                                   "  # Fields:         2\n"
                                                   "  # Columns:        2\n"
                                                   "  # Alias Columns:  0\n"
                                                   "  # Pages:          3\n"
                                                   "  # Clusters:       3\n"
                                                   "  Size on storage:  .* B\n"
                                                   "  Compression rate: .*\n"
                                                   "  Header size:      .* B\n"
                                                   "  Footer size:      .* B\n"
                                                   "  Meta-data / data: .*\n"
                                                   "------------------------------------------------------------\n"
                                                   "CLUSTER DETAILS\n"
                                                   "------------------------------------------------------------\n"
                                                   "  #     0   Entry range:     .0..0.  --  1\n"
                                                   "            # Pages:         1\n"
                                                   "            Size on storage: 4 B\n"
                                                   "            Compression:     1.00\n"
                                                   "  #     1   Entry range:     .1..1.  --  1\n"
                                                   "            # Pages:         1\n"
                                                   "            Size on storage: 2 B\n"
                                                   "            Compression:     2.00\n"
                                                   "  #     2   Entry range:     .2..2.  --  1\n"
                                                   "            # Pages:         1\n"
                                                   "            Size on storage: 4 B\n"
                                                   "            Compression:     1.00\n"
                                                   "------------------------------------------------------------\n"
                                                   "COLUMN DETAILS\n"
                                                   "------------------------------------------------------------\n"
                                                   "  px .#0.  --  Real32                                 .id:0.\n"
                                                   "    # Elements:          2\n"
                                                   "    # Pages:             2\n"
                                                   "    Avg elements / page: 1\n"
                                                   "    Avg page size:       4 B\n"
                                                   "    Size on storage:     8 B\n"
                                                   "    Compression:         1.00\n"
                                                   "............................................................\n"
                                                   "  px .#0 / R.1.  --  Real16                           .id:1.\n"
                                                   "    # Elements:          1\n"
                                                   "    # Pages:             1\n"
                                                   "    Avg elements / page: 1\n"
                                                   "    Avg page size:       2 B\n"
                                                   "    Size on storage:     2 B\n"
                                                   "    Compression:         2.00\n"
                                                   "............................................................\n";
   EXPECT_THAT(osDetails.str(), testing::MatchesRegex(reference));
}

TEST(RNTuple, MultiColumnRepresentationString)
{
   FileRaii fileGuard("test_ntuple_multi_column_representation_string.root");

   {
      auto model = RNTupleModel::Create();
      auto fldStr = RFieldBase::Create("str", "std::string").Unwrap();
      fldStr->SetColumnRepresentatives({{ROOT::ENTupleColumnType::kIndex32, ROOT::ENTupleColumnType::kChar},
                                        {ROOT::ENTupleColumnType::kSplitIndex64, ROOT::ENTupleColumnType::kChar}});
      model->AddField(std::move(fldStr));
      auto ptrStr = model->GetDefaultEntry().GetPtr<std::string>("str");
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      *ptrStr = "abc";
      writer->Fill();
      writer->CommitCluster();
      ROOT::Experimental::Internal::RFieldRepresentationModifier::SetPrimaryColumnRepresentation(
         const_cast<RFieldBase &>(writer->GetModel().GetConstField("str")), 1);
      ptrStr->clear();
      writer->Fill();
      writer->CommitCluster();
      ROOT::Experimental::Internal::RFieldRepresentationModifier::SetPrimaryColumnRepresentation(
         const_cast<RFieldBase &>(writer->GetModel().GetConstField("str")), 0);
      writer->Fill();
      writer->CommitCluster();
      ROOT::Experimental::Internal::RFieldRepresentationModifier::SetPrimaryColumnRepresentation(
         const_cast<RFieldBase &>(writer->GetModel().GetConstField("str")), 1);
      *ptrStr = "x";
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   EXPECT_EQ(4u, reader->GetNEntries());
   auto ptrStr = reader->GetModel().GetDefaultEntry().GetPtr<std::string>("str");
   reader->LoadEntry(0);
   EXPECT_EQ("abc", *ptrStr);
   reader->LoadEntry(1);
   EXPECT_TRUE(ptrStr->empty());
   reader->LoadEntry(2);
   EXPECT_TRUE(ptrStr->empty());
   reader->LoadEntry(3);
   EXPECT_EQ("x", *ptrStr);
}

TEST(RNTuple, MultiColumnRepresentationVector)
{
   FileRaii fileGuard("test_ntuple_multi_column_representation_vector.root");

   {
      auto model = RNTupleModel::Create();
      auto fldVec = RFieldBase::Create("vec", "std::vector<float>").Unwrap();
      fldVec->SetColumnRepresentatives({{ROOT::ENTupleColumnType::kIndex32}, {ROOT::ENTupleColumnType::kSplitIndex64}});
      model->AddField(std::move(fldVec));
      auto ptrVec = model->GetDefaultEntry().GetPtr<std::vector<float>>("vec");
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      writer->Fill();
      writer->CommitCluster();
      ROOT::Experimental::Internal::RFieldRepresentationModifier::SetPrimaryColumnRepresentation(
         const_cast<RFieldBase &>(writer->GetModel().GetConstField("vec")), 1);
      ptrVec->push_back(1.0);
      writer->Fill();
      writer->CommitCluster();
      ROOT::Experimental::Internal::RFieldRepresentationModifier::SetPrimaryColumnRepresentation(
         const_cast<RFieldBase &>(writer->GetModel().GetConstField("vec")), 0);
      ptrVec->clear();
      writer->Fill();
      writer->CommitCluster();
      ROOT::Experimental::Internal::RFieldRepresentationModifier::SetPrimaryColumnRepresentation(
         const_cast<RFieldBase &>(writer->GetModel().GetConstField("vec")), 1);
      ptrVec->push_back(2.0);
      ptrVec->push_back(3.0);
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   EXPECT_EQ(4u, reader->GetNEntries());
   auto ptrVec = reader->GetModel().GetDefaultEntry().GetPtr<std::vector<float>>("vec");
   reader->LoadEntry(0);
   EXPECT_TRUE(ptrVec->empty());
   reader->LoadEntry(1);
   EXPECT_EQ(1u, ptrVec->size());
   EXPECT_FLOAT_EQ(1.0, ptrVec->at(0));
   reader->LoadEntry(2);
   EXPECT_TRUE(ptrVec->empty());
   reader->LoadEntry(3);
   EXPECT_EQ(2u, ptrVec->size());
   EXPECT_FLOAT_EQ(2.0, ptrVec->at(0));
   EXPECT_FLOAT_EQ(3.0, ptrVec->at(1));
}

TEST(RNTuple, MultiColumnRepresentationMany)
{
   FileRaii fileGuard("test_ntuple_multi_column_representation_many.root");

   {
      auto model = RNTupleModel::Create();
      auto fldVec = RFieldBase::Create("vec", "std::vector<float>").Unwrap();
      fldVec->SetColumnRepresentatives({{ROOT::ENTupleColumnType::kIndex32},
                                        {ROOT::ENTupleColumnType::kSplitIndex64},
                                        {ROOT::ENTupleColumnType::kIndex64},
                                        {ROOT::ENTupleColumnType::kSplitIndex32}});
      model->AddField(std::move(fldVec));
      auto ptrVec = model->GetDefaultEntry().GetPtr<std::vector<float>>("vec");
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      ptrVec->push_back(1.0);
      writer->Fill();
      writer->CommitCluster();
      ROOT::Experimental::Internal::RFieldRepresentationModifier::SetPrimaryColumnRepresentation(
         const_cast<RFieldBase &>(writer->GetModel().GetConstField("vec")), 1);
      (*ptrVec)[0] = 2.0;
      writer->Fill();
      writer->CommitCluster();
      ROOT::Experimental::Internal::RFieldRepresentationModifier::SetPrimaryColumnRepresentation(
         const_cast<RFieldBase &>(writer->GetModel().GetConstField("vec")), 2);
      (*ptrVec)[0] = 3.0;
      writer->Fill();
      writer->CommitCluster();
      ROOT::Experimental::Internal::RFieldRepresentationModifier::SetPrimaryColumnRepresentation(
         const_cast<RFieldBase &>(writer->GetModel().GetConstField("vec")), 3);
      (*ptrVec)[0] = 4.0;
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   EXPECT_EQ(4u, reader->GetNEntries());
   auto ptrVec = reader->GetModel().GetDefaultEntry().GetPtr<std::vector<float>>("vec");
   reader->LoadEntry(0);
   EXPECT_EQ(1u, ptrVec->size());
   EXPECT_FLOAT_EQ(1.0, ptrVec->at(0));
   reader->LoadEntry(1);
   EXPECT_EQ(1u, ptrVec->size());
   EXPECT_FLOAT_EQ(2.0, ptrVec->at(0));
   reader->LoadEntry(2);
   EXPECT_EQ(1u, ptrVec->size());
   EXPECT_FLOAT_EQ(3.0, ptrVec->at(0));
   reader->LoadEntry(3);
   EXPECT_EQ(1u, ptrVec->size());
   EXPECT_FLOAT_EQ(4.0, ptrVec->at(0));
}

TEST(RNTuple, MultiColumnRepresentationNullable)
{
   FileRaii fileGuard("test_ntuple_multi_column_representation_nullable.root");

   {
      auto model = RNTupleModel::Create();
      auto fldScalar = RFieldBase::Create("scalar", "std::optional<float>").Unwrap();
      auto fldVector = RFieldBase::Create("vector", "std::vector<std::optional<float>>").Unwrap();
      fldScalar->SetColumnRepresentatives(
         {{ROOT::ENTupleColumnType::kIndex32}, {ROOT::ENTupleColumnType::kSplitIndex64}});
      fldVector->GetMutableSubfields()[0]->SetColumnRepresentatives(
         {{ROOT::ENTupleColumnType::kSplitIndex64}, {ROOT::ENTupleColumnType::kIndex32}});
      model->AddField(std::move(fldScalar));
      model->AddField(std::move(fldVector));
      auto ptrScalar = model->GetDefaultEntry().GetPtr<std::optional<float>>("scalar");
      auto ptrVector = model->GetDefaultEntry().GetPtr<std::vector<std::optional<float>>>("vector");
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      *ptrScalar = 1.0;
      ptrVector->push_back(13.0);
      writer->Fill();
      writer->CommitCluster();
      ROOT::Experimental::Internal::RFieldRepresentationModifier::SetPrimaryColumnRepresentation(
         const_cast<RFieldBase &>(writer->GetModel().GetConstField("scalar")), 1);
      ROOT::Experimental::Internal::RFieldRepresentationModifier::SetPrimaryColumnRepresentation(
         const_cast<RFieldBase &>(writer->GetModel().GetConstField("vector._0")), 1);
      ptrScalar->reset();
      ptrVector->clear();
      ptrVector->push_back(std::optional<float>());
      writer->Fill();
      writer->CommitCluster();
      ROOT::Experimental::Internal::RFieldRepresentationModifier::SetPrimaryColumnRepresentation(
         const_cast<RFieldBase &>(writer->GetModel().GetConstField("scalar")), 0);
      ROOT::Experimental::Internal::RFieldRepresentationModifier::SetPrimaryColumnRepresentation(
         const_cast<RFieldBase &>(writer->GetModel().GetConstField("vector._0")), 0);
      *ptrScalar = 3.0;
      ptrVector->clear();
      ptrVector->push_back(15.0);
      ptrVector->push_back(17.0);
      writer->Fill();
      writer->CommitCluster();
      ptrScalar->reset();
      ptrVector->clear();
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   EXPECT_EQ(4u, reader->GetNEntries());
   auto ptrScalar = reader->GetModel().GetDefaultEntry().GetPtr<std::optional<float>>("scalar");
   auto ptrVector = reader->GetModel().GetDefaultEntry().GetPtr<std::vector<std::optional<float>>>("vector");
   reader->LoadEntry(0);
   EXPECT_FLOAT_EQ(1.0, ptrScalar->value());
   EXPECT_FLOAT_EQ(1u, ptrVector->size());
   EXPECT_FLOAT_EQ(13.0, ptrVector->at(0).value());
   reader->LoadEntry(1);
   EXPECT_FALSE(ptrScalar->has_value());
   EXPECT_FLOAT_EQ(1u, ptrVector->size());
   EXPECT_FALSE(ptrVector->at(0).has_value());
   reader->LoadEntry(2);
   EXPECT_FLOAT_EQ(3.0, ptrScalar->value());
   EXPECT_FLOAT_EQ(2u, ptrVector->size());
   EXPECT_FLOAT_EQ(15.0, ptrVector->at(0).value());
   EXPECT_FLOAT_EQ(17.0, ptrVector->at(1).value());
   reader->LoadEntry(3);
   EXPECT_FALSE(ptrScalar->has_value());
   EXPECT_TRUE(ptrVector->empty());
}

TEST(RNTuple, MultiColumnRepresentationBulk)
{
   FileRaii fileGuard("test_ntuple_multi_column_representation_bulk.root");

   {
      auto model = RNTupleModel::Create();
      auto fldPx = RFieldBase::Create("px", "float").Unwrap();
      fldPx->SetColumnRepresentatives({{ROOT::ENTupleColumnType::kReal32}, {ROOT::ENTupleColumnType::kReal16}});
      model->AddField(std::move(fldPx));
      auto ptrPx = model->GetDefaultEntry().GetPtr<float>("px");
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      *ptrPx = 1.0;
      writer->Fill();
      writer->CommitCluster();
      ROOT::Experimental::Internal::RFieldRepresentationModifier::SetPrimaryColumnRepresentation(
         const_cast<RFieldBase &>(writer->GetModel().GetConstField("px")), 1);
      *ptrPx = 2.0;
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   RFieldBase::RBulk bulk = reader->GetModel().CreateBulk("px");

   auto mask = std::make_unique<bool[]>(1);
   mask[0] = true;
   auto arr = static_cast<float *>(bulk.ReadBulk(RNTupleLocalIndex(0, 0), mask.get(), 1));
   EXPECT_FLOAT_EQ(1.0, arr[0]);

   arr = static_cast<float *>(bulk.ReadBulk(RNTupleLocalIndex(1, 0), mask.get(), 1));
   EXPECT_FLOAT_EQ(2.0, arr[0]);
}

TEST(RNTuple, MultiColumnRepresentationFriends)
{
   FileRaii fileGuard1("test_ntuple_multi_column_representation_friend1.root");
   FileRaii fileGuard2("test_ntuple_multi_column_representation_friend2.root");

   auto model1 = RNTupleModel::Create();
   auto fldPt = RFieldBase::Create("pt", "float").Unwrap();
   fldPt->SetColumnRepresentatives({{ROOT::ENTupleColumnType::kReal32}, {ROOT::ENTupleColumnType::kReal16}});
   model1->AddField(std::move(fldPt));
   auto ptrPt = model1->GetDefaultEntry().GetPtr<float>("pt");

   auto model2 = RNTupleModel::Create();
   auto fldEta = RFieldBase::Create("eta", "float").Unwrap();
   fldEta->SetColumnRepresentatives({{ROOT::ENTupleColumnType::kReal16}, {ROOT::ENTupleColumnType::kReal32}});
   model2->AddField(std::move(fldEta));
   auto ptrEta = model2->GetDefaultEntry().GetPtr<float>("eta");

   {
      auto writer = RNTupleWriter::Recreate(std::move(model1), "ntpl1", fileGuard1.GetPath());
      *ptrPt = 1.0;
      writer->Fill();
      writer->CommitCluster();
      ROOT::Experimental::Internal::RFieldRepresentationModifier::SetPrimaryColumnRepresentation(
         const_cast<RFieldBase &>(writer->GetModel().GetConstField("pt")), 1);
      *ptrPt = 2.0;
      writer->Fill();
   }
   {
      auto writer = RNTupleWriter::Recreate(std::move(model2), "ntpl2", fileGuard2.GetPath());
      *ptrEta = 3.0;
      writer->Fill();
      writer->CommitCluster();
      ROOT::Experimental::Internal::RFieldRepresentationModifier::SetPrimaryColumnRepresentation(
         const_cast<RFieldBase &>(writer->GetModel().GetConstField("eta")), 1);
      *ptrEta = 4.0;
      writer->Fill();
   }
}

TEST(RNTuple, MultiColumnRepresentationDedup)
{
   FileRaii fileGuard("test_ntuple_multi_column_representation_dedup.root");

   auto fldPx = RFieldBase::Create("px", "float").Unwrap();
   fldPx->SetColumnRepresentatives({{ROOT::ENTupleColumnType::kReal16}, {ROOT::ENTupleColumnType::kReal16}});
   EXPECT_EQ(fldPx->GetColumnRepresentatives().size(), 1);
}
