#include "ntuple_test.hxx"

TEST(RNTupleParallelWriter, Basics)
{
   FileRaii fileGuard("test_ntuple_parallel_basics.root");

   auto test = [&](std::unique_ptr<RNTupleParallelWriter> writer) {
      // Create two RNTupleFillContext to prepare clusters in parallel.
      auto c1 = writer->CreateFillContext();
      auto e1 = c1->CreateEntry();
      auto pt1 = e1->GetPtr<float>("pt");

      auto c2 = writer->CreateFillContext();
      auto e2 = c2->CreateEntry();
      auto pt2 = e2->GetPtr<float>("pt");

      // Fill one entry per context and commit a cluster each.
      *pt1 = 1.0;
      c1->Fill(*e1);
      c1->FlushCluster();

      *pt2 = 2.0;
      c2->Fill(*e2);
      c2->FlushCluster();

      // The two contexts should act independently.
      EXPECT_EQ(c1->GetNEntries(), 1);
      EXPECT_EQ(c2->GetNEntries(), 1);
      EXPECT_EQ(c1->GetLastFlushed(), 1);
      EXPECT_EQ(c2->GetLastFlushed(), 1);

      // Fill another entry per context without explicitly committing a cluster.
      *pt1 = 3.0;
      c1->Fill(*e1);

      *pt2 = 4.0;
      c2->Fill(*e2);

      EXPECT_EQ(c1->GetNEntries(), 2);
      EXPECT_EQ(c2->GetNEntries(), 2);

      // Release the contexts (in reverse order) and the writer.
      c2.reset();
      c1.reset();
      writer.reset();

      auto reader = RNTupleReader::Open("f", fileGuard.GetPath());
      const auto &model = reader->GetModel();

      EXPECT_EQ(reader->GetNEntries(), 4);
      auto pt = model.GetDefaultEntry().GetPtr<float>("pt");

      reader->LoadEntry(0);
      EXPECT_EQ(*pt, 1.0);
      reader->LoadEntry(1);
      EXPECT_EQ(*pt, 2.0);

      // This entry ordering is enforced by the context destruction.
      reader->LoadEntry(2);
      EXPECT_EQ(*pt, 4.0);
      reader->LoadEntry(3);
      EXPECT_EQ(*pt, 3.0);
   };

   {
      auto model = RNTupleModel::CreateBare();
      model->MakeField<float>("pt");

      auto writer = RNTupleParallelWriter::Recreate(std::move(model), "f", fileGuard.GetPath());
      test(std::move(writer));
   }

   {
      auto model = RNTupleModel::CreateBare();
      model->MakeField<float>("pt");

      std::unique_ptr<TFile> file(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
      auto writer = RNTupleParallelWriter::Append(std::move(model), "f", *file);
      test(std::move(writer));
   }
}

TEST(RNTupleParallelWriter, Tokens)
{
   FileRaii fileGuard("test_ntuple_parallel_tokens.root");

   auto model = RNTupleModel::CreateBare();
   model->MakeField<float>("pt");
   auto token = model->GetToken("pt");

   auto writer = RNTupleParallelWriter::Recreate(std::move(model), "f", fileGuard.GetPath());
   auto context = writer->CreateFillContext();
   auto entry = context->CreateEntry();
   auto pt = entry->GetPtr<float>(token);
}

TEST(RNTupleParallelWriter, Staged)
{
   FileRaii fileGuard("test_ntuple_parallel_staged.root");

   {
      auto model = RNTupleModel::CreateBare();
      model->MakeField<float>("pt");

      auto writer = RNTupleParallelWriter::Recreate(std::move(model), "f", fileGuard.GetPath());

      // Create two RNTupleFillContext to prepare clusters in parallel.
      auto c1 = writer->CreateFillContext();
      auto e1 = c1->CreateEntry();
      auto pt1 = e1->GetPtr<float>("pt");

      auto c2 = writer->CreateFillContext();
      auto e2 = c2->CreateEntry();
      auto pt2 = e2->GetPtr<float>("pt");

      // Turn on staged cluster committing to logically append staged clusters with an explicit call.
      c1->EnableStagedClusterCommitting();
      c2->EnableStagedClusterCommitting();

      // Fill one entry per context and stage a cluster each.
      *pt1 = 3.0;
      c1->Fill(*e1);
      c1->FlushCluster();

      *pt2 = 1.0;
      c2->Fill(*e2);
      c2->FlushCluster();

      // Stage another cluster per context.
      *pt1 = 4.0;
      c1->Fill(*e1);
      c1->FlushCluster();

      *pt2 = 2.0;
      c2->Fill(*e2);
      c2->FlushCluster();

      // Commit the staged clusters.
      c2->CommitStagedClusters();
      c1->CommitStagedClusters();
   }

   auto reader = RNTupleReader::Open("f", fileGuard.GetPath());
   const auto &model = reader->GetModel();

   ASSERT_EQ(reader->GetNEntries(), 4);
   auto pt = model.GetDefaultEntry().GetPtr<float>("pt");

   reader->LoadEntry(0);
   EXPECT_EQ(*pt, 1.0);
   reader->LoadEntry(1);
   EXPECT_EQ(*pt, 2.0);
   reader->LoadEntry(2);
   EXPECT_EQ(*pt, 3.0);
   reader->LoadEntry(3);
   EXPECT_EQ(*pt, 4.0);
}

TEST(RNTupleParallelWriter, StagedMultiColumn)
{
   // Based on MultiColumnRepresentationSimple from ntuple_multi_column.cxx
   FileRaii fileGuard("test_ntuple_parallel_staged_multi_column.root");

   {
      auto model = RNTupleModel::CreateBare();
      auto fldPx = RFieldBase::Create("px", "float").Unwrap();
      fldPx->SetColumnRepresentatives({{ROOT::ENTupleColumnType::kReal32}, {ROOT::ENTupleColumnType::kReal16}});
      model->AddField(std::move(fldPx));

      auto writer = RNTupleParallelWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());

      auto c = writer->CreateFillContext();
      c->EnableStagedClusterCommitting();
      auto e = c->CreateEntry();
      auto px = e->GetPtr<float>("px");

      *px = 1.0;
      c->Fill(*e);
      c->FlushCluster();
      ROOT::Experimental::Internal::RFieldRepresentationModifier::SetPrimaryColumnRepresentation(
         const_cast<RFieldBase &>(c->GetModel().GetConstField("px")), 1);
      *px = 2.0;
      c->Fill(*e);
      c->FlushCluster();
      ROOT::Experimental::Internal::RFieldRepresentationModifier::SetPrimaryColumnRepresentation(
         const_cast<RFieldBase &>(c->GetModel().GetConstField("px")), 0);
      *px = 3.0;
      c->Fill(*e);
      c->FlushCluster();

      c->CommitStagedClusters();
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
}

TEST(RNTupleParallelWriter, Options)
{
   FileRaii fileGuard("test_ntuple_parallel_options.root");

   RNTupleWriteOptions options;
   options.SetUseBufferedWrite(false);

   try {
      auto model = RNTupleModel::CreateBare();
      RNTupleParallelWriter::Recreate(std::move(model), "f", fileGuard.GetPath(), options);
      FAIL() << "should require buffered writing";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("parallel writing requires buffering"));
   }

   try {
      std::unique_ptr<TFile> file(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
      auto model = RNTupleModel::CreateBare();
      RNTupleParallelWriter::Append(std::move(model), "f", *file, options);
      FAIL() << "should require buffered writing";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("parallel writing requires buffering"));
   }
}

TEST(RNTupleParallelWriter, ForbidModelWithSubfields)
{
   FileRaii fileGuard("test_ntuple_forbid_model_with_subfields.root");

   auto model = RNTupleModel::Create();
   model->MakeField<CustomStruct>("struct");
   model->RegisterSubfield("struct.a");

   try {
      auto writer = RNTupleParallelWriter::Recreate(std::move(model), "f", fileGuard.GetPath());
      FAIL() << "should not able to create a writer using a model with registered subfields";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(),
                  testing::HasSubstr("cannot create an RNTupleParallelWriter from a model with registered subfields"));
   }
}

TEST(RNTupleParallelWriter, ForbidNonRootTFiles)
{
   FileRaii fileGuard("test_ntuple_parallel_forbid_xml.xml");

   auto model = RNTupleModel::Create();
   auto file = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
   // Opening an XML TFile should fail
   EXPECT_THROW(RNTupleParallelWriter::Append(std::move(model), "ntpl", *file), ROOT::RException);
}

TEST(RNTupleFillContext, FlushColumns)
{
   FileRaii fileGuard("test_ntuple_context_flush.root");

   {
      auto model = RNTupleModel::CreateBare();
      model->MakeField<float>("pt");

      auto writer = RNTupleParallelWriter::Recreate(std::move(model), "f", fileGuard.GetPath());

      auto c = writer->CreateFillContext();
      auto e = c->CreateEntry();
      auto pt = e->GetPtr<float>("pt");

      *pt = 1.0;
      c->Fill(*e);

      c->FlushColumns();

      *pt = 2.0;
      c->Fill(*e);
   }

   // If FlushColumns() worked, there will be two pages with one element each.
   auto reader = RNTupleReader::Open("f", fileGuard.GetPath());
   const auto &descriptor = reader->GetDescriptor();

   auto fieldId = descriptor.FindFieldId("pt");
   auto columnId = descriptor.FindPhysicalColumnId(fieldId, 0, 0);
   auto &pageInfos = descriptor.GetClusterDescriptor(0).GetPageRange(columnId).GetPageInfos();
   ASSERT_EQ(pageInfos.size(), 2);
   EXPECT_EQ(pageInfos[0].GetNElements(), 1);
   EXPECT_EQ(pageInfos[1].GetNElements(), 1);
}

TEST(RNTupleParallelWriter, ExplicitCommit)
{
   FileRaii fileGuard("test_ntuple_parallel_explicit_commit.root");

   auto model = RNTupleModel::CreateBare();
   model->MakeField<float>("pt");
   auto writer = RNTupleParallelWriter::Recreate(std::move(model), "f", fileGuard.GetPath());

   auto ctx = writer->CreateFillContext();
   auto entry = ctx->CreateEntry();
   ctx->Fill(*entry);

   EXPECT_THROW(writer->CommitDataset(), ROOT::RException);
   ctx.reset();

   writer->CommitDataset();
   writer->CommitDataset(); // noop
}
