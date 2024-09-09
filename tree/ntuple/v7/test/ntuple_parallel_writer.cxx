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

TEST(RNTupleParallelWriter, Options)
{
   FileRaii fileGuard("test_ntuple_parallel_options.root");

   RNTupleWriteOptions options;
   options.SetUseBufferedWrite(false);

   try {
      auto model = RNTupleModel::CreateBare();
      RNTupleParallelWriter::Recreate(std::move(model), "f", fileGuard.GetPath(), options);
      FAIL() << "should require buffered writing";
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("parallel writing requires buffering"));
   }

   try {
      std::unique_ptr<TFile> file(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
      auto model = RNTupleModel::CreateBare();
      RNTupleParallelWriter::Append(std::move(model), "f", *file, options);
      FAIL() << "should require buffered writing";
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("parallel writing requires buffering"));
   }
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
   auto &pageInfos = descriptor.GetClusterDescriptor(0).GetPageRange(columnId).fPageInfos;
   ASSERT_EQ(pageInfos.size(), 2);
   EXPECT_EQ(pageInfos[0].fNElements, 1);
   EXPECT_EQ(pageInfos[1].fNElements, 1);
}
