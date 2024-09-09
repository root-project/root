#include "ntuple_test.hxx"

TEST(RNTuple, ReconstructModel)
{
   FileRaii fileGuard("test_ntuple_reconstruct.root");
   auto model = RNTupleModel::Create();
   auto fieldPt = model->MakeField<float>("pt", 42.0);
   auto fieldNnlo = model->MakeField<std::vector<std::vector<float>>>("nnlo");
   auto fieldKlass = model->MakeField<CustomStruct>("klass");
   auto fieldArray = model->MakeField<std::array<double, 2>>("array");
   auto fieldVariant = model->MakeField<std::variant<double, std::variant<std::string, double>>>("variant");
   model->Freeze();
   {
      RPageSinkFile sink("myNTuple", fileGuard.GetPath(), RNTupleWriteOptions());
      sink.Init(*model.get());
      sink.CommitClusterGroup();
      sink.CommitDataset();
      model = nullptr;
   }

   RPageSourceFile source("myNTuple", fileGuard.GetPath(), RNTupleReadOptions());
   source.Attach();
   try {
      source.SetEntryRange({0, source.GetNEntries() + 1});
      FAIL() << "invalid entry range should throw";
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("invalid entry range"));
   }
   // Should not throw
   source.SetEntryRange({0, source.GetNEntries()});

   auto modelReconstructed = source.GetSharedDescriptorGuard()->CreateModel();
   try {
      modelReconstructed->GetDefaultEntry().GetPtr<float>("xyz");
      FAIL() << "invalid field name should throw";
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("invalid field name"));
   }
   auto vecPtr = modelReconstructed->GetDefaultEntry().GetPtr<std::vector<std::vector<float>>>("nnlo");
   EXPECT_TRUE(vecPtr != nullptr);
   // Don't crash
   vecPtr->push_back(std::vector<float>{1.0});
   auto array = modelReconstructed->GetDefaultEntry().GetPtr<std::array<double, 2>>("array");
   EXPECT_TRUE(array != nullptr);
   auto variant =
      modelReconstructed->GetDefaultEntry().GetPtr<std::variant<double, std::variant<std::string, double>>>("variant");
   EXPECT_TRUE(variant != nullptr);
}

TEST(RNTuple, MultipleInFile)
{
   FileRaii fileGuard("test_ntuple_multi.root");
   auto file = TFile::Open(fileGuard.GetPath().c_str(), "RECREATE");
   {
      auto model = RNTupleModel::Create();
      auto fieldPt = model->MakeField<float>("pt", 42.0);
      auto ntuple = RNTupleWriter::Append(std::move(model), "first", *file);
      ntuple->Fill();
   }
   {
      auto model = RNTupleModel::Create();
      auto fieldPt = model->MakeField<float>("E", 1.0);
      auto ntuple = RNTupleWriter::Append(std::move(model), "second", *file);
      ntuple->Fill();
   }
   file->Close();
   delete file;

   auto ntupleFirst = RNTupleReader::Open("first", fileGuard.GetPath());
   auto viewPt = ntupleFirst->GetView<float>("pt");
   int n = 0;
   for (auto i : ntupleFirst->GetEntryRange()) {
      EXPECT_EQ(42.0, viewPt(i));
      n++;
   }
   EXPECT_EQ(1, n);

   auto ntupleSecond = RNTupleReader::Open("second", fileGuard.GetPath());
   auto viewE = ntupleSecond->GetView<float>("E");
   n = 0;
   for (auto i : ntupleSecond->GetEntryRange()) {
      EXPECT_EQ(1.0, viewE(i));
      n++;
   }
   EXPECT_EQ(1, n);
}


TEST(RNTuple, WriteRead)
{
   FileRaii fileGuard("test_ntuple_writeread.root");

   auto modelWrite = RNTupleModel::Create();
   auto wrSignal = modelWrite->MakeField<bool>("signal", true);
   auto wrPt = modelWrite->MakeField<float>("pt", 42.0);
   auto wrEnergy = modelWrite->MakeField<float>("energy", 7.0);
   auto wrTag = modelWrite->MakeField<std::string>("tag", "xyz");
   auto wrJets = modelWrite->MakeField<std::vector<float>>("jets");
   wrJets->push_back(1.0);
   wrJets->push_back(2.0);
   auto wrNnlo = modelWrite->MakeField<std::vector<std::vector<float>>>("nnlo");
   wrNnlo->push_back(std::vector<float>());
   wrNnlo->push_back(std::vector<float>{1.0});
   wrNnlo->push_back(std::vector<float>{1.0, 2.0, 4.0, 8.0});
   auto wrKlass = modelWrite->MakeField<CustomStruct>("klass");
   wrKlass->s = "abc";

   modelWrite->Freeze();
   auto modelRead = modelWrite->Clone();

   {
      auto writer = RNTupleWriter::Recreate(std::move(modelWrite), "myNTuple", fileGuard.GetPath());
      writer->Fill();
   }

   auto rdSignal = modelRead->GetDefaultEntry().GetPtr<bool>("signal");
   auto rdPt = modelRead->GetDefaultEntry().GetPtr<float>("pt");
   auto rdEnergy = modelRead->GetDefaultEntry().GetPtr<float>("energy");
   auto rdTag = modelRead->GetDefaultEntry().GetPtr<std::string>("tag");
   auto rdJets = modelRead->GetDefaultEntry().GetPtr<std::vector<float>>("jets");
   auto rdNnlo = modelRead->GetDefaultEntry().GetPtr<std::vector<std::vector<float>>>("nnlo");
   auto rdKlass = modelRead->GetDefaultEntry().GetPtr<CustomStruct>("klass");

   auto reader = RNTupleReader::Open(std::move(modelRead), "myNTuple", fileGuard.GetPath());
   EXPECT_EQ(1U, reader->GetNEntries());
   reader->LoadEntry(0);

   EXPECT_TRUE(*rdSignal);
   EXPECT_EQ(42.0, *rdPt);
   EXPECT_EQ(7.0, *rdEnergy);
   EXPECT_STREQ("xyz", rdTag->c_str());

   EXPECT_EQ(2U, rdJets->size());
   EXPECT_EQ(1.0, (*rdJets)[0]);
   EXPECT_EQ(2.0, (*rdJets)[1]);

   EXPECT_EQ(3U, rdNnlo->size());
   EXPECT_EQ(0U, (*rdNnlo)[0].size());
   EXPECT_EQ(1U, (*rdNnlo)[1].size());
   EXPECT_EQ(4U, (*rdNnlo)[2].size());
   EXPECT_EQ(1.0, (*rdNnlo)[1][0]);
   EXPECT_EQ(1.0, (*rdNnlo)[2][0]);
   EXPECT_EQ(2.0, (*rdNnlo)[2][1]);
   EXPECT_EQ(4.0, (*rdNnlo)[2][2]);
   EXPECT_EQ(8.0, (*rdNnlo)[2][3]);

   EXPECT_STREQ("abc", rdKlass->s.c_str());
}

TEST(RNTuple, WriteReadInlinedModel)
{
   FileRaii fileGuard("test_ntuple_writeread_inlinedmodel.root");

   {
      auto writer = RNTupleWriter::Recreate(
         {
            {"std::uint32_t", "id"},
            {"std::vector<float>", "vpx"},
            {"std::vector<float>", "vpy"},
            {"std::vector<float>", "vpz"},
         },
         "NTuple", fileGuard.GetPath());

      auto entry = writer->CreateEntry();
      *entry->GetPtr<std::uint32_t>("id") = 1;
      *entry->GetPtr<std::vector<float>>("vpx") = {1.0, 1.1, 1.2};
      *entry->GetPtr<std::vector<float>>("vpy") = {2.0, 2.1, 2.2};
      *entry->GetPtr<std::vector<float>>("vpz") = {3.0, 3.1, 3.2};

      writer->Fill(*entry);
   }

   auto reader = RNTupleReader::Open("NTuple", fileGuard.GetPath());
   EXPECT_EQ(1U, reader->GetNEntries());
   reader->LoadEntry(0);
   auto readid = reader->GetModel().GetDefaultEntry().GetPtr<std::uint32_t>("id");
   EXPECT_EQ(1, *readid);
   auto readvpx = reader->GetModel().GetDefaultEntry().GetPtr<std::vector<float>>("vpx");
   EXPECT_FLOAT_EQ(1.0, (*readvpx)[0]);
   EXPECT_FLOAT_EQ(1.1, (*readvpx)[1]);
   EXPECT_FLOAT_EQ(1.2, (*readvpx)[2]);
   auto readvpy = reader->GetModel().GetDefaultEntry().GetPtr<std::vector<float>>("vpy");
   EXPECT_FLOAT_EQ(2.0, (*readvpy)[0]);
   EXPECT_FLOAT_EQ(2.1, (*readvpy)[1]);
   EXPECT_FLOAT_EQ(2.2, (*readvpy)[2]);
   auto readvpz = reader->GetModel().GetDefaultEntry().GetPtr<std::vector<float>>("vpz");
   EXPECT_FLOAT_EQ(3.0, (*readvpz)[0]);
   EXPECT_FLOAT_EQ(3.1, (*readvpz)[1]);
   EXPECT_FLOAT_EQ(3.2, (*readvpz)[2]);
}

TEST(RNTuple, FileAnchor)
{
   FileRaii fileGuard("test_ntuple_file_anchor.root");

   {
      auto model = RNTupleModel::Create();
      model->MakeField<int>("a", 42);
      auto writer = RNTupleWriter::Recreate(std::move(model), "A", fileGuard.GetPath());
      writer->Fill();
   }

   {
      auto model = RNTupleModel::Create();
      model->MakeField<int>("b", 137);
      auto f = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "UPDATE"));
      {
         auto writer = RNTupleWriter::Append(std::move(model), "B", *f);
         writer->Fill();
      }
      f->Close();
   }

   auto readerB = RNTupleReader::Open("B", fileGuard.GetPath());

   auto f = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str()));
   auto ntuple = std::unique_ptr<RNTuple>(f->Get<RNTuple>("A"));
   auto readerA = RNTupleReader::Open(*ntuple);

   EXPECT_EQ(1U, readerA->GetNEntries());
   EXPECT_EQ(1U, readerB->GetNEntries());

   auto a = readerA->GetModel().GetDefaultEntry().GetPtr<int>("a");
   auto b = readerB->GetModel().GetDefaultEntry().GetPtr<int>("b");
   readerA->LoadEntry(0);
   readerB->LoadEntry(0);
   EXPECT_EQ(42, *a);
   EXPECT_EQ(137, *b);
}

TEST(RNTuple, Clusters)
{
   FileRaii fileGuard("test_ntuple_clusters.root");

   auto modelWrite = RNTupleModel::Create();
   auto wrPt = modelWrite->MakeField<float>("pt", 42.0);
   auto wrTag = modelWrite->MakeField<std::string>("tag", "xyz");
   auto wrNnlo = modelWrite->MakeField<std::vector<std::vector<float>>>("nnlo");
   auto wrFourVec = modelWrite->MakeField<std::array<float, 4>>("fourVec");
   wrNnlo->push_back(std::vector<float>());
   wrNnlo->push_back(std::vector<float>{1.0});
   wrNnlo->push_back(std::vector<float>{1.0, 2.0, 4.0, 8.0});
   wrFourVec->at(0) = 0.0;
   wrFourVec->at(1) = 1.0;
   wrFourVec->at(2) = 2.0;
   wrFourVec->at(3) = 3.0;

   modelWrite->Freeze();
   auto modelRead = modelWrite->Clone();

   {
      auto writer = RNTupleWriter::Recreate(std::move(modelWrite), "myNTuple", fileGuard.GetPath());
      writer->Fill();
      writer->CommitCluster();
      *wrPt = 24.0;
      wrNnlo->clear();
      *wrTag = "";
      wrFourVec->at(2) = 42.0;
      writer->Fill();
      *wrPt = 12.0;
      wrNnlo->push_back(std::vector<float>{42.0});
      *wrTag = "12345";
      wrFourVec->at(1) = 24.0;
      writer->Fill();
   }

   auto rdPt = modelRead->GetDefaultEntry().GetPtr<float>("pt");
   auto rdTag = modelRead->GetDefaultEntry().GetPtr<std::string>("tag");
   auto rdNnlo = modelRead->GetDefaultEntry().GetPtr<std::vector<std::vector<float>>>("nnlo");
   auto rdFourVec = modelRead->GetDefaultEntry().GetPtr<std::array<float, 4>>("fourVec");

   auto reader = RNTupleReader::Open(std::move(modelRead), "myNTuple", fileGuard.GetPath());
   EXPECT_EQ(3U, reader->GetNEntries());

   reader->LoadEntry(0);
   EXPECT_EQ(42.0, *rdPt);
   EXPECT_STREQ("xyz", rdTag->c_str());
   EXPECT_EQ(3U, rdNnlo->size());
   EXPECT_EQ(0U, (*rdNnlo)[0].size());
   EXPECT_EQ(1U, (*rdNnlo)[1].size());
   EXPECT_EQ(4U, (*rdNnlo)[2].size());
   EXPECT_EQ(1.0, (*rdNnlo)[1][0]);
   EXPECT_EQ(1.0, (*rdNnlo)[2][0]);
   EXPECT_EQ(2.0, (*rdNnlo)[2][1]);
   EXPECT_EQ(4.0, (*rdNnlo)[2][2]);
   EXPECT_EQ(8.0, (*rdNnlo)[2][3]);
   EXPECT_EQ(0.0, (*rdFourVec)[0]);
   EXPECT_EQ(1.0, (*rdFourVec)[1]);
   EXPECT_EQ(2.0, (*rdFourVec)[2]);
   EXPECT_EQ(3.0, (*rdFourVec)[3]);

   reader->LoadEntry(1);
   EXPECT_EQ(24.0, *rdPt);
   EXPECT_STREQ("", rdTag->c_str());
   EXPECT_TRUE(rdNnlo->empty());
   EXPECT_EQ(42.0, (*rdFourVec)[2]);

   reader->LoadEntry(2);
   EXPECT_EQ(12.0, *rdPt);
   EXPECT_STREQ("12345", rdTag->c_str());
   EXPECT_EQ(1U, rdNnlo->size());
   EXPECT_EQ(1U, (*rdNnlo)[0].size());
   EXPECT_EQ(42.0, (*rdNnlo)[0][0]);
   EXPECT_EQ(24.0, (*rdFourVec)[1]);
}

TEST(RNTuple, ClusterEntries)
{
   FileRaii fileGuard("test_ntuple_cluster_entries.root");
   auto model = RNTupleModel::Create();
   auto field = model->MakeField<float>({"pt", "transverse momentum"}, 42.0);

   {
      RNTupleWriteOptions opt;
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath(), opt);
      for (int i = 0; i < 100; i++) {
         ntuple->Fill();
         if (((i + 1) % 5) == 0)
            ntuple->CommitCluster();
      }
   }

   auto ntuple = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   // 100 entries / 5 entries per cluster
   EXPECT_EQ(20, ntuple->GetDescriptor().GetNClusters());
}

TEST(RNTuple, ClusterEntriesAuto)
{
   FileRaii fileGuard("test_ntuple_cluster_entries_auto.root");
   auto model = RNTupleModel::Create();
   auto field = model->MakeField<float>({"pt", "transverse momentum"}, 42.0);

   {
      RNTupleWriteOptions options;
      options.SetCompression(0);
      options.SetEnablePageChecksums(false);
      options.SetApproxZippedClusterSize(5 * sizeof(float));
      options.SetPageBufferBudget(1000 * 1000);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath(), options);
      for (int i = 0; i < 100; i++) {
         ntuple->Fill();
      }
   }

   auto ntuple = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   // 100 entries / 5 entries per cluster
   EXPECT_EQ(20, ntuple->GetDescriptor().GetNClusters());
}

TEST(RNTuple, ClusterEntriesAutoStatus)
{
   FileRaii fileGuard("test_ntuple_cluster_entries_auto_status.root");
   {
      auto model = RNTupleModel::CreateBare();
      auto field = model->MakeField<float>({"pt", "transverse momentum"}, 42.0);

      int FlushClusterCalled = 0;
      RNTupleFillStatus status;

      RNTupleWriteOptions options;
      options.SetCompression(0);
      options.SetEnablePageChecksums(false);
      options.SetApproxZippedClusterSize(5 * sizeof(float));
      options.SetPageBufferBudget(1000 * 1000);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath(), options);
      auto entry = ntuple->CreateEntry();
      for (int i = 0; i < 100; i++) {
         ntuple->FillNoFlush(*entry, status);
         if (status.ShouldFlushCluster()) {
            ntuple->FlushCluster();
            FlushClusterCalled++;
         }
      }
      EXPECT_EQ(20, FlushClusterCalled);
   }

   auto ntuple = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   // 100 entries / 5 entries per cluster
   EXPECT_EQ(20, ntuple->GetDescriptor().GetNClusters());
}

TEST(RNTuple, PageSize)
{
   FileRaii fileGuard("test_ntuple_elements_per_page.root");
   auto model = RNTupleModel::Create();
   auto field = model->MakeField<float>({"pt", "transverse momentum"}, 42.0);

   {
      RNTupleWriteOptions opt;
      opt.SetInitialNElementsPerPage(10);
      opt.SetMaxUnzippedPageSize(200);
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath(), opt);
      for (int i = 0; i < 1000; i++) {
         writer->Fill();
      }
   }

   auto ntuple = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   const auto &col0_pages = ntuple->GetDescriptor().GetClusterDescriptor(0).GetPageRange(0);
   // 1000 column elements / 50 elements per page
   EXPECT_EQ(20, col0_pages.fPageInfos.size());
}

TEST(RNTupleModel, EnforceValidFieldNames)
{
   auto model = RNTupleModel::Create();

   // MakeField
   try {
      auto field3 = model->MakeField<float>("", 42.0);
      FAIL() << "empty string as field name should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("name cannot be empty string"));
   }
   try {
      auto field3 = model->MakeField<float>("pt.pt", 42.0);
      FAIL() << "field name with periods should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("name 'pt.pt' cannot contain dot characters '.'"));
   }

   // Previous failures to create 'pt' should not block the name
   auto field = model->MakeField<float>("pt", 42.0);

   try {
      auto field2 = model->MakeField<float>("pt", 42.0);
      FAIL() << "repeated field names should throw";
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("field name 'pt' already exists"));
   }

   // AddField
   try {
      model->AddField(std::make_unique<RField<float>>("pt"));
      FAIL() << "repeated field names should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("field name 'pt' already exists"));
   }

   // MakeCollection
   try {
      auto otherModel = RNTupleModel::Create();
      auto collection = model->MakeCollection("pt", std::move(otherModel));
      FAIL() << "repeated field names should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("field name 'pt' already exists"));
   }

   // subfield names don't throw because full name differs (otherModel.pt)
   auto otherModel = RNTupleModel::Create();
   auto otherField = otherModel->MakeField<float>("pt", 42.0);
   auto collection = model->MakeCollection("otherModel", std::move(otherModel));
}

TEST(RNTupleModel, FieldDescriptions)
{
   FileRaii fileGuard("test_ntuple_field_descriptions.root");
   auto model = RNTupleModel::Create();

   auto pt = model->MakeField<float>({"pt", "transverse momentum"}, 42.0);

   auto charge = std::make_unique<RField<float>>(RField<float>("charge"));
   charge->SetDescription("electric charge");
   model->AddField(std::move(charge));

   {
      RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath());
   }

   auto ntuple = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   std::vector<std::string> fieldDescriptions;
   for (auto &f : ntuple->GetDescriptor().GetTopLevelFields()) {
      fieldDescriptions.push_back(f.GetFieldDescription());
   }
   ASSERT_EQ(2u, fieldDescriptions.size());
   EXPECT_EQ(std::string("transverse momentum"), fieldDescriptions[0]);
   EXPECT_EQ(std::string("electric charge"), fieldDescriptions[1]);
}

TEST(RNTupleModel, CollectionFieldDescriptions)
{
   FileRaii fileGuard("test_ntuple_collection_field_descriptions.root");
   {
      auto muon = RNTupleModel::Create();
      muon->SetDescription("muons after basic selection");

      auto model = RNTupleModel::Create();
      model->MakeCollection("Muon", std::move(muon));
      RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath());
   }

   auto ntuple = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   const auto &muon_desc = *ntuple->GetDescriptor().GetTopLevelFields().begin();
   EXPECT_EQ(std::string("muons after basic selection"), muon_desc.GetFieldDescription());
}

TEST(RNTupleModel, GetField)
{
   auto m = RNTupleModel::Create();
   m->MakeField<int>("x");
   m->MakeField<CustomStruct>("cs");
   m->Freeze();
   EXPECT_EQ(m->GetField("x").GetFieldName(), "x");
   EXPECT_EQ(m->GetField("x").GetTypeName(), "std::int32_t");
   EXPECT_EQ(m->GetField("cs.v1").GetFieldName(), "v1");
   EXPECT_EQ(m->GetField("cs.v1").GetTypeName(), "std::vector<float>");
   try {
      m->GetField("nonexistent");
      FAIL() << "invalid field name should throw";
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("invalid field"));
   }
   try {
      m->GetField("");
      FAIL() << "empty field name should throw";
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("invalid field"));
   }
   EXPECT_EQ("", m->GetFieldZero().GetQualifiedFieldName());
   EXPECT_EQ("x", m->GetField("x").GetQualifiedFieldName());
   EXPECT_EQ("cs", m->GetField("cs").GetQualifiedFieldName());
   EXPECT_EQ("cs.v1", m->GetField("cs.v1").GetQualifiedFieldName());
}

TEST(RNTuple, EmptyString)
{
   // empty storage string
   try {
      auto model = RNTupleModel::Create();
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "myNTuple", "");
      FAIL() << "empty writer storage location should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("empty storage location"));
   }
   try {
      auto ntuple = RNTupleReader::Open("myNTuple", "");
      FAIL() << "empty reader storage location should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("empty storage location"));
   }

   // empty RNTuple name
   try {
      auto model = RNTupleModel::Create();
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "", "file.root");
      FAIL() << "empty RNTuple name should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("empty RNTuple name"));
   }
   try {
      auto ntuple = RNTupleReader::Open("", "file.root");
      FAIL() << "empty RNTuple name should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("empty RNTuple name"));
   }
}

TEST(RNTuple, NullSafety)
{
   // RNTupleModel
   auto model = RNTupleModel::Create();
   try {
      auto bad = model->MakeCollection("bad", nullptr);
      FAIL() << "null submodels should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("null collectionModel"));
   }
   try {
      model->AddField(nullptr);
      FAIL() << "null fields should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("null field"));
   }
}

TEST(RNTuple, ModelId)
{
   auto m1 = RNTupleModel::Create();
   auto m2 = RNTupleModel::Create();
   EXPECT_FALSE(m1->IsFrozen());
   EXPECT_NE(m1->GetModelId(), m2->GetModelId());

   m1->Freeze();
   EXPECT_TRUE(m1->IsFrozen());

   try {
      m1->SetDescription("abc");
      FAIL() << "changing frozen model should throw";
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("invalid attempt to modify frozen model"));
   }
   try {
      m1->MakeField<float>("pt");
      FAIL() << "changing frozen model should throw";
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("invalid attempt to modify frozen model"));
   }

   // Freeze() should be idempotent call
   auto id = m1->GetModelId();
   m1->Freeze();
   EXPECT_TRUE(m1->IsFrozen());
   EXPECT_EQ(id, m1->GetModelId());

   auto m2c = m2->Clone();
   EXPECT_NE(m2->GetModelId(), m2c->GetModelId());
}

TEST(RNTuple, Entry)
{
   auto m1 = RNTupleModel::Create();
   try {
      m1->CreateEntry();
      FAIL() << "creating entry of unfrozen model should throw";
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("invalid attempt to create entry"));
   }
   m1->Freeze();
   auto e1 = m1->CreateEntry();

   auto m2 = RNTupleModel::Create();
   m2->Freeze();
   auto e2 = m2->CreateEntry();

   FileRaii fileGuard("test_ntuple_entry.root");
   auto ntuple = RNTupleWriter::Recreate(std::move(m1), "ntpl", fileGuard.GetPath());
   ntuple->Fill();
   ntuple->Fill(*e1);
   try {
      ntuple->Fill(*e2);
      FAIL() << "filling with wrong entry should throw";
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("mismatch between entry and model"));
   }
}

TEST(RNTuple, BareEntry)
{
   auto m = RNTupleModel::CreateBare();
   auto f = m->MakeField<float>("pt");
   EXPECT_TRUE(m->IsBare());
   EXPECT_FALSE(f);

   FileRaii fileGuard("test_ntuple_bare_entry.root");
   {
      auto ntuple = RNTupleWriter::Recreate(std::move(m), "ntpl", fileGuard.GetPath());
      const auto &model = ntuple->GetModel();
      try {
         model.GetDefaultEntry();
         FAIL() << "accessing default entry of bare model should throw";
      } catch (const RException &err) {
         EXPECT_THAT(err.what(), testing::HasSubstr("invalid attempt to use default entry of bare model"));
      }
      try {
         model.GetDefaultEntry().GetPtr<float>("pt");
         FAIL() << "accessing default entry of bare model should throw";
      } catch (const RException &err) {
         EXPECT_THAT(err.what(), testing::HasSubstr("invalid attempt to use default entry of bare model"));
      }

      auto e1 = model.CreateEntry();
      ASSERT_NE(nullptr, e1->GetPtr<float>("pt"));
      *(e1->GetPtr<float>("pt")) = 1.0;
      auto e2 = model.CreateBareEntry();
      EXPECT_EQ(nullptr, e2->GetPtr<float>("pt"));
      float pt = 2.0;
      e2->BindRawPtr("pt", &pt);

      ntuple->Fill(*e1);
      ntuple->Fill(*e2);
   }

   auto ntuple = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   ASSERT_EQ(2U, ntuple->GetNEntries());
   ntuple->LoadEntry(0);
   EXPECT_EQ(1.0, *ntuple->GetModel().GetDefaultEntry().GetPtr<float>("pt"));
   ntuple->LoadEntry(1);
   EXPECT_EQ(2.0, *ntuple->GetModel().GetDefaultEntry().GetPtr<float>("pt"));
}

namespace ROOT::Experimental::Internal {
struct RFieldCallbackInjector {
   template <typename FieldT>
   static void Inject(FieldT &field, ROOT::Experimental::RFieldBase::ReadCallback_t func)
   {
      field.AddReadCallback(func);
   }
};
} // namespace ROOT::Experimental::Internal
namespace {
unsigned gNCallReadCallback = 0;
}

TEST(RNTuple, ReadCallback)
{
   using RFieldCallbackInjector = ROOT::Experimental::Internal::RFieldCallbackInjector;

   FileRaii fileGuard("test_ntuple_readcb.ntuple");
   {
      auto model = RNTupleModel::Create();
      auto fieldI32 = model->MakeField<std::int32_t>("i32", 0);
      auto fieldKlass = model->MakeField<CustomStruct>("klass");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "f", fileGuard.GetPath());
      fieldKlass->a = 42.0;
      fieldKlass->s = "abc";
      ntuple->Fill();
      *fieldI32 = 1;
      fieldKlass->a = 24.0;
      ntuple->Fill();
   }

   auto model = RNTupleModel::Create();
   auto fieldI32 = std::make_unique<RField<std::int32_t>>("i32");
   auto fieldKlass = std::make_unique<RField<CustomStruct>>("klass");
   RFieldCallbackInjector::Inject(*fieldI32, [](void *target) {
      static std::int32_t expected = 0;
      EXPECT_EQ(*static_cast<std::int32_t *>(target), expected++);
      gNCallReadCallback++;
   });
   RFieldCallbackInjector::Inject(*fieldI32, [](void *) { gNCallReadCallback++; });
   RFieldCallbackInjector::Inject(*fieldKlass, [](void *target) {
      auto typedValue = static_cast<CustomStruct *>(target);
      typedValue->a = 1337.0; // should change the value on the default entry
      gNCallReadCallback++;
   });
   model->AddField(std::move(fieldI32));
   model->AddField(std::move(fieldKlass));

   auto ntuple = RNTupleReader::Open(std::move(model), "f", fileGuard.GetPath());
   auto rdKlass = ntuple->GetModel().GetDefaultEntry().GetPtr<CustomStruct>("klass");
   EXPECT_EQ(2U, ntuple->GetNEntries());
   ntuple->LoadEntry(0);
   EXPECT_EQ(1337.0, rdKlass->a);
   ntuple->LoadEntry(1);
   EXPECT_EQ(1337.0, rdKlass->a);
   EXPECT_EQ(6U, gNCallReadCallback);
}

TEST(RNTuple, FillBytesWritten)
{
   FileRaii fileGuard("test_ntuple_fillbytes.ntuple");

   // TODO(jalopezg): improve test coverage by adding other field types
   auto model = RNTupleModel::Create();
   auto fieldI32 = model->MakeField<std::int32_t>("i32");
   auto fieldStr = model->MakeField<std::string>("s");
   auto fieldBoolVec = model->MakeField<std::vector<bool>>("bool_vec");
   auto fieldFloatVec = model->MakeField<std::vector<float>>("float_vec");
   auto ntuple = RNTupleWriter::Recreate(std::move(model), "f", fileGuard.GetPath());
   *fieldI32 = 42;
   *fieldStr = "abc";
   // A 32bit integer + "abc" literal + one 64bit integer for each index column
   EXPECT_EQ(7U + (3 * sizeof(std::uint64_t)), ntuple->Fill());
   *fieldBoolVec = {true, false, true};
   *fieldFloatVec = {42.0f, 1.1f};
   // A 32bit integer + "abc" literal + one 64bit integer for each index column + 3 bools + 2 floats
   EXPECT_EQ(18U + (3 * sizeof(std::uint64_t)), ntuple->Fill());
}

TEST(RNTuple, FillBytesWrittenCollections)
{
   FileRaii fileGuard("test_ntuple_fillbytes_collections.root");

   auto eventModel = RNTupleModel::Create();
   auto fldPt = eventModel->MakeField<float>("pt", 0.0);

   auto hitModel = RNTupleModel::Create();
   auto fldHitX = hitModel->MakeField<float>("x", 0.0);
   auto fldHitY = hitModel->MakeField<float>("y", 0.0);

   auto trackModel = RNTupleModel::Create();
   auto fldTrackEnergy = trackModel->MakeField<float>("energy", 0.0);

   auto fldHits = trackModel->MakeCollection("hits", std::move(hitModel));
   auto fldTracks = eventModel->MakeCollection("tracks", std::move(trackModel));

   {
      RNTupleWriteOptions options;
      options.SetApproxZippedClusterSize(1024 * 1024);
      auto writer = RNTupleWriter::Recreate(std::move(eventModel), "myNTuple", fileGuard.GetPath(), options);

      for (unsigned i = 0; i < 30000; ++i) {
         for (unsigned t = 0; t < 3; ++t) {
            for (unsigned h = 0; h < 2; ++h) {
               *fldHitX = 4.0;
               *fldHitY = 8.0;
               EXPECT_EQ(4 + 4, fldHits->Fill());
            }
            *fldTrackEnergy = i * t;
            EXPECT_EQ(2 * (4 + 4) + 8 + 4, fldTracks->Fill());
         }
         *fldPt = float(i);
         EXPECT_EQ(3 * (2 * (4 + 4) + 8 + 4) + 8 + 4, writer->Fill());
      }
   }

   auto reader = RNTupleReader::Open("myNTuple", fileGuard.GetPath());
   EXPECT_EQ(reader->GetDescriptor().GetNClusters(), 2);
}

TEST(RNTuple, RValue)
{
   auto f1 = RFieldBase::Create("f1", "CustomStruct").Unwrap();
   auto f2 = RFieldBase::Create("f2", "std::vector<std::vector<std::variant<std::string, float>>>").Unwrap();
   auto f3 = RFieldBase::Create("f3", "std::variant<std::unique_ptr<CustomStruct>, ComplexStruct>>").Unwrap();
   auto v1 = f1->CreateValue();
   auto v2 = f2->CreateValue();
   auto v3 = f3->CreateValue();

   auto p = v3.GetPtr<void>();
   v3.EmplaceNew();
   EXPECT_NE(p.get(), v3.GetPtr<void>().get());

   // Destruct fields to check if the deleters work without the fields
   f1 = nullptr;
   f2 = nullptr;
   f3 = nullptr;

   // The deleters are called in the destructors of the values
}

TEST(REntry, Basics)
{
   auto model = RNTupleModel::Create();
   EXPECT_FALSE(model->IsBare());
   model->MakeField<float>("pt");
   model->Freeze();

   auto e = model->CreateEntry();
   EXPECT_EQ(e->GetModelId(), model->GetModelId());
   for (const auto &v : *e) {
      EXPECT_STREQ("pt", v.GetField().GetFieldName().c_str());
   }

   EXPECT_THROW(e->GetToken(""), ROOT::Experimental::RException);
   EXPECT_THROW(e->GetToken("eta"), ROOT::Experimental::RException);
   EXPECT_THROW(model->GetToken("eta"), ROOT::Experimental::RException);

   std::shared_ptr<float> ptrPt;
   e->BindValue("pt", ptrPt);
   EXPECT_EQ(ptrPt.get(), e->GetPtr<float>("pt").get());

   auto model2 = model->Clone();
   EXPECT_THROW(e->GetPtr<void>(model2->GetDefaultEntry().GetToken("pt")), ROOT::Experimental::RException);
   std::shared_ptr<double> ptrDouble;
   EXPECT_THROW(e->BindValue("pt", ptrDouble), ROOT::Experimental::RException);

   float pt;
   e->BindRawPtr("pt", &pt);
   EXPECT_EQ(&pt, e->GetPtr<void>("pt").get());

   e->EmplaceNewValue(model->GetToken("pt"));
   EXPECT_NE(&pt, e->GetPtr<void>("pt").get());
}

TEST(RFieldBase, CreateObject)
{
   auto ptrInt = RField<int>("name").CreateObject<int>();
   EXPECT_EQ(0, *ptrInt);

   void *rawPtr = RField<int>("name").CreateObject<void>().release();
   EXPECT_EQ(0, *static_cast<int *>(rawPtr));
   delete static_cast<int *>(rawPtr);

   {
      ROOT::TestSupport::CheckDiagsRAII diagRAII;
      diagRAII.requiredDiag(kWarning, "[ROOT.NTuple]", "possibly leaking", false /* matchFullMessage */);
      auto p = RField<int>("name").CreateObject<void>();
      delete static_cast<int *>(p.get()); // avoid release
   }

   EXPECT_THROW(RField<int>("name").CreateObject<float>(), RException);

   auto ptrClass = RField<LowPrecisionFloats>("name").CreateObject<LowPrecisionFloats>();
   EXPECT_DOUBLE_EQ(1.0, ptrClass->b);
}
