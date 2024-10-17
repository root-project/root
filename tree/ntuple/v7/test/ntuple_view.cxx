#include "ntuple_test.hxx"

#include <TMemFile.h>

#include <limits>

TEST(RNTuple, View)
{
   FileRaii fileGuard("test_ntuple_view.root");

   auto model = RNTupleModel::Create();
   auto fieldPt = model->MakeField<float>("pt", 42.0);
   auto fieldTag = model->MakeField<std::string>("tag", "xyz");
   auto fieldJets = model->MakeField<std::vector<std::int32_t>>("jets");
   fieldJets->push_back(1);
   fieldJets->push_back(2);
   fieldJets->push_back(3);

   {
      auto writer = RNTupleWriter::Recreate(std::move(model), "myNTuple", fileGuard.GetPath());
      writer->Fill();
      writer->CommitCluster();
      fieldJets->clear();
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("myNTuple", fileGuard.GetPath());
   auto viewPt = reader->GetView<float>("pt");
   int n = 0;
   for (auto i : reader->GetEntryRange()) {
      EXPECT_EQ(42.0, viewPt(i));
      n++;
   }
   EXPECT_EQ(2, n);

   auto viewJets = reader->GetView<std::vector<std::int32_t>>("jets");
   n = 0;
   for (auto i : reader->GetEntryRange()) {
      if (i == 0) {
         EXPECT_EQ(3U, viewJets(i).size());
         EXPECT_EQ(1, viewJets(i)[0]);
         EXPECT_EQ(2, viewJets(i)[1]);
         EXPECT_EQ(3, viewJets(i)[2]);
      } else {
         EXPECT_EQ(0U, viewJets(i).size());
      }
      n++;
   }
   EXPECT_EQ(2, n);

   auto viewJetElements = reader->GetView<std::int32_t>("jets._0");
   n = 0;
   for (auto i : viewJetElements.GetFieldRange()) {
      n++;
      EXPECT_EQ(n, viewJetElements(i));
   }
   EXPECT_EQ(3, n);
}

TEST(RNTuple, ViewCast)
{
   FileRaii fileGuard("test_ntuple_view_cast.root");

   auto model = RNTupleModel::Create();
   auto fieldF = model->MakeField<float>("f", 1.0);
   auto fieldD = model->MakeField<double>("d", 2.0);
   auto field32 = model->MakeField<std::uint32_t>("u32", 32);
   auto field64 = model->MakeField<std::int64_t>("i64", -64);

   {
      auto writer = RNTupleWriter::Recreate(std::move(model), "myNTuple", fileGuard.GetPath());
      writer->Fill();
      *fieldF = -42.0;
      *fieldD = -63.0;
      *field32 = std::numeric_limits<std::uint32_t>::max();
      *field64 = std::numeric_limits<std::int32_t>::min();
      writer->Fill();
   }

   // Test views that cast to a different type.
   auto reader = RNTupleReader::Open("myNTuple", fileGuard.GetPath());
   auto viewF = reader->GetView<double>("f");
   auto viewD = reader->GetView<float>("d");
   auto view32 = reader->GetView<std::uint64_t>("u32");
   auto view64 = reader->GetView<std::int32_t>("i64");

   EXPECT_FLOAT_EQ(1.0, viewF(0));
   EXPECT_FLOAT_EQ(2.0, viewD(0));
   EXPECT_EQ(32, view32(0));
   EXPECT_EQ(-64, view64(0));

   EXPECT_FLOAT_EQ(-42.0, viewF(1));
   EXPECT_FLOAT_EQ(-63.0, viewD(1));
   EXPECT_EQ(std::numeric_limits<std::uint32_t>::max(), view32(1));
   EXPECT_EQ(std::numeric_limits<std::int32_t>::min(), view64(1));
}

TEST(RNTuple, DirectAccessView)
{
   FileRaii fileGuard("test_ntuple_direct_access_view.root");

   auto model = RNTupleModel::Create();
   auto fieldPt = model->MakeField<float>("pt", 42.0);
   auto fieldVec = model->MakeField<std::vector<float>>("vec");
   {
      RNTupleWriteOptions opt;
      auto writer = RNTupleWriter::Recreate(std::move(model), "myNTuple", fileGuard.GetPath(), opt);
      writer->Fill();
      writer->CommitCluster();
      *fieldPt = 137.0;
      fieldVec->emplace_back(1.0);
      fieldVec->emplace_back(2.0);
      writer->Fill();
   }
   auto reader = RNTupleReader::Open("myNTuple", fileGuard.GetPath());
   auto viewPt = reader->GetDirectAccessView<float>("pt");
   auto viewVec = reader->GetCollectionView("vec");
   auto viewVecInner = viewVec.GetDirectAccessView<float>("_0");

   EXPECT_FLOAT_EQ(42.0, viewPt(0));
   EXPECT_FLOAT_EQ(137.0, viewPt(1));

   EXPECT_EQ(0u, viewVec(0));
   EXPECT_EQ(2u, viewVec(1));

   EXPECT_FLOAT_EQ(1.0, viewVecInner(0));
   EXPECT_FLOAT_EQ(2.0, viewVecInner(1));
}

TEST(RNTuple, VoidView)
{
   FileRaii fileGuard("test_ntuple_voidview.root");

   auto model = RNTupleModel::Create();
   auto fieldPt = model->MakeField<float>("pt", 42.0);

   {
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   EXPECT_EQ(1u, reader->GetNEntries());
   auto viewPt = reader->GetView<void>("pt");
   viewPt(0);
   EXPECT_FLOAT_EQ(42.0, viewPt.GetValue().GetRef<float>());
   EXPECT_STREQ("pt", viewPt.GetField().GetFieldName().c_str());
}

TEST(RNTuple, MissingViewNames)
{
   FileRaii fileGuard("test_ntuple_missing_view_names.root");
   auto model = RNTupleModel::Create();
   model->MakeField<float>("pt");
   model->MakeField<std::vector<float>>("Muon");
   {
      RNTupleWriter::Recreate(std::move(model), "myNTuple", fileGuard.GetPath());
   }
   auto ntuple = RNTupleReader::Open("myNTuple", fileGuard.GetPath());
   auto viewPt = ntuple->GetView<float>("pt");
   auto viewMuon = ntuple->GetCollectionView("Muon");
   try {
      auto badView = ntuple->GetView<float>("pT");
      FAIL() << "missing field names should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("no field named 'pT' in RNTuple 'myNTuple'"));
   }
   try {
      auto badView = ntuple->GetCollectionView("Moun");
      FAIL() << "missing field names should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("no field named 'Moun' in RNTuple 'myNTuple'"));
   }
   try {
      auto badView = viewMuon.GetView<float>("badField");
      FAIL() << "missing field names should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("no field named 'badField' in collection 'Muon'"));
   }
   try {
      auto badView = viewMuon.GetCollectionView("badC");
      FAIL() << "missing field names should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("no field named 'badC' in collection 'Muon'"));
   }
}

TEST(RNTuple, ViewWithExternalAddress)
{
   FileRaii fileGuard("test_ntuple_viewexternal.root");

   auto model = RNTupleModel::Create();
   auto fieldPt = model->MakeField<float>("pt", 42.0);

   {
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());

   // Typed shared_ptr
   auto data_1 = std::make_shared<float>();
   auto view_1 = reader->GetView("pt", data_1);
   view_1(0);
   EXPECT_FLOAT_EQ(42.0, *data_1);

   // Void shared_ptr
   std::shared_ptr<void> data_2{new float()};
   auto view_2 = reader->GetView("pt", data_2);
   view_2(0);
   EXPECT_FLOAT_EQ(42.0, *static_cast<float *>(data_2.get()));
}

TEST(RNTuple, BindEmplaceTyped)
{
   FileRaii fileGuard("test_ntuple_bindvalueemplacetyped.root");

   auto model = RNTupleModel::Create();
   auto fieldPt = model->MakeField<float>("pt");

   {
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      *fieldPt = 11.f;
      writer->Fill();
      *fieldPt = 22.f;
      writer->Fill();
      *fieldPt = 33.f;
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());

   // bind to shared_ptr
   auto value1 = std::make_shared<float>();
   auto view = reader->GetView<float>("pt", nullptr);
   view.Bind(value1);
   view(0);
   EXPECT_FLOAT_EQ(11.f, *value1);

   // bind to raw pointer
   float value2;
   view.BindRawPtr(&value2);
   view(1);
   EXPECT_FLOAT_EQ(22.f, value2);

   // emplace new value
   view.EmplaceNew();
   EXPECT_FLOAT_EQ(33.f, view(2));
   EXPECT_FLOAT_EQ(22.f, value2); // The previous value was not modified
}

TEST(RNTuple, BindEmplaceVoid)
{
   FileRaii fileGuard("test_ntuple_bindvalueemplacevoid.root");

   auto model = RNTupleModel::Create();
   auto fieldPt = model->MakeField<float>("pt");

   {
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      *fieldPt = 11.f;
      writer->Fill();
      *fieldPt = 22.f;
      writer->Fill();
      *fieldPt = 33.f;
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());

   // bind to shared_ptr
   std::shared_ptr<void> value1{new float()};
   auto view = reader->GetView<void>("pt", nullptr);
   view.Bind(value1);
   view(0);
   EXPECT_FLOAT_EQ(11.f, *reinterpret_cast<float *>(value1.get()));

   // bind to raw pointer
   float value2;
   view.BindRawPtr(&value2);
   view(1);
   EXPECT_FLOAT_EQ(22.f, value2);

   // emplace new value
   view.EmplaceNew();
   view(2);
   EXPECT_FLOAT_EQ(33.f, view.GetValue().GetRef<float>());
   EXPECT_FLOAT_EQ(22.f, value2); // The previous value was not modified
}

TEST(RNTuple, ViewStandardIntegerTypes)
{
   FileRaii fileGuard("test_ntuple_viewstandardintegertypes.root");

   {
      auto model = RNTupleModel::Create();
      auto c = model->MakeField<char>("c", 'a');
      auto uc = model->MakeField<unsigned char>("uc", 1);
      auto s = model->MakeField<short>("s", 2);
      auto us = model->MakeField<unsigned short>("us", 3);
      auto i = model->MakeField<int>("i", 4);
      auto ui = model->MakeField<unsigned int>("ui", 5);
      auto l = model->MakeField<long>("l", 6);
      auto ul = model->MakeField<unsigned long>("ul", 7);
      auto ll = model->MakeField<long long>("ll", 8);
      auto ull = model->MakeField<unsigned long long>("ull", 9);

      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   ASSERT_EQ(1, reader->GetNEntries());

   EXPECT_EQ('a', reader->GetView<char>("c")(0));
   EXPECT_EQ(1, reader->GetView<unsigned char>("uc")(0));
   EXPECT_EQ(2, reader->GetView<short>("s")(0));
   EXPECT_EQ(3, reader->GetView<unsigned short>("us")(0));
   EXPECT_EQ(4, reader->GetView<int>("i")(0));
   EXPECT_EQ(5, reader->GetView<unsigned int>("ui")(0));
   EXPECT_EQ(6, reader->GetView<long>("l")(0));
   EXPECT_EQ(7, reader->GetView<unsigned long>("ul")(0));
   EXPECT_EQ(8, reader->GetView<long long>("ll")(0));
   EXPECT_EQ(9, reader->GetView<unsigned long long>("ull")(0));
}

TEST(RNTuple, ViewFrameworkUse)
{
   TMemFile file("memfile.root", "RECREATE");

   {
      auto model = RNTupleModel::Create();
      auto ptrPx = model->MakeField<float>("px");
      auto ptrPy = model->MakeField<float>("py");
      // The trigger pages make a hole in the on-disk layout that is not (purposefully) read
      model->MakeField<bool>("trigger");
      auto ptrPz = model->MakeField<float>("pz");

      // Ensure that we use RTFileRawFile
      auto writer = RNTupleWriter::Append(std::move(model), "ntpl", file);

      for (int i = 0; i < 50; ++i) {
         for (int j = 0; j < 5; ++j) {
            *ptrPx = i * 5 + j;
            *ptrPy = 0.2 + i * 5 + j;
            *ptrPz = 0.4 + i * 5 + j;
            writer->Fill();
         }
         writer->CommitCluster();
      }
   }

   auto ntpl = std::unique_ptr<RNTuple>(file.Get<RNTuple>("ntpl"));
   auto reader = RNTupleReader::Open(*ntpl);
   reader->EnableMetrics();

   std::optional<ROOT::Experimental::RNTupleView<void>> viewPx;
   std::optional<ROOT::Experimental::RNTupleView<void>> viewPy;
   std::optional<ROOT::Experimental::RNTupleView<void>> viewPz;

   float px = 0, py = 0, pz = 0;
   for (auto i : reader->GetEntryRange()) {
      if (i > 1) {
         if (!viewPx) {
            viewPx = reader->GetView<void>("px", &px);
         }
         (*viewPx)(i);
         EXPECT_FLOAT_EQ(i, px);
      }

      if (i > 3) {
         if (!viewPy) {
            viewPy = reader->GetView<void>("py", &py);
         }
         (*viewPy)(i);
         EXPECT_FLOAT_EQ(0.2 + i, py);
      }

      if (i > 7) {
         if (!viewPz) {
            viewPz = reader->GetView<void>("pz", &pz);
         }
         (*viewPz)(i);
         EXPECT_FLOAT_EQ(0.4 + i, pz);
      }
   }

   // Ensure that cluster prefetching and smearing of read requests works
   // Note that "nClusterLoaded" is the number of _partial_ clusters preloaded from storage. Because we read
   // from the first cluster first px and then py, we'll call two times `LoadCluster()` on the first cluster.
   EXPECT_LT(reader->GetDescriptor().GetNClusters(),
             reader->GetMetrics().GetCounter("RNTupleReader.RPageSourceFile.nClusterLoaded")->GetValueAsInt());
   EXPECT_LT(reader->GetMetrics().GetCounter("RNTupleReader.RPageSourceFile.nReadV")->GetValueAsInt(),
             reader->GetMetrics().GetCounter("RNTupleReader.RPageSourceFile.nPageRead")->GetValueAsInt());
   EXPECT_LT(reader->GetMetrics().GetCounter("RNTupleReader.RPageSourceFile.nRead")->GetValueAsInt(),
             reader->GetMetrics().GetCounter("RNTupleReader.RPageSourceFile.nPageRead")->GetValueAsInt());
}
