#include "ntuple_test.hxx"

#include <TMemFile.h>

#include <limits>

TEST(RNTuple, View)
{
   FileRaii fileGuard("test_ntuple_view.root");

   auto model = RNTupleModel::Create();
   *model->MakeField<float>("pt") = 42.0;
   *model->MakeField<std::string>("tag") = "xyz";
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

TEST(RNTuple, CollectionView)
{
   FileRaii fileGuard("test_ntuple_collection_view.root");

   {
      auto model = RNTupleModel::Create();
      auto fieldJets = model->MakeField<std::vector<std::int32_t>>("jets");
      *fieldJets = {1, 2, 3};

      auto writer = RNTupleWriter::Recreate(std::move(model), "myNTuple", fileGuard.GetPath());
      writer->Fill();
      *fieldJets = {4, 5};
      writer->Fill();
      writer->CommitCluster();
      fieldJets->clear();
      writer->Fill();
      *fieldJets = {6, 7, 8, 9};
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("myNTuple", fileGuard.GetPath());
   ASSERT_EQ(4, reader->GetNEntries());
   auto viewJets = reader->GetCollectionView("jets");
   auto viewJetsItems = viewJets.GetView<std::int32_t>("_0");

   // The call operator returns the size of the collection.
   EXPECT_EQ(3, viewJets(0));
   EXPECT_EQ(2, viewJets(1));
   EXPECT_EQ(0, viewJets(2));
   EXPECT_EQ(4, viewJets(3));
   EXPECT_EQ(4, viewJets(RClusterIndex(1, 1)));

   // Via the collection range, we can get the items.
   auto range = viewJets.GetCollectionRange(1);
   EXPECT_EQ(2, range.size());
   EXPECT_EQ(RClusterIndex(0, 3), *range.begin());
   EXPECT_EQ(RClusterIndex(0, 5), *range.end());

   std::int32_t expected = 4;
   for (auto &&index : range) {
      EXPECT_EQ(expected, viewJetsItems(index));
      expected++;
   }

   // The same items can be bulk-read.
   auto bulk = viewJetsItems.CreateBulk();
   std::int32_t *values = static_cast<std::int32_t *>(bulk.ReadBulk(range));
   for (std::size_t i = 0; i < range.size(); i++) {
      EXPECT_EQ(i + 4, values[i]);
   }

   // The pointer can be adopted by an RVec.
   ROOT::RVec<std::int32_t> v(values, range.size());
   ROOT::RVec<std::int32_t> expectedV = {4, 5};
   EXPECT_TRUE(ROOT::VecOps::All(expectedV == v));

   // Bulk reading can also adopt a provided buffer.
   auto buffer = std::make_unique<std::int32_t[]>(range.size());
   bulk.AdoptBuffer(buffer.get(), range.size());
   bulk.ReadBulk(range);
   for (std::size_t i = 0; i < range.size(); i++) {
      EXPECT_EQ(i + 4, buffer[i]);
   }
}

TEST(RNTuple, ViewCast)
{
   FileRaii fileGuard("test_ntuple_view_cast.root");

   auto model = RNTupleModel::Create();
   auto fieldF = model->MakeField<float>("f");
   auto fieldD = model->MakeField<double>("d");
   auto field32 = model->MakeField<std::uint32_t>("u32");
   auto field64 = model->MakeField<std::int64_t>("i64");

   {
      auto writer = RNTupleWriter::Recreate(std::move(model), "myNTuple", fileGuard.GetPath());
      *fieldF = 1.0;
      *fieldD = 2.0;
      *field32 = 32;
      *field64 = -64;
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
   auto fieldPt = model->MakeField<float>("pt");
   auto fieldVec = model->MakeField<std::vector<float>>("vec");
   {
      RNTupleWriteOptions opt;
      auto writer = RNTupleWriter::Recreate(std::move(model), "myNTuple", fileGuard.GetPath(), opt);
      *fieldPt = 42.0;
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
   *model->MakeField<float>("pt") = 42.0;

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
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("no field named 'pT' in RNTuple 'myNTuple'"));
   }
   try {
      auto badView = ntuple->GetCollectionView("Moun");
      FAIL() << "missing field names should throw";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("no field named 'Moun' in RNTuple 'myNTuple'"));
   }
   try {
      auto badView = viewMuon.GetView<float>("badField");
      FAIL() << "missing field names should throw";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("no field named 'badField' in collection 'Muon'"));
   }
   try {
      auto badView = viewMuon.GetCollectionView("badC");
      FAIL() << "missing field names should throw";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("no field named 'badC' in collection 'Muon'"));
   }
}

TEST(RNTuple, ViewWithExternalAddress)
{
   FileRaii fileGuard("test_ntuple_viewexternal.root");

   auto model = RNTupleModel::Create();
   *model->MakeField<float>("pt") = 42.0;

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
      *model->MakeField<char>("c") = 'a';
      *model->MakeField<unsigned char>("uc") = 1;
      *model->MakeField<short>("s") = 2;
      *model->MakeField<unsigned short>("us") = 3;
      *model->MakeField<int>("i") = 4;
      *model->MakeField<unsigned int>("ui") = 5;
      *model->MakeField<long>("l") = 6;
      *model->MakeField<unsigned long>("ul") = 7;
      *model->MakeField<long long>("ll") = 8;
      *model->MakeField<unsigned long long>("ull") = 9;

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

   auto ntpl = std::unique_ptr<ROOT::RNTuple>(file.Get<ROOT::RNTuple>("ntpl"));
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

TEST(RNTuple, ViewOutOfBounds)
{
   FileRaii fileGuard("test_ntuple_view_oob.root");

   auto model = RNTupleModel::Create();
   auto foo = model->MakeField<std::int32_t>("foo");

   {
      auto writer = RNTupleWriter::Recreate(std::move(model), "myNTuple", fileGuard.GetPath());
      *foo = 1;
      writer->Fill();
      *foo = 2;
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("myNTuple", fileGuard.GetPath());
   try {
      auto viewJets = reader->GetView<std::int32_t>("foo");
      viewJets(3);
      FAIL() << "accessing an out-of-bounds entry with a view should throw";
   } catch (const ROOT::RException &ex) {
      EXPECT_THAT(ex.what(), testing::HasSubstr("out of bounds"));
   }
}

TEST(RNTuple, ViewFieldIteration)
{
   FileRaii fileGuard("test_ntuple_viewfielditeration.root");

   {
      auto model = RNTupleModel::Create();
      model->MakeField<float>("pt");
      model->MakeField<std::vector<float>>("vec");
      model->MakeField<std::atomic<int>>("atomic");
      model->MakeField<CustomEnum>("enum");
      model->MakeField<std::array<CustomEnum, 2>>("array");
      model->MakeField<CustomStruct>("struct");
      model->MakeField<EmptyStruct>("empty");

      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());

   auto viewPt = reader->GetView<void>("pt");
   EXPECT_EQ(1u, viewPt.GetFieldRange().size());
   auto viewVec = reader->GetView<void>("vec");
   EXPECT_EQ(1u, viewVec.GetFieldRange().size());
   auto viewAtomic = reader->GetView<void>("atomic");
   EXPECT_EQ(1u, viewAtomic.GetFieldRange().size());
   auto viewEnum = reader->GetView<void>("enum");
   EXPECT_EQ(1u, viewEnum.GetFieldRange().size());
   auto viewStruct = reader->GetView<void>("struct");
   EXPECT_EQ(1u, viewStruct.GetFieldRange().size());
   auto viewArray = reader->GetView<void>("array");
   EXPECT_EQ(1u, viewArray.GetFieldRange().size());

   auto viewEmpty = reader->GetView<void>("empty");
   try {
      viewEmpty.GetFieldRange();
      FAIL() << "accessing the field range of a view on an empty field should throw";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("field iteration over empty fields is unsupported"));
   }
}
