#include "ntuple_test.hxx"

#if __cplusplus >= 201703L
TEST(RNTuple, ReconstructModel)
{
   FileRaii fileGuard("test_ntuple_reconstruct.root");
   auto model = RNTupleModel::Create();
   auto fieldPt = model->MakeField<float>("pt", 42.0);
   auto fieldNnlo = model->MakeField<std::vector<std::vector<float>>>("nnlo");
   auto fieldKlass = model->MakeField<CustomStruct>("klass");
   auto fieldArray = model->MakeField<std::array<double, 2>>("array");
   auto fieldVariant = model->MakeField<std::variant<double, std::variant<std::string, double>>>("variant");
   {
      RPageSinkFile sink("myNTuple", fileGuard.GetPath(), RNTupleWriteOptions());
      sink.Create(*model.get());
      sink.CommitDataset();
      model = nullptr;
   }

   RPageSourceFile source("myNTuple", fileGuard.GetPath(), RNTupleReadOptions());
   source.Attach();

   auto modelReconstructed = source.GetDescriptor().GenerateModel();
   EXPECT_EQ(nullptr, modelReconstructed->GetDefaultEntry()->Get<float>("xyz"));
   auto vecPtr = modelReconstructed->GetDefaultEntry()->Get<std::vector<std::vector<float>>>("nnlo");
   EXPECT_TRUE(vecPtr != nullptr);
   // Don't crash
   vecPtr->push_back(std::vector<float>{1.0});
   auto array = modelReconstructed->GetDefaultEntry()->Get<std::array<double, 2>>("array");
   EXPECT_TRUE(array != nullptr);
   auto variant = modelReconstructed->GetDefaultEntry()->Get<
      std::variant<double, std::variant<std::string, double>>>("variant");
   EXPECT_TRUE(variant != nullptr);
}
#endif // __cplusplus >= 201703L

TEST(RNTuple, Multi)
{
   FileRaii fileGuard("test_ntuple_multi.root");
   auto file = TFile::Open(fileGuard.GetPath().c_str(), "RECREATE");
   {
      auto model = RNTupleModel::Create();
      auto fieldPt = model->MakeField<float>("pt", 42.0);
      RNTupleWriter ntuple(std::move(model), std::make_unique<RPageSinkFile>("first", *file, RNTupleWriteOptions()));
      ntuple.Fill();
   }
   {
      auto model = RNTupleModel::Create();
      auto fieldPt = model->MakeField<float>("E", 1.0);
      RNTupleWriter ntuple(std::move(model), std::make_unique<RPageSinkFile>("second", *file, RNTupleWriteOptions()));
      ntuple.Fill();
   }
   file->Close();
   delete file;

   RNTupleReader ntupleFirst(std::make_unique<RPageSourceFile>("first", fileGuard.GetPath(), RNTupleReadOptions()));
   auto viewPt = ntupleFirst.GetView<float>("pt");
   int n = 0;
   for (auto i : ntupleFirst.GetEntryRange()) {
      EXPECT_EQ(42.0, viewPt(i));
      n++;
   }
   EXPECT_EQ(1, n);

   RNTupleReader ntupleSecond(std::make_unique<RPageSourceFile>("second", fileGuard.GetPath(), RNTupleReadOptions()));
   auto viewE = ntupleSecond.GetView<float>("E");
   n = 0;
   for (auto i : ntupleSecond.GetEntryRange()) {
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

   auto modelRead = std::unique_ptr<RNTupleModel>(modelWrite->Clone());

   {
      RNTupleWriter ntuple(std::move(modelWrite),
         std::make_unique<RPageSinkFile>("myNTuple", fileGuard.GetPath(), RNTupleWriteOptions()));
      ntuple.Fill();
   }

   auto rdSignal = modelRead->Get<bool>("signal");
   auto rdPt = modelRead->Get<float>("pt");
   auto rdEnergy = modelRead->Get<float>("energy");
   auto rdTag = modelRead->Get<std::string>("tag");
   auto rdJets = modelRead->Get<std::vector<float>>("jets");
   auto rdNnlo = modelRead->Get<std::vector<std::vector<float>>>("nnlo");
   auto rdKlass = modelRead->Get<CustomStruct>("klass");

   RNTupleReader ntuple(std::move(modelRead),
      std::make_unique<RPageSourceFile>("myNTuple", fileGuard.GetPath(), RNTupleReadOptions()));
   EXPECT_EQ(1U, ntuple.GetNEntries());
   ntuple.LoadEntry(0);

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

   auto modelRead = std::unique_ptr<RNTupleModel>(modelWrite->Clone());

   {
      RNTupleWriter ntuple(std::move(modelWrite),
         std::make_unique<RPageSinkFile>("myNTuple", fileGuard.GetPath(), RNTupleWriteOptions()));
      ntuple.Fill();
      ntuple.CommitCluster();
      *wrPt = 24.0;
      wrNnlo->clear();
      *wrTag = "";
      wrFourVec->at(2) = 42.0;
      ntuple.Fill();
      *wrPt = 12.0;
      wrNnlo->push_back(std::vector<float>{42.0});
      *wrTag = "12345";
      wrFourVec->at(1) = 24.0;
      ntuple.Fill();
   }

   auto rdPt = modelRead->Get<float>("pt");
   auto rdTag = modelRead->Get<std::string>("tag");
   auto rdNnlo = modelRead->Get<std::vector<std::vector<float>>>("nnlo");
   auto rdFourVec = modelRead->Get<std::array<float, 4>>("fourVec");

   RNTupleReader ntuple(std::move(modelRead),
      std::make_unique<RPageSourceFile>("myNTuple", fileGuard.GetPath(), RNTupleReadOptions()));
   EXPECT_EQ(3U, ntuple.GetNEntries());

   ntuple.LoadEntry(0);
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

   ntuple.LoadEntry(1);
   EXPECT_EQ(24.0, *rdPt);
   EXPECT_STREQ("", rdTag->c_str());
   EXPECT_TRUE(rdNnlo->empty());
   EXPECT_EQ(42.0, (*rdFourVec)[2]);

   ntuple.LoadEntry(2);
   EXPECT_EQ(12.0, *rdPt);
   EXPECT_STREQ("12345", rdTag->c_str());
   EXPECT_EQ(1U, rdNnlo->size());
   EXPECT_EQ(1U, (*rdNnlo)[0].size());
   EXPECT_EQ(42.0, (*rdNnlo)[0][0]);
   EXPECT_EQ(24.0, (*rdFourVec)[1]);
}

TEST(RNTuple, TypeName) {
   EXPECT_STREQ("float", ROOT::Experimental::RField<float>::TypeName().c_str());
   EXPECT_STREQ("std::vector<std::string>",
                ROOT::Experimental::RField<std::vector<std::string>>::TypeName().c_str());
   EXPECT_STREQ("CustomStruct",
                ROOT::Experimental::RField<CustomStruct>::TypeName().c_str());
}

namespace {
class RNoDictionary {};
} // namespace

TEST(RNTuple, TClass) {
   auto modelFail = RNTupleModel::Create();
   EXPECT_THROW(modelFail->MakeField<RNoDictionary>("nodict"), std::runtime_error);

   auto model = RNTupleModel::Create();
   auto ptrKlass = model->MakeField<CustomStruct>("klass");

   FileRaii fileGuard("test_ntuple_tclass.root");
   auto ntuple = RNTupleWriter::Recreate(std::move(model), "f", fileGuard.GetPath());
}


TEST(RNTuple, RealWorld1)
{
   FileRaii fileGuard("test_ntuple_realworld1.root");

   // See https://github.com/olifre/root-io-bench/blob/master/benchmark.cpp
   auto modelWrite = RNTupleModel::Create();
   auto wrEvent   = modelWrite->MakeField<std::uint32_t>("event");
   auto wrSignal  = modelWrite->MakeField<bool>("signal");
   auto wrEnergy  = modelWrite->MakeField<double>("energy");
   auto wrTimes   = modelWrite->MakeField<std::vector<double>>("times");
   auto wrIndices = modelWrite->MakeField<std::vector<std::uint32_t>>("indices");

   TRandom3 rnd(42);
   double chksumWrite = 0.0;
   {
      auto ntuple = RNTupleWriter::Recreate(std::move(modelWrite), "myNTuple", fileGuard.GetPath());
      constexpr unsigned int nEvents = 60000;
      for (unsigned int i = 0; i < nEvents; ++i) {
         *wrEvent = i;
         *wrEnergy = rnd.Rndm() * 1000.;
         *wrSignal = i % 2;

         chksumWrite += double(*wrEvent);
         chksumWrite += double(*wrSignal);
         chksumWrite += *wrEnergy;

         auto nTimes = 1 + floor(rnd.Rndm() * 1000.);
         wrTimes->resize(nTimes);
         for (unsigned int n = 0; n < nTimes; ++n) {
            wrTimes->at(n) = 1 + rnd.Rndm()*1000. - 500.;
            chksumWrite += wrTimes->at(n);
         }

         auto nIndices = 1 + floor(rnd.Rndm() * 1000.);
         wrIndices->resize(nIndices);
         for (unsigned int n = 0; n < nIndices; ++n) {
            wrIndices->at(n) = 1 + floor(rnd.Rndm() * 1000.);
            chksumWrite += double(wrIndices->at(n));
         }

         ntuple->Fill();
      }
   }

   auto modelRead  = RNTupleModel::Create();
   auto rdEvent   = modelRead->MakeField<std::uint32_t>("event");
   auto rdSignal  = modelRead->MakeField<bool>("signal");
   auto rdEnergy  = modelRead->MakeField<double>("energy");
   auto rdTimes   = modelRead->MakeField<std::vector<double>>("times");
   auto rdIndices = modelRead->MakeField<std::vector<std::uint32_t>>("indices");

   double chksumRead = 0.0;
   auto ntuple = RNTupleReader::Open(std::move(modelRead), "myNTuple", fileGuard.GetPath());
   for (auto entryId : *ntuple) {
      ntuple->LoadEntry(entryId);
      chksumRead += double(*rdEvent) + double(*rdSignal) + *rdEnergy;
      for (auto t : *rdTimes) chksumRead += t;
      for (auto ind : *rdIndices) chksumRead += double(ind);
   }

   // The floating point arithmetic should have been executed in the same order for reading and writing,
   // thus we expect the checksums to be bitwise identical
   EXPECT_EQ(chksumRead, chksumWrite);
}


TEST(RNTuple, Descriptor)
{
   RNTupleDescriptorBuilder descBuilder;
   descBuilder.SetNTuple("MyTuple", "Description", "Me", RNTupleVersion(1, 2, 3), ROOT::Experimental::RNTupleUuid());
   descBuilder.AddField(0, RNTupleVersion(), RNTupleVersion(), "", "", 0, ENTupleStructure::kRecord);
   descBuilder.AddField(1, RNTupleVersion(), RNTupleVersion(), "list", "std::vector<std::int32_t>",
                        0, ENTupleStructure::kCollection);
   descBuilder.AddFieldLink(0, 1);
   descBuilder.AddField(2, RNTupleVersion(), RNTupleVersion(), "list", "std::int32_t", 0, ENTupleStructure::kLeaf);
   descBuilder.AddFieldLink(1, 2);
   descBuilder.AddField(42, RNTupleVersion(), RNTupleVersion(), "x", "std::string", 0, ENTupleStructure::kLeaf);
   descBuilder.AddFieldLink(0, 42);
   descBuilder.AddColumn(3, 42, RNTupleVersion(), RColumnModel(EColumnType::kIndex, true), 0);
   descBuilder.AddColumn(4, 42, RNTupleVersion(), RColumnModel(EColumnType::kByte, true), 1);

   ROOT::Experimental::RClusterDescriptor::RColumnRange columnRange;
   ROOT::Experimental::RClusterDescriptor::RPageRange::RPageInfo pageInfo;
   // Description of cluster #0
   descBuilder.AddCluster(0, RNTupleVersion(), 0, ROOT::Experimental::ClusterSize_t(100));
   columnRange.fColumnId = 3;
   columnRange.fFirstElementIndex = 0;
   columnRange.fNElements = 100;
   descBuilder.AddClusterColumnRange(0, columnRange);
   ROOT::Experimental::RClusterDescriptor::RPageRange pageRange0;
   pageRange0.fPageInfos.clear();
   pageRange0.fColumnId = 3;
   pageInfo.fNElements = 40;
   pageInfo.fLocator.fPosition = 0;
   pageRange0.fPageInfos.emplace_back(pageInfo);
   pageInfo.fNElements = 60;
   pageInfo.fLocator.fPosition = 1024;
   pageRange0.fPageInfos.emplace_back(pageInfo);
   descBuilder.AddClusterPageRange(0, std::move(pageRange0));

   columnRange.fColumnId = 4;
   columnRange.fFirstElementIndex = 0;
   columnRange.fNElements = 300;
   descBuilder.AddClusterColumnRange(0, columnRange);
   ROOT::Experimental::RClusterDescriptor::RPageRange pageRange1;
   pageRange1.fColumnId = 4;
   pageInfo.fNElements = 200;
   pageInfo.fLocator.fPosition = 2048;
   pageRange1.fPageInfos.emplace_back(pageInfo);
   pageInfo.fNElements = 100;
   pageInfo.fLocator.fPosition = 4096;
   pageRange1.fPageInfos.emplace_back(pageInfo);
   descBuilder.AddClusterPageRange(0, std::move(pageRange1));

   // Description of cluster #1
   descBuilder.AddCluster(1, RNTupleVersion(), 100, ROOT::Experimental::ClusterSize_t(1000));
   columnRange.fColumnId = 3;
   columnRange.fFirstElementIndex = 100;
   columnRange.fNElements = 1000;
   descBuilder.AddClusterColumnRange(1, columnRange);
   ROOT::Experimental::RClusterDescriptor::RPageRange pageRange2;
   pageRange2.fColumnId = 3;
   pageInfo.fNElements = 1000;
   pageInfo.fLocator.fPosition = 8192;
   pageRange2.fPageInfos.emplace_back(pageInfo);
   descBuilder.AddClusterPageRange(1, std::move(pageRange2));

   columnRange.fColumnId = 4;
   columnRange.fFirstElementIndex = 300;
   columnRange.fNElements = 3000;
   descBuilder.AddClusterColumnRange(1, columnRange);
   ROOT::Experimental::RClusterDescriptor::RPageRange pageRange3;
   pageRange3.fColumnId = 4;
   pageInfo.fNElements = 3000;
   pageInfo.fLocator.fPosition = 16384;
   pageRange3.fPageInfos.emplace_back(pageInfo);
   descBuilder.AddClusterPageRange(1, std::move(pageRange3));

   const auto &reference = descBuilder.GetDescriptor();
   EXPECT_EQ("MyTuple", reference.GetName());
   EXPECT_EQ(1U, reference.GetVersion().GetVersionUse());
   EXPECT_EQ(2U, reference.GetVersion().GetVersionMin());
   EXPECT_EQ(3U, reference.GetVersion().GetFlags());

   auto szHeader = reference.SerializeHeader(nullptr);
   auto headerBuffer = new unsigned char[szHeader];
   reference.SerializeHeader(headerBuffer);
   auto szFooter = reference.SerializeFooter(nullptr);
   auto footerBuffer = new unsigned char[szFooter];
   reference.SerializeFooter(footerBuffer);

   const auto nbytesPostscript = RNTupleDescriptor::kNBytesPostscript;
   ASSERT_GE(szFooter, nbytesPostscript);
   std::uint32_t szPsHeader;
   std::uint32_t szPsFooter;
   RNTupleDescriptor::LocateMetadata(footerBuffer + szFooter - nbytesPostscript, szPsHeader, szPsFooter);
   EXPECT_EQ(szHeader, szPsHeader);
   EXPECT_EQ(szFooter, szPsFooter);

   RNTupleDescriptorBuilder reco;
   reco.SetFromHeader(headerBuffer);
   reco.AddClustersFromFooter(footerBuffer);
   EXPECT_EQ(reference, reco.GetDescriptor());

   EXPECT_EQ(NTupleSize_t(1100), reference.GetNEntries());
   EXPECT_EQ(NTupleSize_t(1100), reference.GetNElements(3));
   EXPECT_EQ(NTupleSize_t(3300), reference.GetNElements(4));

   EXPECT_EQ(DescriptorId_t(0), reference.FindFieldId("", ROOT::Experimental::kInvalidDescriptorId));
   EXPECT_EQ(DescriptorId_t(1), reference.FindFieldId("list", 0));
   EXPECT_EQ(DescriptorId_t(1), reference.FindFieldId("list"));
   EXPECT_EQ(DescriptorId_t(2), reference.FindFieldId("list", 1));
   EXPECT_EQ(DescriptorId_t(42), reference.FindFieldId("x", 0));
   EXPECT_EQ(DescriptorId_t(42), reference.FindFieldId("x"));
   EXPECT_EQ(ROOT::Experimental::kInvalidDescriptorId, reference.FindFieldId("listX", 1));
   EXPECT_EQ(ROOT::Experimental::kInvalidDescriptorId, reference.FindFieldId("list", 1024));

   EXPECT_EQ(DescriptorId_t(3), reference.FindColumnId(42, 0));
   EXPECT_EQ(DescriptorId_t(4), reference.FindColumnId(42, 1));
   EXPECT_EQ(ROOT::Experimental::kInvalidDescriptorId, reference.FindColumnId(42, 2));
   EXPECT_EQ(ROOT::Experimental::kInvalidDescriptorId, reference.FindColumnId(43, 0));

   EXPECT_EQ(DescriptorId_t(0), reference.FindClusterId(3, 0));
   EXPECT_EQ(DescriptorId_t(1), reference.FindClusterId(3, 100));
   EXPECT_EQ(ROOT::Experimental::kInvalidDescriptorId, reference.FindClusterId(3, 40000));

   delete[] footerBuffer;
   delete[] headerBuffer;
}

#if !defined(_MSC_VER) || defined(R__ENABLE_BROKEN_WIN_TESTS)
TEST(RNTuple, LargeFile)
{
   FileRaii fileGuard("test_large_file.root");

   auto modelWrite = RNTupleModel::Create();
   auto& wrEnergy  = *modelWrite->MakeField<double>("energy");

   TRandom3 rnd(42);
   double chksumWrite = 0.0;
   {
      RNTupleWriteOptions options;
      options.SetCompression(0);
      auto ntuple = RNTupleWriter::Recreate(std::move(modelWrite), "myNTuple", fileGuard.GetPath(), options);
      constexpr unsigned long nEvents = 1024 * 1024 * 256; // Exceed 2GB file size
      for (unsigned int i = 0; i < nEvents; ++i) {
         wrEnergy = rnd.Rndm();
         chksumWrite += wrEnergy;
         ntuple->Fill();
      }
   }
   FILE *file = fopen(fileGuard.GetPath().c_str(), "rb");
   ASSERT_TRUE(file != nullptr);
   EXPECT_EQ(0, fseek(file, 0, SEEK_END));
   EXPECT_GT(ftell(file), 2048LL * 1024LL * 1024LL);
   fclose(file);

   auto ntuple = RNTupleReader::Open("myNTuple", fileGuard.GetPath());
   auto rdEnergy  = ntuple->GetView<double>("energy");
   double chksumRead = 0.0;

   for (auto i : ntuple->GetEntryRange()) {
      chksumRead += rdEnergy(i);
   }

   EXPECT_EQ(chksumRead, chksumWrite);
   auto f = TFile::Open(fileGuard.GetPath().c_str(), "READ");
   EXPECT_TRUE(f != nullptr);
   delete f;
}
#endif
