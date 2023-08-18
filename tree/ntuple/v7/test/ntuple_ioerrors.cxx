#include "ntuple_test.hxx"
#include <TFile.h>

using RRawFile = ROOT::Internal::RRawFile;

auto &context = RRawFile::GetFailureInjectionContext();
constexpr char const* kNTupleFileName = "test_ntuple.root";
constexpr char const* kNTupleName = "ntuple";
FileRaii fileGuard(kNTupleFileName);

struct RNTupleTester {
   ROOT::Experimental::RNTuple fNtpl;

   explicit RNTupleTester(const ROOT::Experimental::RNTuple &ntpl) : fNtpl(ntpl) {}
   
   uint32_t GetSeekHeader(RNTupleTester &tester) const {return tester.fNtpl.GetSeekHeader();}
   uint32_t GetNBytesHeader(RNTupleTester &tester) const {return tester.fNtpl.GetNBytesHeader();}
   uint32_t GetLenHeader(RNTupleTester &tester) const {return tester.fNtpl.GetLenHeader();}
   uint32_t GetSeekFooter(RNTupleTester &tester) const {return tester.fNtpl.GetSeekFooter();}
   uint32_t GetNBytesFooter(RNTupleTester &tester) const {return tester.fNtpl.GetNBytesFooter();}
   uint32_t GetLenFooter(RNTupleTester &tester) const {return tester.fNtpl.GetLenFooter();}
};

void setUpNtuple(int compression) {
   auto model = RNTupleModel::Create();
   auto ptrPt = model->MakeField<float>("pt");
   auto ptrE = model->MakeField<float>("energy");

   RNTupleWriteOptions options;
   options.SetCompression(compression);
   auto writer = RNTupleWriter::Recreate(std::move(model), kNTupleName, fileGuard.GetPath(), options);

   *ptrPt = 1.0;
   *ptrE = 2.0;
   writer->Fill();  
}

RNTuple readNTuple(){
   auto f = TFile::Open(fileGuard.GetPath().c_str(),"READ");
   auto ntpl = f->Get<ROOT::Experimental::RNTuple>(kNTupleName);

   return *ntpl;
}

RNTupleTester getTester() {
   RNTupleTester tester(readNTuple());

   return tester;
}

void openNTuple(){
   auto reader = RNTupleReader::Open(kNTupleName, fileGuard.GetPath());
}

void getPageRange(std::string fieldName, uint32_t columnIndex, uint32_t entryIndex, uint32_t* size, uint64_t* offset){
   auto ntupleReader = RNTupleReader::Open(kNTupleName, fileGuard.GetPath());
   const auto descriptor = ntupleReader->GetDescriptor();
   auto fieldId = descriptor->FindFieldId(fieldName);
   auto columnId = descriptor->FindPhysicalColumnId(fieldId, columnIndex);
   auto clusterId = descriptor->FindClusterId(columnId, entryIndex);
   const auto &clusterDescriptor = descriptor->GetClusterDescriptor(clusterId);
   auto &pageRangeXcl = clusterDescriptor.GetPageRange(columnId);

   const auto &pageInfo = pageRangeXcl.fPageInfos[0];
   auto loc = pageInfo.fLocator;
   *size = loc.fBytesOnStorage;
   *offset = loc.GetPosition<std::uint64_t>();
}

TEST(RNTuple, UncompressedShortReadPage)
{
   try {
      uint32_t size;
      uint64_t offset;

      setUpNtuple(0);
      getPageRange("pt", 0, 0, &size, &offset);

      context.fFailureType = RRawFile::EFailureType::kShortRead;
      context.fRangeBegin = offset;
      context.fRangeEnd = offset + size;
      
      openNTuple();      
      FAIL() << "short reads in pages should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("no RNTuple named '"+std::string(kNTupleName)+"' in file '"+std::string(kNTupleFileName)+"'"));
   }
}

TEST(RNTuple, CompressedShortReadPage)
{
   try {
      uint32_t size;
      uint64_t offset;

      setUpNtuple(505);
      getPageRange("pt", 0, 0, &size, &offset);

      context.fFailureType = RRawFile::EFailureType::kShortRead;
      context.fRangeBegin = offset;
      context.fRangeEnd = offset + size;
      
      openNTuple();      
      FAIL() << "short reads in pages should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("no RNTuple named '"+std::string(kNTupleName)+"' in file '"+std::string(kNTupleFileName)+"'"));
   }
}

TEST(RNTuple, UncompressedShortReadHeader)
{
   try {
      setUpNtuple(0);
      RNTupleTester tester = getTester();

      context.fFailureType = RRawFile::EFailureType::kShortRead;
      context.fRangeBegin = tester.GetSeekHeader(tester);
      context.fRangeEnd = tester.GetSeekHeader(tester) + tester.GetNBytesHeader(tester);
      
      openNTuple();      
      FAIL() << "short reads in header should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("no RNTuple named '"+std::string(kNTupleName)+"' in file '"+std::string(kNTupleFileName)+"'"));
   }
}

TEST(RNTuple, CompressedShortReadHeader)
{
   try {
      setUpNtuple(505);
      RNTupleTester tester = getTester();

      context.fFailureType = RRawFile::EFailureType::kShortRead;
      context.fRangeBegin = tester.GetSeekHeader(tester);
      context.fRangeEnd = tester.GetSeekHeader(tester) + tester.GetNBytesHeader(tester);
      
      openNTuple();      
      FAIL() << "short reads in header should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("no RNTuple named '"+std::string(kNTupleName)+"' in file '"+std::string(kNTupleFileName)+"'"));
   }
}

TEST(RNTuple, UncompressedShortReadFooter)
{
   try {
      setUpNtuple(0);
      RNTupleTester tester = getTester();

      context.fFailureType = RRawFile::EFailureType::kShortRead;
      context.fRangeBegin = tester.GetSeekFooter(tester);
      context.fRangeEnd = tester.GetSeekFooter(tester) + tester.GetNBytesFooter(tester);
      
      openNTuple();      
      FAIL() << "short reads in footer should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("no RNTuple named '"+std::string(kNTupleName)+"' in file '"+std::string(kNTupleFileName)+"'"));
   }
}

TEST(RNTuple, CompressedShortReadFooter)
{
   try {
      setUpNtuple(505);
      RNTupleTester tester = getTester();

      context.fFailureType = RRawFile::EFailureType::kShortRead;
      context.fRangeBegin = tester.GetSeekFooter(tester);
      context.fRangeEnd = tester.GetSeekFooter(tester) + tester.GetNBytesFooter(tester);
      
      openNTuple();      
      FAIL() << "short reads in footer should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("no RNTuple named '"+std::string(kNTupleName)+"' in file '"+std::string(kNTupleFileName)+"'"));
   }
}

// RRawFile tests should check that the buffer has actually changed

// TEST(RNTuple, UncompressedShortRead)
// {
//    // ensure file clean-up after test completes
//    FileRaii fileGuard("test_ntuple.root");

//    //auto &params = RRawFile::GetFailureInjectionContext();
//    //params.fFailureType = RRawFile::EFailureType::kShortRead;
   
//    //params.fRangeBegin = 0;
//    //params.fRangeEnd = 0;
   
//    {
//       // create an RNTuple instance with 2 fields and no compression and fill with data
//       auto model = RNTupleModel::Create();
//       auto ptrPt = model->MakeField<float>("pt");
//       auto ptrE = model->MakeField<float>("energy");

//       RNTupleWriteOptions options;
//       options.SetCompression(0);
//       auto writer = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath(), options);

//       // TODO: add more content to the file than just one entry
//       *ptrPt = 1.0;
//       *ptrE = 2.0;
//       writer->Fill();
//    }

//    auto reader = RNTupleReader::Open("ntuple", fileGuard.GetPath());
//    auto viewPt = reader->GetView<float>("pt");

//    // for (auto i : reader->GetEntryRange()) {
//    //   std::cout << viewPt(i) << std::endl;
//    // }

//    EXPECT_EQ(1.0, viewPt(0));

//    // try {
//    //    auto reader = RNTupleReader::Open("ntuple", fileGuard.GetPath());
//    //    auto viewPt = reader->GetView<float>("pt");
//    //    FAIL() << "repeated field names should throw";
//    // } catch (const RException& err) {
//    //    EXPECT_THAT(err.what(), testing::HasSubstr("expected RNTuple named"));
//    // }
// }









// // bit flip in footer. compressed file
// TEST(RNTuple, BitFlipInPageCompressedFile)
// {
//    constexpr char const* kNTupleFileName = "test_ntuple_compression_bitflips.root";
//    // ensure file clean-up after test completes
//    FileRaii fileGuard(kNTupleFileName);

//    {
//       // create an RNTuple instance with 2 fields and no compression and fill with data
//       auto model = RNTupleModel::Create();
//       auto ptrPt = model->MakeField<float>("pt");
//       auto ptrE = model->MakeField<float>("energy");

//       RNTupleWriteOptions options;
//       options.SetCompression(505);
//       auto writer = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath(), options);

//       // TODO: add more content to the file than just one entry
//       *ptrPt = 1.0;
//       *ptrE = 2.0;
//       writer->Fill();
//    }

//    // open file and inject bit flip
//    {
//       auto f = TFile::Open(fileGuard.GetPath().c_str(), "READ");
//       auto ntpl = f->Get<ROOT::Experimental::RNTuple>("ntuple");

//       std::unique_ptr<RNTupleReader> ntupleReader = RNTupleReader::Open("ntuple", fileGuard.GetPath());
//       const auto descriptor = ntupleReader->GetDescriptor();
//       auto fieldId = descriptor->FindFieldId("pt");
//       auto columnId = descriptor->FindPhysicalColumnId(fieldId,0);
//       auto clusterId = descriptor->FindClusterId(columnId,0);
//       const auto &clusterDescriptor = descriptor->GetClusterDescriptor(clusterId);
//       auto &pageRangeXcl = clusterDescriptor.GetPageRange(clusterId);
//       const auto &pageInfo = pageRangeXcl.fPageInfos[0];
//       auto loc = pageInfo.fLocator;
//       auto nelem = pageInfo.fNElements;
//       auto offset = loc.GetPosition<std::uint64_t>();

//       ROOT::Internal::RRawFile::GetBitFlipParams().rng_begin = offset;
//       ROOT::Internal::RRawFile::GetBitFlipParams().rng_end = nelem;
//    }

//    auto f = TFile::Open(fileGuard.GetPath().c_str(), "READ_COMPRESS");

// }



// // bit flip in footer. uncompressed file
// TEST(RNTuple, BitFlipInPageUncompressedFile)
// {
//    constexpr char const* kNTupleFileName = "test_ntuple_compression_bitflips.root";
//    // ensure file clean-up after test completes
//    FileRaii fileGuard(kNTupleFileName);

//    {
//       // create an RNTuple instance with 2 fields and no compression and fill with data
//       auto model = RNTupleModel::Create();
//       auto ptrPt = model->MakeField<float>("pt");
//       auto ptrE = model->MakeField<float>("energy");

//       RNTupleWriteOptions options;
//       options.SetCompression(0);
//       auto writer = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath(), options);

//       // TODO: add more content to the file than just one entry
//       *ptrPt = 1.0;
//       *ptrE = 2.0;
//       writer->Fill();
//    }

//    // open file and inject bit flip
//    {
//       auto f = TFile::Open(fileGuard.GetPath().c_str(), "READ");
//       auto ntpl = f->Get<ROOT::Experimental::RNTuple>("ntuple");

//       std::unique_ptr<RNTupleReader> ntupleReader = RNTupleReader::Open("ntuple", fileGuard.GetPath());
//       const auto descriptor = ntupleReader->GetDescriptor();
//       auto fieldId = descriptor->FindFieldId("pt");
//       auto columnId = descriptor->FindPhysicalColumnId(fieldId,0);
//       auto clusterId = descriptor->FindClusterId(columnId,0);
//       const auto &clusterDescriptor = descriptor->GetClusterDescriptor(clusterId);
//       auto &pageRangeXcl = clusterDescriptor.GetPageRange(clusterId);
//       const auto &pageInfo = pageRangeXcl.fPageInfos[0];
//       auto loc = pageInfo.fLocator;
//       auto nelem = pageInfo.fNElements;
//       auto offset = loc.GetPosition<std::uint64_t>();

//       ROOT::Internal::RRawFile::GetBitFlipParams().rng_begin = offset;
//       ROOT::Internal::RRawFile::GetBitFlipParams().rng_end = nelem;
//    }

//    // attempt to open file again
//    try {
//       auto f = TFile::Open(fileGuard.GetPath().c_str(), "READ");
//    }
//    catch (const std::exception& ex){

//       FAIL() << "Operation failed: " << ex.what();
//    }
// }



// // bit flip in footer. compressed file
// TEST(RNTuple, BitFlipInFooterCompressedFile)
// {
//    constexpr char const* kNTupleFileName = "test_ntuple_compression_bitflips.root";
//    // ensure file clean-up after test completes
//    FileRaii fileGuard(kNTupleFileName);

//    {
//       // create an RNTuple instance with 2 fields and no compression and fill with data
//       auto model = RNTupleModel::Create();
//       auto ptrPt = model->MakeField<float>("pt");
//       auto ptrE = model->MakeField<float>("energy");

//       RNTupleWriteOptions options;
//       options.SetCompression(505);
//       auto writer = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath(), options);

//       // TODO: add more content to the file than just one entry
//       *ptrPt = 1.0;
//       *ptrE = 2.0;
//       writer->Fill();
//    }

//    // open file and inject bit flip
//    {
//       auto f = TFile::Open(fileGuard.GetPath().c_str(), "READ");
//       auto ntpl = f->Get<ROOT::Experimental::RNTuple>("ntuple");
//       ROOT::Internal::RRawFile::GetBitFlipParams().rng_begin = ntpl->GetSeekFooter();
//       ROOT::Internal::RRawFile::GetBitFlipParams().rng_end = ntpl->GetSeekFooter() + ntpl->GetSeekFooter();
//    }

//    auto f = TFile::Open(fileGuard.GetPath().c_str(), "READ_COMPRESS");

// }


// // bit flip in footer. uncompressed file
// TEST(RNTuple, BitFlipInFooterUncompressedFile)
// {
//    constexpr char const* kNTupleFileName = "test_ntuple_compression_bitflips.root";
//    // ensure file clean-up after test completes
//    FileRaii fileGuard(kNTupleFileName);

//    {
//       // create an RNTuple instance with 2 fields and no compression and fill with data
//       auto model = RNTupleModel::Create();
//       auto ptrPt = model->MakeField<float>("pt");
//       auto ptrE = model->MakeField<float>("energy");

//       RNTupleWriteOptions options;
//       options.SetCompression(0);
//       auto writer = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath(), options);

//       // TODO: add more content to the file than just one entry
//       *ptrPt = 1.0;
//       *ptrE = 2.0;
//       writer->Fill();
//    }

//    // open file and inject bit flip
//    {
//       auto f = TFile::Open(fileGuard.GetPath().c_str(), "READ");
//       auto ntpl = f->Get<ROOT::Experimental::RNTuple>("ntuple");
//       ROOT::Internal::RRawFile::GetBitFlipParams().rng_begin = ntpl->GetSeekFooter();
//       ROOT::Internal::RRawFile::GetBitFlipParams().rng_end = ntpl->GetSeekFooter() + ntpl->GetNBytesFooter();
//    }

//    // attempt to open file again
//    try {
//       auto f = TFile::Open(fileGuard.GetPath().c_str(), "READ");
//    }
//    catch (const std::exception& ex){

//       FAIL() << "Operation failed: " << ex.what();
//    }
// }



// // bit flip in header. compressed file
// TEST(RNTuple, BitFlipInHeaderCompressedFile)
// {
//    constexpr char const* kNTupleFileName = "test_ntuple_compression_bitflips.root";
//    // ensure file clean-up after test completes
//    FileRaii fileGuard(kNTupleFileName);

//    {
//       // create an RNTuple instance with 2 fields and no compression and fill with data
//       auto model = RNTupleModel::Create();
//       auto ptrPt = model->MakeField<float>("pt");
//       auto ptrE = model->MakeField<float>("energy");

//       RNTupleWriteOptions options;
//       options.SetCompression(505);
//       auto writer = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath(), options);

//       // TODO: add more content to the file than just one entry
//       *ptrPt = 1.0;
//       *ptrE = 2.0;
//       writer->Fill();
//    }

//    // open file and inject bit flip
//    {
//       auto f = TFile::Open(fileGuard.GetPath().c_str(), "READ");
//       auto ntpl = f->Get<ROOT::Experimental::RNTuple>("ntuple");
//       ROOT::Internal::RRawFile::GetBitFlipParams().rng_begin = ntpl->GetSeekHeader();
//       ROOT::Internal::RRawFile::GetBitFlipParams().rng_end = ntpl->GetSeekHeader() + ntpl->GetNBytesHeader();
//    }

//    auto f = TFile::Open(fileGuard.GetPath().c_str(), "READ_COMPRESS");

// }






// // no error injection. uncompressed file
// TEST(RNTuple, UncompressedAndNoBitFlip)
// {
//    //test
//    // ensure file clean-up after test completes
//    FileRaii fileGuard("test_ntuple_uncompressed_bitflips.root");
   
//    {
//       // create an RNTuple instance with 2 fields and no compression and fill with data
//       auto model = RNTupleModel::Create();
//       auto ptrPt = model->MakeField<float>("pt");
//       auto ptrE = model->MakeField<float>("energy");

//       RNTupleWriteOptions options;
//       options.SetCompression(0);
//       auto writer = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath(), options);

//       // TODO: add more content to the file than just one entry
//       *ptrPt = 1.0;
//       *ptrE = 2.0;
//       writer->Fill();
//    }

//    auto reader = RNTupleReader::Open("ntuple", fileGuard.GetPath());
//    auto viewPt = reader->GetView<float>("pt");

//    for (auto i : reader->GetEntryRange()) {
//      std::cout << viewPt(i) << std::endl;
//    }

//    EXPECT_EQ(1.0, viewPt(0));
// }



// // no error injection. compressed file
// TEST(RNTuple, CompressedAndNoBitFlip)
// {
//    // ensure file clean-up after test completes
//    FileRaii fileGuard("test_ntuple_compression_bitflips.root");
   
//    {
//       // create an RNTuple instance with 2 fields and no compression and fill with data
//       auto model = RNTupleModel::Create();
//       auto ptrPt = model->MakeField<float>("pt");
//       auto ptrE = model->MakeField<float>("energy");

//       RNTupleWriteOptions options;
//       options.SetCompression(505);
//       auto writer = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath(), options);

//       // TODO: add more content to the file than just one entry
//       *ptrPt = 1.0;
//       *ptrE = 2.0;
//       writer->Fill();
//    }

//    auto reader = RNTupleReader::Open("ntuple", fileGuard.GetPath());
//    auto viewPt = reader->GetView<float>("pt");

//    for (auto i : reader->GetEntryRange()) {
//      std::cout << viewPt(i) << std::endl;
//    }

//    EXPECT_EQ(1.0, viewPt(0));
// }