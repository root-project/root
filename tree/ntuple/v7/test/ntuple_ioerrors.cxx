#include "ntuple_test.hxx"
#include <TFile.h>
#include <fstream>

using RRawFile = ROOT::Internal::RRawFile;

// these should not be global
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

   for (size_t i = 0; i < 1000; ++i){
      *ptrPt = i;
      *ptrE = i;
      writer->Fill(); 
   }
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

TEST(RNTuple, TriggerBitFlip)
{
   constexpr char const* kNTupleFileName = "bitFlipTest.root";
   constexpr char const* kNTupleName = "ntuple";
   FileRaii fileGuard(kNTupleFileName);

   std::ofstream ostrm(fileGuard.GetPath(), std::ios::binary);
   std::string s = "Hello, this is a test message.";
   ostrm.write(s.c_str(), s.length());
   ostrm.close();

   size_t bufferSize = s.length();
   char buffer[bufferSize];

   context.fFailureType = RRawFile::EFailureType::kBitFlip;
   context.fRangeBegin = 0;
   context.fRangeEnd = s.length();
   
   RRawFile::ROptions options;   
   auto rrawfileinstance = RRawFile::Create(fileGuard.GetPath(), options);
   
   size_t bytesRead = rrawfileinstance->ReadAt(&buffer, bufferSize, 0);

   EXPECT_FALSE(bytesRead == bufferSize && memcmp(buffer, s.c_str(), bufferSize) == 0);
}

TEST(RNTuple, NoErrorTriggered)
{
   constexpr char const* kNTupleFileName = "noErrorTest.root";
   constexpr char const* kNTupleName = "ntuple";
   FileRaii fileGuard(kNTupleFileName);

   std::ofstream ostrm(fileGuard.GetPath(), std::ios::binary);
   std::string s = "Hello, this is a test message.";
   ostrm.write(s.c_str(), s.length());
   ostrm.close();

   size_t bufferSize = s.length();
   char buffer[bufferSize];

   // set context including a range to show that no error is triggered
   context.fFailureType = RRawFile::EFailureType::kNone;
   context.fRangeBegin = 0;
   context.fRangeEnd = s.length();
   
   RRawFile::ROptions options;   
   auto rrawfileinstance = RRawFile::Create(fileGuard.GetPath(), options);
   
   size_t bytesRead = rrawfileinstance->ReadAt(&buffer, bufferSize, 0);

   EXPECT_TRUE(bytesRead == bufferSize && memcmp(buffer, s.c_str(), bufferSize) == 0);
}

TEST(RNTuple, TriggerShortRead)
{
   constexpr char const* kNTupleFileName = "shortReadTest.root";
   constexpr char const* kNTupleName = "ntuple";
   FileRaii fileGuard(kNTupleFileName);

   std::ofstream ostrm(fileGuard.GetPath(), std::ios::binary);
   std::string s = "Hello, this is a test message.";
   ostrm.write(s.c_str(), s.length());
   ostrm.close();

   size_t bufferSize = s.length();
   char buffer[bufferSize];

   context.fFailureType = RRawFile::EFailureType::kShortRead;
   context.fRangeBegin = 0;
   context.fRangeEnd = s.length();
   
   RRawFile::ROptions options;   
   auto rrawfileinstance = RRawFile::Create(fileGuard.GetPath(), options);
   
   size_t bytesRead = rrawfileinstance->ReadAt(&buffer, bufferSize, 0);

   EXPECT_FALSE(bytesRead == bufferSize && memcmp(buffer, s.c_str(), bufferSize) == 0);
}

// bit flip tests
// expect checksum errors OR failures - run tutorial file to see what the errors are

// short read tests
// require more data to be written








// TEST(RNTuple, UncompressedBitFlipHeader)
// {
//    try {
//       setUpNtuple(0);
//       RNTupleTester tester = getTester();

//       context.fFailureType = RRawFile::EFailureType::kBitFlip;
//       context.fRangeBegin = tester.GetSeekHeader(tester);
//       context.fRangeEnd = tester.GetSeekHeader(tester) + tester.GetNBytesHeader(tester);

//       // test fails when bit and byte are both set to 0
//       // context.fByteIndex = 800;
//       // context.fBitIndex = 1;
      
//       openNTuple();      
//       FAIL() << "bit flips in header should throw";
//    } catch (const RException& err) {
//       EXPECT_THAT(err.what(), testing::HasSubstr("CRC32 checksum mismatch"));
//    }

// reference test from ntuple_types.cxx
// TEST(RNTuple, TClassReadRules)
// {
//    ROOT::TestSupport::CheckDiagsRAII diags;
//    diags.requiredDiag(kWarning, "[ROOT.NTuple]", "ignoring I/O customization rule with non-transient member: a", false);
//    diags.requiredDiag(kWarning, "ROOT::Experimental::Detail::RPageSinkFile::RPageSinkFile",
//                       "The RNTuple file format will change.", false);
//    diags.requiredDiag(kWarning, "[ROOT.NTuple]", "Pre-release format version: RC 1", false);

//    FileRaii fileGuard("test_ntuple_tclassrules.ntuple");
//    char c[4] = {'R', 'O', 'O', 'T'};
//    {
//       auto model = RNTupleModel::Create();
//       auto fieldKlass = model->MakeField<StructWithIORules>("klass");
//       auto ntuple = RNTupleWriter::Recreate(std::move(model), "f", fileGuard.GetPath());
//       for (int i = 0; i < 20; i++) {
//          *fieldKlass = StructWithIORules{/*a=*/static_cast<float>(i), /*chars=*/c};
//          ntuple->Fill();
//       }
//    }

//    auto ntuple = RNTupleReader::Open("f", fileGuard.GetPath());
//    EXPECT_EQ(20U, ntuple->GetNEntries());
//    auto viewKlass = ntuple->GetView<StructWithIORules>("klass");
//    for (auto i : ntuple->GetEntryRange()) {
//       float fi = static_cast<float>(i);
//       EXPECT_EQ(fi, viewKlass(i).a);
//       EXPECT_TRUE(0 == memcmp(c, viewKlass(i).s.chars, sizeof(c)));

//       // The following values are set from a read rule; see CustomStructLinkDef.h
//       EXPECT_EQ(fi + 1.0f, viewKlass(i).b);
//       EXPECT_EQ(viewKlass(i).a + viewKlass(i).b, viewKlass(i).c);
//       EXPECT_EQ("ROOT", viewKlass(i).s.str);
//    }
// }

      // assert failures and also assert checksum
   // ROOT::TestSupport::CheckDiagsRAII diags;
   // diags.requiredDiag(kWarning, "[ROOT.NTuple]", "ignoring I/O customization rule with non-transient member: a", false);
   // diags.requiredDiag(kWarning, "ROOT::Experimental::Detail::RPageSinkFile::RPageSinkFile",
   //                    "The RNTuple file format will change.", false);
   // diags.requiredDiag(kWarning, "[ROOT.NTuple]", "Pre-release format version: RC 1", false);
// }







// https://en.cppreference.com/w/cpp/io/basic_ofstream

// std::string filename = "Test.b";
// FileRaii fileGuard(kNTupleFileName);
// std::ofstream ostrm(filename, std::ios::binary);
// write some data (maybe string instead): ostrm.write(reinterpret_cast<char*>(&d), sizeof d); 
// create an instance of rrawfile: RRawFile(std::string_view url, ROptions options);
// call readat / read v
// use memcmp to test if the failure occurred
// none, shortread, bitflip need tested



// also short reads

// TEST(RNTuple, UncompressedShortReadPage)
// {
//    try {
//       uint32_t size;
//       uint64_t offset;

//       setUpNtuple(0);
//       getPageRange("pt", 0, 0, &size, &offset);

//       context.fFailureType = RRawFile::EFailureType::kShortRead;
//       context.fRangeBegin = offset;
//       context.fRangeEnd = offset + size;
      
//       openNTuple();      
//       FAIL() << "short reads in pages should throw";
//    } catch (const RException& err) {
//       EXPECT_THAT(err.what(), testing::HasSubstr("no RNTuple named '"+std::string(kNTupleName)+"' in file '"+std::string(kNTupleFileName)+"'"));
//    }
// }

// TEST(RNTuple, CompressedShortReadPage)
// {
//    try {
//       uint32_t size;
//       uint64_t offset;

//       setUpNtuple(505);
//       getPageRange("pt", 0, 0, &size, &offset);

//       context.fFailureType = RRawFile::EFailureType::kShortRead;
//       context.fRangeBegin = offset;
//       context.fRangeEnd = offset + size;
      
//       openNTuple();      
//       FAIL() << "short reads in pages should throw";
//    } catch (const RException& err) {
//       EXPECT_THAT(err.what(), testing::HasSubstr("no RNTuple named '"+std::string(kNTupleName)+"' in file '"+std::string(kNTupleFileName)+"'"));
//    }
// }

// TEST(RNTuple, UncompressedShortReadHeader)
// {
//    try {
//       setUpNtuple(0);
//       RNTupleTester tester = getTester();

//       context.fFailureType = RRawFile::EFailureType::kShortRead;
//       context.fRangeBegin = tester.GetSeekHeader(tester);
//       context.fRangeEnd = tester.GetSeekHeader(tester) + tester.GetNBytesHeader(tester);
      
//       openNTuple();      
//       FAIL() << "short reads in header should throw";
//    } catch (const RException& err) {
//       EXPECT_THAT(err.what(), testing::HasSubstr("no RNTuple named '"+std::string(kNTupleName)+"' in file '"+std::string(kNTupleFileName)+"'"));
//    }
// }

// TEST(RNTuple, CompressedShortReadHeader)
// {
//    try {
//       setUpNtuple(505);
//       RNTupleTester tester = getTester();

//       context.fFailureType = RRawFile::EFailureType::kShortRead;
//       context.fRangeBegin = tester.GetSeekHeader(tester);
//       context.fRangeEnd = tester.GetSeekHeader(tester) + tester.GetNBytesHeader(tester);
      
//       openNTuple();      
//       FAIL() << "short reads in header should throw";
//    } catch (const RException& err) {
//       EXPECT_THAT(err.what(), testing::HasSubstr("no RNTuple named '"+std::string(kNTupleName)+"' in file '"+std::string(kNTupleFileName)+"'"));
//    }
// }

// TEST(RNTuple, UncompressedShortReadFooter)
// {
//    try {
//       setUpNtuple(0);
//       RNTupleTester tester = getTester();

//       context.fFailureType = RRawFile::EFailureType::kShortRead;
//       context.fRangeBegin = tester.GetSeekFooter(tester);
//       context.fRangeEnd = tester.GetSeekFooter(tester) + tester.GetNBytesFooter(tester);
      
//       openNTuple();      
//       FAIL() << "short reads in footer should throw";
//    } catch (const RException& err) {
//       EXPECT_THAT(err.what(), testing::HasSubstr("no RNTuple named '"+std::string(kNTupleName)+"' in file '"+std::string(kNTupleFileName)+"'"));
//    }
// }

// TEST(RNTuple, CompressedShortReadFooter)
// {
//    try {
//       setUpNtuple(505);
//       RNTupleTester tester = getTester();

//       context.fFailureType = RRawFile::EFailureType::kShortRead;
//       context.fRangeBegin = tester.GetSeekFooter(tester);
//       context.fRangeEnd = tester.GetSeekFooter(tester) + tester.GetNBytesFooter(tester);
      
//       openNTuple();      
//       FAIL() << "short reads in footer should throw";
//    } catch (const RException& err) {
//       EXPECT_THAT(err.what(), testing::HasSubstr("no RNTuple named '"+std::string(kNTupleName)+"' in file '"+std::string(kNTupleFileName)+"'"));
//    }
// }



// TEST(RNTuple, CompressedBitFlipHeader)
// {
//    try {
//       setUpNtuple(505);
//       RNTupleTester tester = getTester();

//       context.fFailureType = RRawFile::EFailureType::kBitFlip;
//       context.fRangeBegin = tester.GetSeekHeader(tester);
//       context.fRangeEnd = tester.GetSeekHeader(tester) + tester.GetNBytesHeader(tester);
      
//       openNTuple();      
//       FAIL() << "bit flips in header should throw";
//    } catch (const RException& err) {
//       EXPECT_THAT(err.what(), testing::HasSubstr("CRC32 checksum mismatch"));
//    }
// }

// TEST(RNTuple, UncompressedBitFlipFooter)
// {
//    try {
//       setUpNtuple(0);
//       RNTupleTester tester = getTester();

//       context.fFailureType = RRawFile::EFailureType::kBitFlip;
//       context.fRangeBegin = tester.GetSeekHeader(tester);
//       context.fRangeEnd = tester.GetSeekHeader(tester) + tester.GetNBytesHeader(tester);
      
//       openNTuple();      
//       FAIL() << "bit flips in footer should throw";
//    } catch (const RException& err) {
//       EXPECT_THAT(err.what(), testing::HasSubstr("CRC32 checksum mismatch"));
//    }
// }

// TEST(RNTuple, CompressedBitFlipFooter)
// {
//    try {
//       setUpNtuple(505);
//       RNTupleTester tester = getTester();

//       context.fFailureType = RRawFile::EFailureType::kBitFlip;
//       context.fRangeBegin = tester.GetSeekHeader(tester);
//       context.fRangeEnd = tester.GetSeekHeader(tester) + tester.GetNBytesHeader(tester);
      
//       openNTuple();      
//       FAIL() << "bit flips in footer should throw";
//    } catch (const RException& err) {
//       EXPECT_THAT(err.what(), testing::HasSubstr("CRC32 checksum mismatch"));
//    }
// }

// TEST(RNTuple, UncompressedBitFlipPage)
// {
//    try {
//       uint32_t size;
//       uint64_t offset;

//       setUpNtuple(0);
//       getPageRange("pt", 0, 0, &size, &offset);

//       context.fFailureType = RRawFile::EFailureType::kBitFlip;
//       context.fRangeBegin = offset;
//       context.fRangeEnd = offset + size;
      
//       openNTuple();      
//       FAIL() << "bit flips in pages should throw";
//    } catch (const RException& err) {
//       EXPECT_THAT(err.what(), testing::HasSubstr("")); // what is expected?
//    }
// }

// TEST(RNTuple, CompressedBitFlipPage)
// {
//    try {
//       uint32_t size;
//       uint64_t offset;

//       setUpNtuple(505);
//       getPageRange("pt", 0, 0, &size, &offset);

//       context.fFailureType = RRawFile::EFailureType::kBitFlip;
//       context.fRangeBegin = offset;
//       context.fRangeEnd = offset + size;
      
//       openNTuple();      
//       FAIL() << "bit flips in pages should throw";
//    } catch (const RException& err) {
//       EXPECT_THAT(err.what(), testing::HasSubstr("")); // what is expected?
//    }
// }