#include "ntuple_test.hxx"
#include <TFile.h>
#include <fstream>

using RRawFile = ROOT::Internal::RRawFile;

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

/* 
   RRAWFILE TESTS
   Test that each failure type results in a physical change to the buffer
 */
TEST(RRawFile, NoErrorTriggered)
{
   auto &context = RRawFile::GetFailureInjectionContext();
   constexpr char const* kNTupleFileName = "noErrorTest.root";
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

TEST(RRawFile, TriggerShortRead)
{
   auto &context = RRawFile::GetFailureInjectionContext();
   constexpr char const* kNTupleFileName = "shortReadTest.root";
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

TEST(RRawFile, TriggerBitFlip)
{
   auto &context = RRawFile::GetFailureInjectionContext();
   constexpr char const* kNTupleFileName = "bitFlipTest.root";
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
   context.fOccurrenceProbability = 1;
   
   RRawFile::ROptions options;   
   auto rrawfileinstance = RRawFile::Create(fileGuard.GetPath(), options);
   
   size_t bytesRead = rrawfileinstance->ReadAt(&buffer, bufferSize, 0);

   EXPECT_FALSE(bytesRead == bufferSize && memcmp(buffer, s.c_str(), bufferSize) == 0);
}
/* 
   END RRAWFILE TESTS
 */

/* 
   RNTUPLE BIT FLIP TESTS
 */
TEST(RNTuple, UncompressedBitFlipHeader)
{
   auto &context = RRawFile::GetFailureInjectionContext();
   constexpr char const* kNTupleFileName = "test_ntuple.root";
   constexpr char const* kNTupleName = "ntuple";
   FileRaii fileGuard(kNTupleFileName);
   int compression = 0;

   // set up the ntuple   
   {
      auto model = RNTupleModel::Create();
      auto ptrPt = model->MakeField<float>("pt");
      auto ptrE = model->MakeField<float>("energy");

      RNTupleWriteOptions options;
      options.SetCompression(compression);
      auto writer = RNTupleWriter::Recreate(std::move(model), kNTupleName, fileGuard.GetPath(), options);

      for (size_t i = 0; i < 10000; ++i){
         *ptrPt = i;
         *ptrE = i;
         writer->Fill(); 
      }
   }

   // set up the tester
   auto f = TFile::Open(fileGuard.GetPath().c_str(),"READ");
   auto ntpl = f->Get<ROOT::Experimental::RNTuple>(kNTupleName);
   RNTupleTester tester(*ntpl);

   // set failure parameters
   context.fFailureType = RRawFile::EFailureType::kBitFlip;
   context.fRangeBegin = tester.GetSeekHeader(tester);
   context.fRangeEnd = tester.GetSeekHeader(tester) + tester.GetNBytesHeader(tester);
   context.fOccurrenceProbability = 1;

   // attempt to open the file  
   try {      
      auto reader = RNTupleReader::Open(kNTupleName, fileGuard.GetPath());
    
      FAIL() << "bit flips in header should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("CRC32 checksum mismatch"));
   }
}

TEST(RNTuple, UncompressedBitFlipFooter)
{
   auto &context = RRawFile::GetFailureInjectionContext();
   constexpr char const* kNTupleFileName = "test_ntuple.root";
   constexpr char const* kNTupleName = "ntuple";
   FileRaii fileGuard(kNTupleFileName);
   int compression = 0;

   // set up the ntuple   
   {
      auto model = RNTupleModel::Create();
      auto ptrPt = model->MakeField<float>("pt");
      auto ptrE = model->MakeField<float>("energy");

      RNTupleWriteOptions options;
      options.SetCompression(compression);
      auto writer = RNTupleWriter::Recreate(std::move(model), kNTupleName, fileGuard.GetPath(), options);

      for (size_t i = 0; i < 10000; ++i){
         *ptrPt = i;
         *ptrE = i;
         writer->Fill(); 
      }
   }

   // set up the tester
   auto f = TFile::Open(fileGuard.GetPath().c_str(),"READ");
   auto ntpl = f->Get<ROOT::Experimental::RNTuple>(kNTupleName);
   RNTupleTester tester(*ntpl);

   // set failure parameters
   context.fFailureType = RRawFile::EFailureType::kBitFlip;
   context.fRangeBegin = tester.GetSeekFooter(tester);
   context.fRangeEnd = tester.GetSeekFooter(tester) + tester.GetNBytesFooter(tester);
   context.fOccurrenceProbability = 1;

   // attempt to open the file  
   try {      
      auto reader = RNTupleReader::Open(kNTupleName, fileGuard.GetPath());
    
      FAIL() << "bit flips in header should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("CRC32 checksum mismatch"));
   }
}

// giving a seg fault - needs looked at
// also need to do a compressed test for this
// TEST(RNTuple, UncompressedBitFlipPage)
// {
//    try {
//       auto &context = RRawFile::GetFailureInjectionContext();
//       constexpr char const* kNTupleFileName = "test_ntuple.root";
//       constexpr char const* kNTupleName = "ntuple";
//       FileRaii fileGuard(kNTupleFileName);
//       uint32_t* size;
//       uint64_t* offset;
//       int compression = 0;

//       {
//          auto model = RNTupleModel::Create();
//          auto ptrPt = model->MakeField<float>("pt");
//          auto ptrE = model->MakeField<float>("energy");

//          RNTupleWriteOptions options;
//          options.SetCompression(compression);
//          auto writer = RNTupleWriter::Recreate(std::move(model), kNTupleName, fileGuard.GetPath(), options);

//          for (size_t i = 0; i < 1000; ++i){
//             *ptrPt = i;
//             *ptrE = i;
//             writer->Fill(); 
//          }
//       }
      
//       {
//          auto ntupleReader = RNTupleReader::Open(kNTupleName, fileGuard.GetPath());
//          const auto descriptor = ntupleReader->GetDescriptor();
//          auto fieldId = descriptor->FindFieldId("pt");
//          auto columnId = descriptor->FindPhysicalColumnId(fieldId, 0);
//          auto clusterId = descriptor->FindClusterId(columnId, 0);
//          const auto &clusterDescriptor = descriptor->GetClusterDescriptor(clusterId);
//          auto &pageRangeXcl = clusterDescriptor.GetPageRange(columnId);

//          const auto &pageInfo = pageRangeXcl.fPageInfos[0];
//          auto loc = pageInfo.fLocator;
//          *size = loc.fBytesOnStorage;
//          *offset = loc.GetPosition<std::uint64_t>();
//       }

//       context.fFailureType = RRawFile::EFailureType::kBitFlip;
//       context.fRangeBegin = *offset;
//       context.fRangeEnd = *offset + *size;
      
//       auto reader = RNTupleReader::Open(kNTupleName, fileGuard.GetPath());     
//    } catch (const RException& err) {
//       EXPECT_THAT(err.what(), testing::HasSubstr("")); // what is expected?
//    }
// }
/* 
   END RNTUPLE BIT FLIP TESTS
 */

/* 
   RNTUPLE SHORT READ TESTS
 */
TEST(RNTuple, UncompressedShortReadHeader)
{
   auto &context = RRawFile::GetFailureInjectionContext();
   constexpr char const* kNTupleFileName = "test_ntuple.root";
   constexpr char const* kNTupleName = "ntuple";
   FileRaii fileGuard(kNTupleFileName);
   int compression = 0;

   // set up the ntuple   
   {
      auto model = RNTupleModel::Create();
      auto ptrPt = model->MakeField<float>("pt");
      auto ptrE = model->MakeField<float>("energy");

      RNTupleWriteOptions options;
      options.SetCompression(compression);
      auto writer = RNTupleWriter::Recreate(std::move(model), kNTupleName, fileGuard.GetPath(), options);

      for (size_t i = 0; i < 10000; ++i){
         *ptrPt = i;
         *ptrE = i;
         writer->Fill(); 
      }
   }

   // set up the tester
   auto f = TFile::Open(fileGuard.GetPath().c_str(),"READ");
   auto ntpl = f->Get<ROOT::Experimental::RNTuple>(kNTupleName);
   RNTupleTester tester(*ntpl);

   context.fFailureType = RRawFile::EFailureType::kShortRead;
   context.fRangeBegin = tester.GetSeekHeader(tester);
   context.fRangeEnd = tester.GetSeekHeader(tester) + tester.GetNBytesHeader(tester);
   context.fOccurrenceProbability = 1;
   context.fBitIndex = 4;

   // attempt to open the file  
   try {      
      auto reader = RNTupleReader::Open(kNTupleName, fileGuard.GetPath());
    
      FAIL() << "short reads in header should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("CRC32 checksum mismatch"));
   }
}

TEST(RNTuple, UncompressedShortReadFooter)
{
   auto &context = RRawFile::GetFailureInjectionContext();
   constexpr char const* kNTupleFileName = "test_ntuple.root";
   constexpr char const* kNTupleName = "ntuple";
   FileRaii fileGuard(kNTupleFileName);
   int compression = 0;

   // set up the ntuple   
   {
      auto model = RNTupleModel::Create();
      auto ptrPt = model->MakeField<float>("pt");
      auto ptrE = model->MakeField<float>("energy");

      RNTupleWriteOptions options;
      options.SetCompression(compression);
      auto writer = RNTupleWriter::Recreate(std::move(model), kNTupleName, fileGuard.GetPath(), options);

      for (size_t i = 0; i < 10000; ++i){
         *ptrPt = i;
         *ptrE = i;
         writer->Fill(); 
      }
   }

   // set up the tester
   auto f = TFile::Open(fileGuard.GetPath().c_str(),"READ");
   auto ntpl = f->Get<ROOT::Experimental::RNTuple>(kNTupleName);
   RNTupleTester tester(*ntpl);

   // trigger a short read anywhere outside of the header and footer
   context.fFailureType = RRawFile::EFailureType::kShortRead;
   context.fRangeBegin = tester.GetSeekFooter(tester);
   context.fRangeEnd = tester.GetSeekFooter(tester) + tester.GetNBytesFooter(tester);
   context.fOccurrenceProbability = 1;
   context.fBitIndex = 4;

   // attempt to open the file  
   try {      
      auto reader = RNTupleReader::Open(kNTupleName, fileGuard.GetPath());
    
      FAIL() << "short reads in footer should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("CRC32 checksum mismatch"));
   }
}

// need to specific a page range rather than current approach
// TEST(RNTuple, UncompressedShortReadPages)
// {
//    auto &context = RRawFile::GetFailureInjectionContext();
//    constexpr char const* kNTupleFileName = "test_ntuple.root";
//    constexpr char const* kNTupleName = "ntuple";
//    FileRaii fileGuard(kNTupleFileName);
//    int compression = 0;

//    // set up the ntuple   
//    {
//       auto model = RNTupleModel::Create();
//       auto ptrPt = model->MakeField<float>("pt");
//       auto ptrE = model->MakeField<float>("energy");

//       RNTupleWriteOptions options;
//       options.SetCompression(compression);
//       auto writer = RNTupleWriter::Recreate(std::move(model), kNTupleName, fileGuard.GetPath(), options);

//       for (size_t i = 0; i < 10000; ++i){
//          *ptrPt = i;
//          *ptrE = i;
//          writer->Fill(); 
//       }
//    }

//    // set up the tester
//    auto f = TFile::Open(fileGuard.GetPath().c_str(),"READ");
//    auto ntpl = f->Get<ROOT::Experimental::RNTuple>(kNTupleName);
//    RNTupleTester tester(*ntpl);

//    // trigger a short read anywhere outside of the header and footer
//    context.fFailureType = RRawFile::EFailureType::kShortRead;
//    context.fRangeBegin = tester.GetSeekHeader(tester) + tester.GetNBytesHeader(tester);
//    context.fRangeEnd = tester.GetSeekFooter(tester);
//    context.fOccurrenceProbability = 1;
//    context.fBitIndex = 0;

//    // attempt to open the file  
//    try {      
//       auto reader = RNTupleReader::Open(kNTupleName, fileGuard.GetPath());
    
//       FAIL() << "short reads in pages should throw";
//    } catch (const RException& err) {
//       EXPECT_THAT(err.what(), testing::HasSubstr("The RNTuple format is too old"));
//    }
// }
/* 
   END RNTUPLE SHORT READ TESTS
 */