#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <ROOT/RError.hxx>
#include <ROOT/RNTupleZip.hxx>
#include <ROOT/TestSupport.hxx>

#include <Bytes.h>
#include <RZip.h>
#include <TKey.h>
#include <TMemFile.h>
#include <TMessage.h>
#include <TNamed.h>
#include <TTree.h>
#include <TTreeCacheUnzip.h>

#include <cstring>
#include <memory>
#include <string>

using ROOT::Internal::RNTupleCompressor;
using ROOT::Internal::RNTupleDecompressor;

TEST(RZip, HeaderBasics)
{
   unsigned char header[9] = {'Z', 'S', '\1', 1, 0, 0, 2, 1, 0};
   int srcsize = 0;
   int tgtsize = 0;

   EXPECT_EQ(0, R__unzip_header(&srcsize, header, &tgtsize));
   EXPECT_EQ(10, srcsize);
   EXPECT_EQ(258, tgtsize);
}

TEST(RZip, CorruptHeaderRNTuple)
{
   constexpr char content[] = {7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7};
   unsigned char blocks[50];
   unsigned char verify[50];
   static_assert(sizeof(content) < sizeof(blocks));

   const auto sz1 = RNTupleCompressor::Zip(content, sizeof(content), 101, blocks);
   EXPECT_LT(sz1, sizeof(content));

   RNTupleDecompressor::Unzip(blocks, sz1, sizeof(content), verify);
   EXPECT_EQ(0, memcmp(content, verify, sizeof(content)));

   try {
      RNTupleDecompressor::Unzip(blocks, 8, sizeof(content), verify);
      FAIL() << "too short input buffer should throw";
   } catch (const ROOT::RException &e) {
      EXPECT_THAT(e.what(), ::testing::HasSubstr("zip buffer too short"));
   }

   try {
      RNTupleDecompressor::Unzip(blocks, sz1 + 1, sizeof(content), verify);
      FAIL() << "too long input buffer should throw";
   } catch (const ROOT::RException &e) {
      EXPECT_THAT(e.what(), ::testing::HasSubstr("unexpected trailing bytes in zip buffer"));
   }

   blocks[6]++;
   try {
      RNTupleDecompressor::Unzip(blocks, sz1, sizeof(content) + 1, verify);
      FAIL() << "too long target size should throw";
   } catch (const ROOT::RException &e) {
      EXPECT_THAT(e.what(), ::testing::HasSubstr("unexpected length after unzipping the buffer"));
   }
   blocks[6]--;

   EXPECT_LT(sz1, sizeof(blocks) + 10);
   blocks[sz1] = 'Z';
   blocks[sz1 + 1] = 'S';
   blocks[sz1 + 2] = '\1';
   blocks[sz1 + 3] = 0;
   blocks[sz1 + 4] = 0;
   blocks[sz1 + 5] = 0;
   blocks[sz1 + 6] = 0;
   blocks[sz1 + 7] = 0;
   blocks[sz1 + 8] = 0;
   try {
      RNTupleDecompressor::Unzip(blocks, sz1 + 10, sizeof(verify), verify);
      FAIL() << "source length zero in the header should fail";
   } catch (const ROOT::RException &e) {
      EXPECT_THAT(e.what(), ::testing::HasSubstr("failed to unzip buffer header"));
   }

   blocks[sz1 + 3] = 1;
   try {
      RNTupleDecompressor::Unzip(blocks, sz1 + 10, sizeof(verify), verify);
      FAIL() << "source size < target size should fail";
   } catch (const ROOT::RException &e) {
      EXPECT_THAT(e.what(), ::testing::HasSubstr("failed to unzip buffer header"));
   }

   blocks[sz1 + 3] = 11;
   blocks[sz1 + 6] = 12;
   try {
      RNTupleDecompressor::Unzip(blocks, sz1 + 10, sizeof(verify), verify);
      FAIL() << "too big source size should fail";
   } catch (const ROOT::RException &e) {
      EXPECT_THAT(e.what(), ::testing::HasSubstr("failed to unzip buffer header"));
   }

   blocks[sz1 + 3] = 1;
   blocks[sz1 + 6] = sizeof(verify);
   try {
      RNTupleDecompressor::Unzip(blocks, sz1 + 10, sizeof(verify), verify);
      FAIL() << "too big target size should fail";
   } catch (const ROOT::RException &e) {
      EXPECT_THAT(e.what(), ::testing::HasSubstr("failed to unzip buffer header"));
   }
}

TEST(RZip, CorruptHeaderTKey)
{
   TMemFile writableFile("memfile.root", "RECREATE");
   writableFile.SetCompressionSettings(101);

   std::string stdstring(1000, 'x');
   TNamed tnamed;
   tnamed.SetName(stdstring.c_str());
   writableFile.WriteObject(&stdstring, "stdstring");
   writableFile.WriteObject(&tnamed, "tnamed");

   auto keysInfo = writableFile.WalkTKeys();
   std::size_t posZipStdString = 0;
   std::size_t posZipTNamed = 0;
   for (const auto &ki : keysInfo) {
      if (ki.fKeyName == "stdstring") {
         EXPECT_LT(ki.fLen, ki.fObjLen); // ensure it's compressed
         posZipStdString = ki.fSeekKey + ki.fKeyLen;
      } else if (ki.fKeyName == "tnamed") {
         EXPECT_LT(ki.fLen, ki.fObjLen); // ensure it's compressed
         posZipTNamed = ki.fSeekKey + ki.fKeyLen;
      }
   }
   EXPECT_GT(posZipStdString, 0);
   EXPECT_GT(posZipTNamed, 0);

   writableFile.Close();

   auto buffer = std::make_unique<char[]>(writableFile.GetSize());
   writableFile.CopyTo(buffer.get(), writableFile.GetSize());

   TMemFile verifyFile("memfile.root", TMemFile::ZeroCopyView_t(buffer.get(), writableFile.GetSize()));

   buffer[posZipStdString + 3]++;
   EXPECT_FALSE(verifyFile.Get<std::string>("stdstring"));
   buffer[posZipStdString + 3]--;
   buffer[posZipStdString + 6]++;
   EXPECT_FALSE(verifyFile.Get<std::string>("stdstring"));
   buffer[posZipStdString + 6]--;

   auto k = verifyFile.GetKey("tnamed");
   EXPECT_TRUE(k);
   EXPECT_TRUE(dynamic_cast<TNamed *>(k->ReadObj()));

   buffer[posZipTNamed + 3]++;
   EXPECT_FALSE(dynamic_cast<TNamed *>(k->ReadObj()));
   buffer[posZipTNamed + 3]--;
   buffer[posZipTNamed + 6]++;
   EXPECT_FALSE(dynamic_cast<TNamed *>(k->ReadObj()));
   buffer[posZipTNamed + 6]--;

   EXPECT_TRUE(verifyFile.Get<std::string>("stdstring"));
   EXPECT_TRUE(verifyFile.Get<TNamed>("tnamed"));
}

TEST(RZip, CorruptHeaderTTree)
{
   TMemFile writableFile("memfile.root", "RECREATE");
   writableFile.SetCompressionSettings(101);
   auto tree = new TTree("t", "");
   int val = 137;
   tree->Branch("val", &val);
   for (int i = 0; i < 1000; ++i)
      tree->Fill();
   tree->Write();

   auto keysInfo = writableFile.WalkTKeys();
   std::size_t posTBasket = 0;
   std::size_t keylenTBasket = 0;
   for (const auto &ki : keysInfo) {
      if (ki.fClassName != "TBasket")
         continue;

      EXPECT_EQ(0u, posTBasket);      // We expect only one basket
      EXPECT_LT(ki.fLen, ki.fObjLen); // ensure it's compressed
      posTBasket = ki.fSeekKey + ki.fKeyLen;
      keylenTBasket = ki.fKeyLen;
   }
   EXPECT_GT(posTBasket, 0);

   writableFile.Close();

   auto buffer = std::make_unique<char[]>(writableFile.GetSize());
   writableFile.CopyTo(buffer.get(), writableFile.GetSize());

   for (int headerOffset : {3, 6}) {
      TMemFile verifyFile("memfile.root", TMemFile::ZeroCopyView_t(buffer.get(), writableFile.GetSize()));

      tree = verifyFile.Get<TTree>("t");

      ROOT::TestSupport::CheckDiagsRAII checkDiag;
      checkDiag.requiredDiag(kError, "TBasket::ReadBasketBuffers", "fNbytes", /* matchFullMessage= */ false);
      checkDiag.requiredDiag(kError, "TBranch::GetBasket", "File: memfile.root", /* matchFullMessage= */ false);

      buffer[posTBasket + headerOffset]++;
      EXPECT_EQ(-1, tree->GetEntry(0));
      buffer[posTBasket + headerOffset]--;
   }

   {
      TMemFile verifyFile("memfile.root", TMemFile::ZeroCopyView_t(buffer.get(), writableFile.GetSize()));

      tree = verifyFile.Get<TTree>("t");

      TTreeCacheUnzip cache(tree);
      char *dest = nullptr;

      for (int headerOffset : {3, 6}) {
         ROOT::TestSupport::CheckDiagsRAII checkDiag;
         checkDiag.requiredDiag(kError, "TTreeCacheUnzip::UnzipBuffer", "nbytes", /* matchFullMessage= */ false);

         buffer[posTBasket + headerOffset]++;
         EXPECT_EQ(-1, cache.UnzipBuffer(&dest, &buffer[posTBasket - keylenTBasket]));
         buffer[posTBasket + headerOffset]--;
      }

      EXPECT_GT(cache.UnzipBuffer(&dest, &buffer[posTBasket - keylenTBasket]), 0);
      delete[] dest;
   }

   TMemFile verifyFile("memfile.root", TMemFile::ZeroCopyView_t(buffer.get(), writableFile.GetSize()));
   tree = verifyFile.Get<TTree>("t");
   tree->SetBranchAddress("val", &val);
   val = 0;

   EXPECT_EQ(sizeof(int), tree->GetEntry(0));
   EXPECT_EQ(137, val);
}

TEST(RZip, CorruptHeaderMessage)
{
   TNamed named;
   named.SetName(std::string(1000, 'x').c_str());
   TMessage msg(kMESS_OBJECT | kMESS_ZIP);
   msg.SetCompressionSettings(101);
   msg.WriteObject(&named);

   EXPECT_EQ(0, msg.Compress());
   EXPECT_TRUE(msg.CompBuffer());
   EXPECT_LT(msg.CompLength(), 1000);
   EXPECT_GT(msg.CompLength(), 9);

   Int_t ziplen;
   Int_t what;
   Int_t length;
   char *bufcur = msg.CompBuffer();

   frombuf(bufcur, &ziplen);
   EXPECT_EQ(msg.CompLength() - sizeof(UInt_t), ziplen);
   frombuf(bufcur, &what);
   EXPECT_EQ(kMESS_OBJECT | kMESS_ZIP, what);
   frombuf(bufcur, &length);
   EXPECT_EQ(length, msg.Length());

   {
      ROOT::TestSupport::CheckDiagsRAII checkDiag;
      checkDiag.requiredDiag(kError, "TMessage::Uncompress", "objlenRemain", /* matchFullMessage= */ false);

      bufcur[3]++;
      EXPECT_NE(0, msg.Uncompress());
      bufcur[3]--;
      bufcur[6]++;
      EXPECT_NE(0, msg.Uncompress());
      bufcur[6]--;
   }

   EXPECT_EQ(0, msg.Uncompress());
}
