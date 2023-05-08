
#include "ROOT/TIOFeatures.hxx"
#include "TBasket.h"
#include "TBranch.h"
#include "TEnum.h"
#include "TEnumConstant.h"
#include "TMemFile.h"
#include "TTree.h"

#include "ROOT/TestSupport.hxx"
#include "gtest/gtest.h"

#include <vector>

static const Int_t gSampleEvents = 100;

void CreateSampleFile(TMemFile *&f)
{
   f = new TMemFile("tbasket_test.root", "CREATE");
   ASSERT_TRUE(f != nullptr);
   ASSERT_FALSE(f->IsZombie());

   TTree t1("t1", "Simple tree for testing.");
   ASSERT_FALSE(t1.IsZombie());
   Int_t idx;
   t1.Branch("idx", &idx, "idx/I");

   for (idx = 0; idx < gSampleEvents; idx++) {
      t1.Fill();
   }
   t1.Write();
}

void VerifySampleFile(TFile *f)
{
   TTree *tree = nullptr;
   f->GetObject("t1", tree);
   ASSERT_TRUE(tree != nullptr);

   Int_t saved_idx;
   tree->SetBranchAddress("idx", &saved_idx);

   EXPECT_EQ(tree->GetEntries(), gSampleEvents);
   for (Int_t idx = 0; idx < tree->GetEntries(); idx++) {
      tree->GetEntry(idx);
      EXPECT_EQ(idx, saved_idx);
   }
}

TEST(TBasket, IOBits)
{
   EXPECT_EQ(static_cast<Int_t>(TBasket::EIOBits::kSupported) |
                static_cast<Int_t>(TBasket::EUnsupportedIOBits::kUnsupported),
             (1 << static_cast<Int_t>(TBasket::kIOBitCount)) - 1);

   EXPECT_EQ(static_cast<Int_t>(TBasket::EIOBits::kSupported) &
                static_cast<Int_t>(TBasket::EUnsupportedIOBits::kUnsupported),
             0);

   TClass *cl = TClass::GetClass("TBasket");
   ASSERT_NE(cl, nullptr);
   TEnum *eIOBits = (TEnum *)cl->GetListOfEnums()->FindObject("EIOBits");
   ASSERT_NE(eIOBits, nullptr);

   Int_t supported = static_cast<Int_t>(TBasket::EIOBits::kSupported);
   Bool_t foundSupported = false;
   for (auto constant : ROOT::Detail::TRangeStaticCast<TEnumConstant>(eIOBits->GetConstants())) {

      if (!strcmp(constant->GetName(), "kSupported")) {
         foundSupported = true;
         continue;
      }
      EXPECT_EQ(constant->GetValue() & supported, constant->GetValue());
      supported -= constant->GetValue();
   }
   EXPECT_TRUE(foundSupported);
   EXPECT_EQ(supported, 0);

   TEnum *eUnsupportedIOBits = (TEnum *)cl->GetListOfEnums()->FindObject("EUnsupportedIOBits");
   ASSERT_NE(eUnsupportedIOBits, nullptr);
   Int_t unsupported = static_cast<Int_t>(TBasket::EUnsupportedIOBits::kUnsupported);
   Bool_t foundUnsupported = false;
   for (auto constant : ROOT::Detail::TRangeStaticCast<TEnumConstant>(eUnsupportedIOBits->GetConstants())) {

      if (!strcmp(constant->GetName(), "kUnsupported")) {
         foundUnsupported = true;
         continue;
      }
      EXPECT_EQ(constant->GetValue() & unsupported, constant->GetValue());
      unsupported -= constant->GetValue();
   }
   EXPECT_TRUE(foundUnsupported);
   EXPECT_EQ(unsupported, 0);
}

// Basic "sanity check" test -- can we create and delete trees?
TEST(TBasket, CreateAndDestroy)
{
   std::vector<char> memBuffer;

   TMemFile *f;
   CreateSampleFile(f);
   f->Close();

   Long64_t maxsize = f->GetSize();
   memBuffer.resize(maxsize);
   f->CopyTo(&memBuffer[0], maxsize);

   delete f;

   TMemFile f2("tbasket_test.root", &memBuffer[0], maxsize, "READ");
   ASSERT_FALSE(f2.IsZombie());
   VerifySampleFile(&f2);
}

// Create a TTree, pull out a TBasket.
TEST(TBasket, CreateAndGetBasket)
{
   TMemFile *f;
   CreateSampleFile(f);
   ASSERT_FALSE(f->IsZombie());

   TTree *tree = nullptr;
   f->GetObject("t1", tree);
   ASSERT_NE(tree, nullptr);
   ASSERT_FALSE(tree->IsZombie());

   TBranch *br = tree->GetBranch("idx");
   ASSERT_NE(br, nullptr);
   ASSERT_FALSE(br->IsZombie());

   TBasket *basket = br->GetBasket(0);
   ASSERT_NE(basket, nullptr);
   ASSERT_FALSE(basket->IsZombie());

   EXPECT_EQ(basket->GetNevBuf(), gSampleEvents);
   VerifySampleFile(f);

   f->Close();
   delete f;
}

TEST(TBasket, TestUnsupportedIO)
{
   ROOT::TestSupport::CheckDiagsRAII diags;
   diags.requiredDiag(kError, "TBasket::Streamer", "indicating this was written with a newer version of ROOT utilizing critical IO features this version of ROOT does not support", /*matchFullMessage=*/false);
   diags.requiredDiag(kError, "TBranch::GetBasket", "File: tbasket_test.root at byte:", /*matchFullMessage=*/false);

   TMemFile *f;
   // Create a file; not using the CreateSampleFile helper as
   // we must corrupt the basket here.
   f = new TMemFile("tbasket_test.root", "CREATE");
   ASSERT_NE(f, nullptr);
   ASSERT_FALSE(f->IsZombie());

   TTree t1("t1", "Simple tree for testing.");
   ASSERT_FALSE(t1.IsZombie());
   Int_t idx;
   t1.Branch("idx", &idx, "idx/I");
   for (idx = 0; idx < gSampleEvents; idx++) {
      t1.Fill();
   }

   TBranch *br = t1.GetBranch("idx");
   ASSERT_NE(br, nullptr);

   TBasket *basket = br->GetBasket(0);
   ASSERT_NE(basket, nullptr);

   TClass *cl = basket->IsA();
   ASSERT_NE(cl, nullptr);
   Longptr_t offset = cl->GetDataMemberOffset("fIOBits");
   ASSERT_GT(offset, 0); // 0 can be returned on error
   UChar_t *ioBits = reinterpret_cast<UChar_t *>(reinterpret_cast<char *>(basket) + offset);

   EXPECT_EQ(*ioBits, 0);

   // This tests that at least one bit in the bitset is available.
   // When we are down to one bitset, we'll have to expand the field.
   UChar_t unsupportedbits = ~static_cast<UChar_t>(TBasket::EIOBits::kSupported);
   EXPECT_TRUE(unsupportedbits);

   *ioBits = unsupportedbits & ((1 << 7) - 1); // Last bit should always be clear.
   br->FlushBaskets();
   t1.Write();
   f->Close();

   std::vector<char> memBuffer;
   Long64_t maxsize = f->GetSize();
   memBuffer.resize(maxsize);
   f->CopyTo(&memBuffer[0], maxsize);

   TMemFile f2("tbasket_test.root", &memBuffer[0], maxsize, "READ");
   TTree *tree;
   f2.GetObject("t1", tree);
   ASSERT_NE(tree, nullptr);
   ASSERT_FALSE(tree->IsZombie());

   br = tree->GetBranch("idx");
   ASSERT_NE(br, nullptr);

   basket = br->GetBasket(0);
   // Getting the basket should fail here and an error should have been triggered.
   ASSERT_EQ(basket, nullptr);
}

// This tests that variable-length arrays still work -- make sure various modifications
// haven't messed up this basic case.
TEST(TBasket, TestVarLengthArrays)
{
   TMemFile *f;
   // Create a file; not using the CreateSampleFile helper as
   // we want to change around the IOBits
   f = new TMemFile("tbasket_test.root", "CREATE");
   ASSERT_NE(f, nullptr);
   ASSERT_FALSE(f->IsZombie());

   TTree t1("t1", "Simple tree for testing.");
   ASSERT_FALSE(t1.IsZombie());

   Int_t idx, idx2;
   Int_t sample[10];
   Int_t elem;
   t1.Branch("idx", &idx, "idx/I");
   t1.Branch("elem", &elem, "elem/I");
   t1.Branch("sample", &sample, "sample[elem]/I");
   for (idx = 0; idx < gSampleEvents; idx++) {
      for (idx2 = 0; idx2 < 10; idx2++) {
         sample[idx2] = idx2;
      }
      elem = idx % 9;
      t1.Fill();
   }
   t1.Write();
   f->Close();
   std::vector<char> memBuffer;
   Long64_t maxsize = f->GetSize();
   memBuffer.resize(maxsize);
   f->CopyTo(&memBuffer[0], maxsize);

   TMemFile f2("tbasket_test.root", &memBuffer[0], maxsize, "READ");
   TTree *saved_t1 = nullptr;
   f2.GetObject("t1", saved_t1);
   ASSERT_NE(saved_t1, nullptr);
   ASSERT_FALSE(saved_t1->IsZombie());

   TBranch *br = saved_t1->GetBranch("sample");
   ASSERT_NE(br, nullptr);

   TBasket *basket = br->GetBasket(0);
   ASSERT_NE(basket, nullptr);

   ASSERT_NE(basket->GetEntryOffset(), nullptr);

   TClass *cl = basket->IsA();
   ASSERT_NE(cl, nullptr);
   Longptr_t offset = cl->GetDataMemberOffset("fIOBits");
   ASSERT_GT(offset, 0); // 0 can be returned on error
   UChar_t *ioBits = reinterpret_cast<UChar_t *>(reinterpret_cast<char *>(basket) + offset);
   EXPECT_EQ(*ioBits, 0);

   Int_t saved_sample[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
   Int_t saved_elem;
   saved_t1->SetBranchAddress("sample", &saved_sample);
   saved_t1->SetBranchAddress("elem", &saved_elem);
   EXPECT_EQ(saved_t1->GetEntries(), gSampleEvents);
   for (idx = 0; idx < saved_t1->GetEntries(); idx++) {
      saved_t1->GetEntry(idx);
      Int_t expected_elem = idx % 9;
      EXPECT_EQ(expected_elem, saved_elem);
      for (idx2 = 0; idx2 < expected_elem; idx2++) {
         EXPECT_EQ(saved_sample[idx2], sample[idx2]);
      }
   }
}

// A simple helper function for determining all supported features.
// Crude, but works without making tests a 'friend' class of ROOT::TIOFeatures.
UChar_t GetFeatures(const ROOT::TIOFeatures &settings) {
   UChar_t features = 0;
   for (Int_t idx = 0; idx < 8; idx++) {
      if (settings.Test(static_cast<ROOT::Experimental::EIOFeatures>(1 << idx))) {
         features |= 1 << idx;
      }
   }
   return features;
}

TEST(TBasket, TestSettingIOBits)
{
   ROOT::TestSupport::CheckDiagsRAII diags;
   diags.requiredDiag(kError, "TestFeature", "A feature is being tested for that is not supported or known.");

   TMemFile *f;
   // Create a file; not using the CreateSampleFile helper as
   // we want to change around the IOBits
   f = new TMemFile("tbasket_test.root", "CREATE");
   ASSERT_NE(f, nullptr);
   ASSERT_FALSE(f->IsZombie());

   // Note we explicitly want to test multiple trees in the file - one with generated offsets of
   // and one with them enabled.
   TTree t1("t1", "Simple tree for testing generated entry offset.");
   ASSERT_FALSE(t1.IsZombie());
   TTree t2("t2", "Simple tree for testing serialized entry offset.");
   ASSERT_FALSE(t2.IsZombie());

   ROOT::TIOFeatures settings;
   ASSERT_EQ(GetFeatures(settings), 0);
   ASSERT_FALSE(settings.Test(ROOT::Experimental::EIOFeatures::kGenerateOffsetMap));
   settings.Set(ROOT::Experimental::EIOFeatures::kGenerateOffsetMap);
   ASSERT_EQ(GetFeatures(settings), static_cast<UChar_t>(ROOT::Experimental::EIOFeatures::kGenerateOffsetMap));
   ASSERT_TRUE(settings.Test(ROOT::Experimental::EIOFeatures::kGenerateOffsetMap));
   settings.Clear(ROOT::Experimental::EIOFeatures::kGenerateOffsetMap);
   ASSERT_FALSE(settings.Test(ROOT::Experimental::EIOFeatures::kGenerateOffsetMap));
   settings.Set(ROOT::Experimental::EIOFeatures::kGenerateOffsetMap);

   t1.SetIOFeatures(settings);
   Int_t idx, idx2;
   Int_t sample[10];
   Int_t sample2[10];
   Int_t elem, elem2;
   t1.Branch("idx", &idx, "idx/I");
   t1.Branch("elem", &elem, "elem/I");
   t1.Branch("sample", &sample, "sample[elem]/I");
   t2.Branch("idx2", &idx2, "idx2/I");
   t2.Branch("elem2", &elem2, "elem2/I");
   t2.Branch("sample2", &sample2, "sample2[elem2]/I");
   for (idx = 0; idx < gSampleEvents; idx++) {
      for (idx2 = 0; idx2 < 10; idx2++) {
         sample[idx2] = idx2;
         sample2[idx2] = idx2;
      }
      idx2 = idx;
      elem = idx % 9;
      elem2 = idx % 9;
      t1.Fill();
      t2.Fill();
   }
   t1.Write();
   t2.Write();
   f->Close();
   std::vector<char> memBuffer;
   Long64_t maxsize = f->GetSize();
   memBuffer.resize(maxsize);
   f->CopyTo(&memBuffer[0], maxsize);

   TMemFile f2("tbasket_test.root", &memBuffer[0], maxsize, "READ");
   TTree *saved_t1 = nullptr, *saved_t2 = nullptr;
   f2.GetObject("t1", saved_t1);
   ASSERT_NE(saved_t1, nullptr);
   ASSERT_FALSE(saved_t1->IsZombie());
   f2.GetObject("t2", saved_t2);
   ASSERT_NE(saved_t2, nullptr);
   ASSERT_FALSE(saved_t2->IsZombie());

   TBranch *br = saved_t1->GetBranch("sample");
   ASSERT_NE(br, nullptr);
   TBranch *br2 = saved_t2->GetBranch("sample2");
   ASSERT_NE(br2, nullptr);

   TBasket *basket = br->GetBasket(0);
   ASSERT_NE(basket, nullptr);
   TBasket *basket2 = br2->GetBasket(0);
   ASSERT_NE(basket2, nullptr);

   Int_t *b1_offsets = basket->GetEntryOffset();
   Int_t *b2_offsets = basket2->GetEntryOffset();
   ASSERT_NE(b1_offsets, nullptr);
   ASSERT_NE(b2_offsets, nullptr);
   for (idx = 0; idx < gSampleEvents; idx++) {
      ASSERT_EQ(b1_offsets[idx], b2_offsets[idx]);
   }

   TClass *cl = basket->IsA();
   ASSERT_NE(cl, nullptr);
   Longptr_t offset = cl->GetDataMemberOffset("fIOBits");
   ASSERT_GT(offset, 0); // 0 can be returned on error
   UChar_t *ioBits = reinterpret_cast<UChar_t *>(reinterpret_cast<char *>(basket) + offset);
   EXPECT_EQ(*ioBits, static_cast<UChar_t>(TBasket::EIOBits::kGenerateOffsetMap));

   ioBits = reinterpret_cast<UChar_t *>(reinterpret_cast<char *>(basket2) + offset);
   EXPECT_EQ(*ioBits, 0);

   Int_t saved_sample[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
   Int_t saved_elem;
   Int_t saved_sample2[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
   Int_t saved_elem2;
   saved_t1->SetBranchAddress("sample", &saved_sample);
   saved_t1->SetBranchAddress("elem", &saved_elem);
   saved_t2->SetBranchAddress("sample2", &saved_sample2);
   saved_t2->SetBranchAddress("elem2", &saved_elem2);
   EXPECT_EQ(saved_t1->GetEntries(), gSampleEvents);
   EXPECT_EQ(saved_t2->GetEntries(), gSampleEvents);
   for (idx = 0; idx < saved_t1->GetEntries(); idx++) {
      saved_t1->GetEntry(idx);
      saved_t2->GetEntry(idx);
      Int_t expected_elem = idx % 9;
      EXPECT_EQ(expected_elem, saved_elem);
      EXPECT_EQ(expected_elem, saved_elem2);
      for (idx2 = 0; idx2 < expected_elem; idx2++) {
         EXPECT_EQ(saved_sample2[idx2], sample[idx2]);
         EXPECT_EQ(saved_sample[idx2], sample[idx2]);
      }
   }

   offset = cl->GetDataMemberOffset("fReadEntryOffset");
   ASSERT_GT(offset, 0);
   Bool_t *readEntryOffset = reinterpret_cast<Bool_t *>(reinterpret_cast<char *>(basket) + offset);
   EXPECT_EQ(*readEntryOffset, kFALSE);
   readEntryOffset = reinterpret_cast<Bool_t *>(reinterpret_cast<char *>(basket2) + offset);
   EXPECT_EQ(*readEntryOffset, kTRUE);
}
