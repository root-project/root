
#include "TBasket.h"
#include "TBranch.h"
#include "TEnum.h"
#include "TEnumConstant.h"
#include "TMemFile.h"
#include "TTree.h"

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
   memBuffer.reserve(maxsize);
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
   Long_t offset = cl->GetDataMemberOffset("fIOBits");
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
   memBuffer.reserve(maxsize);
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
