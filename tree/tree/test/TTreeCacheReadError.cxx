#include "ROOT/TestSupport.hxx"

#include "TFile.h"
#include "TSystem.h"
#include "TTree.h"
#include "TTreeCacheUnzip.h"

#include "gtest/gtest.h"

#include <vector>

namespace {

/// A file whose `failAt`-th vectored read fails without filling the buffer, as a TNetXNGFile does on a socket timeout.
class FailingReadBuffersFile : public TFile {
   int fCalls = 0;
   int fFailAt;

public:
   FailingReadBuffersFile(const char *name, int failAt) : TFile(name), fFailAt(failAt) {}

   Bool_t ReadBuffers(char *buf, Long64_t *pos, Int_t *len, Int_t nbuf) override
   {
      if (++fCalls == fFailAt)
         return kTRUE;
      return TFile::ReadBuffers(buf, pos, len, nbuf);
   }

   int GetCalls() const { return fCalls; }
};

class TTreeCacheReadError : public ::testing::Test {
protected:
   static constexpr const char *kFileName = "ttreecache_readerror.root";
   static constexpr int kEntries = 20000;
   // The size of a cache holding the baskets of a single cluster.
   static constexpr int kCacheSize = 20000;

   static void SetUpTestSuite()
   {
      TFile f(kFileName, "RECREATE");
      TTree t("t", "t");
      int a;
      std::vector<int> v;
      t.Branch("a", &a, 4000);
      t.Branch("v", &v, 4000);
      t.SetAutoFlush(2000);
      for (int i = 0; i < kEntries; ++i) {
         a = i;
         v.assign(i % 7, i);
         t.Fill();
      }
      t.Write();
   }

   static void TearDownTestSuite() { gSystem->Unlink(kFileName); }

   /// Read all entries of `t`, checking their content, and return the number of entries that could not be read.
   static int ReadAll(TTree &t)
   {
      int a = -1;
      std::vector<int> *v = nullptr;
      t.SetBranchAddress("a", &a);
      t.SetBranchAddress("v", &v);
      int nErrors = 0;
      for (int i = 0; i < kEntries; ++i) {
         if (t.GetEntry(i) <= 0) {
            ++nErrors;
            continue;
         }
         EXPECT_EQ(a, i);
         EXPECT_NE(v, nullptr);
         if (!v)
            continue;
         EXPECT_EQ(v->size(), static_cast<std::size_t>(i % 7));
         for (int x : *v)
            EXPECT_EQ(x, i);
      }
      return nErrors;
   }
};

/// Use a TTreeCacheUnzip for the tree while in scope.
class UnzipCacheRAII {
public:
   UnzipCacheRAII(TTree &t, Long64_t cacheSize)
   {
      TTreeCacheUnzip::SetParallelUnzip(TTreeCacheUnzip::kEnable);
      new TTreeCacheUnzip(&t, cacheSize);
   }
   ~UnzipCacheRAII() { TTreeCacheUnzip::SetParallelUnzip(TTreeCacheUnzip::kDisable); }
};

} // namespace

// A failed read of the baskets of a cluster must not leave the cache serving the other baskets of that cluster from a
// buffer that was never filled.
TEST_F(TTreeCacheReadError, FailedReadBuffers)
{
   // The first vectored read is that of the cache in learning phase, the third one that of the second cluster.
   FailingReadBuffersFile f(kFileName, /*failAt=*/3);
   auto t = f.Get<TTree>("t");
   ASSERT_NE(t, nullptr);
   t->SetCacheSize(kCacheSize);

   ROOT::TestSupport::CheckDiagsRAII diags;
   diags.requiredDiag(kError, "TBranch::GetBasket", "badread=1", /*matchFullMessage=*/false);
   // Only the entry whose read failed is lost; the other baskets of its cluster are read from the file.
   EXPECT_EQ(ReadAll(*t), 1);
   EXPECT_GT(f.GetCalls(), 3);
}

// The same for TTreeCacheUnzip, where the read that fails is retried by TBasket.
TEST_F(TTreeCacheReadError, FailedReadBuffersUnzip)
{
   FailingReadBuffersFile f(kFileName, /*failAt=*/3);
   auto t = f.Get<TTree>("t");
   ASSERT_NE(t, nullptr);
   UnzipCacheRAII unzip(*t, kCacheSize);

   EXPECT_EQ(ReadAll(*t), 0);
   EXPECT_GT(f.GetCalls(), 3);
}

// When prefilling the cache during learning, the first basket is requested from TTreeCacheUnzip before its blocks are
// read; a failure to read them must not unzip whatever its buffer for compressed data holds.
TEST_F(TTreeCacheReadError, FailedReadBuffersUnzipLearnPrefill)
{
   FailingReadBuffersFile f(kFileName, /*failAt=*/1);
   auto t = f.Get<TTree>("t");
   ASSERT_NE(t, nullptr);
   UnzipCacheRAII unzip(*t, kCacheSize);
   t->GetReadCache(&f)->SetLearnPrefill(TTreeCache::kAllBranches);

   EXPECT_EQ(ReadAll(*t), 0);
   EXPECT_GT(f.GetCalls(), 1);
}
