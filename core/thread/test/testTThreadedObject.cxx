#include "ROOT/TThreadedObject.hxx"
#include "TH1F.h"
#include "TRandom.h"

#include "gtest/gtest.h"

using namespace ROOT;

bool IsSameHist(const TH1F& a, const TH1F& b)
{
   if( 0 != strcmp(a.GetName(),b.GetName())) {
      std::cout << "The names of the histograms differ: " << a.GetName() << " " << b.GetName() << std::endl;
      return false;
   }
   if( 0 != strcmp(a.GetTitle(),b.GetTitle())) {
      std::cout << "The title of the histograms differ: " << a.GetTitle() << " " << b.GetTitle() << std::endl;
      return false;
   }
   auto nbinsa = a.GetNbinsX();
   auto nbinsb = b.GetNbinsX();
   if( nbinsa != nbinsb) {
      std::cout << "The # of bins of the histograms differ: " << nbinsa << " " << nbinsb << std::endl;
      return false;
   }
   for (int i=0;i<a.GetNbinsX();++i) {
      auto binca = a.GetBinContent(i);
      auto bincb = b.GetBinContent(i);
      if (binca != bincb) {
         std::cout << "The content of bin " << i << "  of the histograms differ: " << binca << " " << bincb << std::endl;
         return false;
      }
      auto binea = a.GetBinError(i);
      auto bineb = b.GetBinError(i);
      if (binea != bineb) {
         std::cout << "The error of bin " << i << "  of the histograms differ: " << binea << " " << bineb << std::endl;
         return false;
      }
   }

   return true;
}

TEST(TThreadedObject, CreateAndDestroy)
{
   ROOT::TThreadedObject<TH1F> tto("h","h",64, -4, 4);
}

TEST(TThreadedObject, Get)
{
   TH1F model("h","h",64, -4, 4);
   ROOT::TThreadedObject<TH1F> tto("h","h",64, -4, 4);
   auto h = tto.Get();
   EXPECT_TRUE(IsSameHist(model,*h));
}

TEST(TThreadedObject, GetAtSlot)
{
   TH1F model("h","h",64, -4, 4);
   ROOT::TThreadedObject<TH1F> tto("h","h",64, -4, 4);
   auto h = tto.GetAtSlot(0);
   EXPECT_TRUE(IsSameHist(model,*h));
}

TEST(TThreadedObject, GetAtSlotUnchecked)
{
   TH1F model("h","h",64, -4, 4);
   ROOT::TThreadedObject<TH1F> tto("h","h",64, -4, 4);
   tto->SetName("h");
   auto h = tto.GetAtSlot(0);
   EXPECT_TRUE(IsSameHist(model,*h));
}

TEST(TThreadedObject, GetAtSlotRaw)
{
   TH1F model("h","h",64, -4, 4);
   ROOT::TThreadedObject<TH1F> tto("h","h",64, -4, 4);
   tto->SetName("h");
   auto h = tto.GetAtSlotRaw(0);
   EXPECT_TRUE(IsSameHist(model,*h));
}

TEST(TThreadedObject, SetAtSlot)
{
   ROOT::TThreadedObject<TH1F> tto("h","h",64, -4, 4);
   tto.SetAtSlot(1, std::make_shared<TH1F>("h","h",64, -4, 4));
   auto h0 = tto.GetAtSlot(0);
   auto h1 = tto.GetAtSlot(1);
   EXPECT_TRUE(IsSameHist(*h0,*h1));
}

TEST(TThreadedObject, Merge)
{
   TH1::AddDirectory(false);

   TH1F m0("h","h",64, -4, 4);
   TH1F m1("h","h",64, -4, 4);
   gRandom->SetSeed(1);
   m0.FillRandom("gaus");
   m1.FillRandom("gaus");
   m0.Add(&m1);

   ROOT::TThreadedObject<TH1F> tto("h","h",64, -4, 4);
   tto->SetName("h");
   tto.SetAtSlot(1, std::make_shared<TH1F>("h","h",64, -4, 4));
   gRandom->SetSeed(1);
   tto->FillRandom("gaus");
   tto.GetAtSlot(1)->FillRandom("gaus");
   auto hsum = tto.Merge();
   EXPECT_TRUE(IsSameHist(*hsum,m0));
}

TEST(TThreadedObject, SnapshotMerge)
{
   TH1::AddDirectory(false);

   TH1F m0("h","h",64, -4, 4);
   TH1F m1("h","h",64, -4, 4);
   gRandom->SetSeed(1);
   m0.FillRandom("gaus",100);
   m1.FillRandom("gaus",100);
   m0.Add(&m1);

   ROOT::TThreadedObject<TH1F> tto("h","h",64, -4, 4);
   tto->SetName("h");
   tto.SetAtSlot(1, std::make_shared<TH1F>("h","h",64, -4, 4));
   gRandom->SetSeed(1);
   tto->FillRandom("gaus",100);
   tto.GetAtSlot(1)->FillRandom("gaus",100);
   auto hsum0 = tto.SnapshotMerge();
   EXPECT_TRUE(IsSameHist(*hsum0,m0));
   auto hsum1 = tto.SnapshotMerge();
   EXPECT_TRUE(IsSameHist(*hsum1,m0));
   EXPECT_TRUE(IsSameHist(*hsum1,*hsum0));
   EXPECT_TRUE(hsum1 != hsum0);
}







