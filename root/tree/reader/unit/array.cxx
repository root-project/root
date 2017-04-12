#include "TFile.h"
#include "TTree.h"
#include "TTreeReader.h"
#include "TTreeReaderArray.h"

#include "gtest/gtest.h"

#include <fstream>

TEST(TTreeReaderArray, MultiReaders) {
   // See https://root.cern.ch/phpBB3/viewtopic.php?f=3&t=22790
   TTree* tree = new TTree("TTreeReaderArrayTree", "In-memory test tree");
   double Double[6] = {42.f, 43.f, 44.f, 45.f, 46.f, 47.f};
   tree->Branch("D", &Double, "D[4]/D");

   tree->Fill();
   tree->Fill();
   tree->Fill();

   TTreeReader TR(tree);
   TTreeReaderArray<double> trDouble0(TR, "D");
   TTreeReaderArray<double> trDouble1(TR, "D");
   TTreeReaderArray<double> trDouble2(TR, "D");
   TTreeReaderArray<double> trDouble3(TR, "D");
   TTreeReaderArray<double> trDouble4(TR, "D");
   TTreeReaderArray<double> trDouble5(TR, "D");

   TR.SetEntry(1);

   EXPECT_EQ(4u, trDouble0.GetSize());
   EXPECT_EQ(4u, trDouble1.GetSize());
   EXPECT_EQ(4u, trDouble2.GetSize());
   EXPECT_EQ(4u, trDouble3.GetSize());
   EXPECT_EQ(4u, trDouble4.GetSize());
   EXPECT_EQ(4u, trDouble5.GetSize());
   for (int i = 0; i < 4; ++i) {
      EXPECT_DOUBLE_EQ(Double[i], trDouble0[i]);
      EXPECT_DOUBLE_EQ(Double[i], trDouble1[i]);
      EXPECT_DOUBLE_EQ(Double[i], trDouble2[i]);
      EXPECT_DOUBLE_EQ(Double[i], trDouble3[i]);
      EXPECT_DOUBLE_EQ(Double[i], trDouble4[i]);
      EXPECT_DOUBLE_EQ(Double[i], trDouble5[i]);
   }
}
