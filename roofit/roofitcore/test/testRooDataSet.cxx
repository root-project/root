// Tests for the RooDataSet
// Authors: Stephan Hageboeck, CERN  04/2020
//          Jonas Rembser, CERN  04/2022

#include <RooAbsPdf.h>
#include <RooDataSet.h>
#include <RooDataHist.h>
#include <RooRealVar.h>
#include <RooHelpers.h>
#include <RooCategory.h>
#include <RooWorkspace.h>
#include <RooVectorDataStore.h>

#include <TFile.h>
#include <TTree.h>
#include <TChain.h>
#include <RooDataHist.h>
#include <TRandom3.h>
#include <TH1F.h>
#include <TCut.h>
#include <TSystem.h>

#include <TRandom3.h>
#include <TH1F.h>
#include <TCut.h>

#include <fstream>
#include <memory>

#include "gtest/gtest.h"

#if !defined(_MSC_VER) || defined(R__ENABLE_BROKEN_WIN_TESTS)
/// ROOT-10676
/// The RooDataSet warns that it's not using all variables if the selection string doesn't
/// make use of all variables. Although true, the user has no way to suppress this.
TEST(RooDataSet, ImportFromTreeWithCut)
{
   RooHelpers::HijackMessageStream hijack(RooFit::INFO, RooFit::InputArguments);

   TTree tree("tree", "tree");
   double thex, they;
   tree.Branch("x", &thex);
   tree.Branch("y", &they);
   tree.Branch("z", &they);
   thex = -0.337;
   they = 1.;
   tree.Fill();

   thex = 0.337;
   they = 1.;
   tree.Fill();

   thex = 1.337;
   they = 1.;
   tree.Fill();

   RooRealVar x("x", "x", 0);
   RooRealVar y("y", "y", 0);
   RooRealVar z("z", "z", 0);
   RooDataSet data("data", "data", &tree, RooArgSet(x, y, z), "x>y");

   EXPECT_TRUE(hijack.str().empty()) << "Messages issued were: " << hijack.str();
   EXPECT_EQ(data.numEntries(), 1);

   RooRealVar *theX = dynamic_cast<RooRealVar *>(data.get(0)->find("x"));
   ASSERT_NE(theX, nullptr);
   EXPECT_FLOAT_EQ(theX->getVal(), 1.337);
}
#endif

/// ROOT-9528 Branch names are capped after a certain number of characters
TEST(RooDataSet, ImportLongBranchNames)
{

   TTree tree("theTree", "theTree");
   double doub = 0.;
   tree.Branch("HLT_mu6_mu4_bBmumux_BsmumuPhi_delayed_L1BPH_2M8_MU6MU4_BPH_0DR15_MU6MU4", &doub);
   doub = 2.;
   tree.Fill();
   doub = 4.;
   tree.Fill();

   RooRealVar *v =
      new RooRealVar("HLT_mu6_mu4_bBmumux_BsmumuPhi_delayed_L1BPH_2M8_MU6MU4_BPH_0DR15_MU6MU4",
                     "HLT_mu6_mu4_bBmumux_BsmumuPhi_delayed_L1BPH_2M8_MU6MU4_BPH_0DR15_MU6MU4", 0., -100000., 100000.);

   RooDataSet ds("ds", "ds", RooArgSet(*v), RooFit::Import(tree));
   EXPECT_EQ(static_cast<RooRealVar *>(ds.get(0)->find(*v))->getVal(), 2.);
   EXPECT_EQ(static_cast<RooRealVar *>(ds.get(1)->find(*v))->getVal(), 4.);
   EXPECT_EQ(ds.numEntries(), 2);
   EXPECT_DOUBLE_EQ(ds.sumEntries("HLT_mu6_mu4_bBmumux_BsmumuPhi_delayed_L1BPH_2M8_MU6MU4_BPH_0DR15_MU6MU4 > 3."), 1.);
}

/// ROOT-3579 Binned clone seemed to create problems with chains.
/// Code adapted from example in JIRA.
TEST(RooDataSet, BinnedClone)
{
   const char *filename[2] = {"RooDataSet_BinnedClone1.root", "RooDataSet_BinnedClone2.root"};
   double sumW = 0;

   for (unsigned int i = 0; i < 2; ++i) {
      TFile file(filename[i], "RECREATE");
      TTree tree("cand", "cand");
      double Mes, weight;
      tree.Branch("Mes", &Mes);
      tree.Branch("weight", &weight);

      for (unsigned int j = 0; j < 20; ++j) {
         Mes = 5.24 + j * 0.05 / 20. + i * 0.0003;
         weight = 1.3 + j + i;
         sumW += weight;
         tree.Fill();
      }
      file.WriteObject(&tree, "cand");
      file.Close();
   }

   TChain chain("cand");
   chain.Add(filename[0]);
   chain.Add(filename[1]);
   RooRealVar mes("Mes", "Mes", 5.28, 5.24, 5.29);
   mes.setBin(40);
   RooRealVar weight("weight", "weight", 1, 0, 100);

   {
      RooDataSet data{"dataset", "dataset", &chain, RooArgSet(mes, weight), 0, weight.GetName()};
      std::unique_ptr<RooDataHist> hist{data.binnedClone()};

      EXPECT_DOUBLE_EQ(hist->sumEntries(), sumW);

      // the original crash happened when "hist" and "data" got destructed
   }

   gSystem->Unlink(filename[0]);
   gSystem->Unlink(filename[1]);
}

/// ROOT-4580, possibly solved by ROOT-10517
TEST(RooDataSet, ReducingData)
{
   // Test Data hist and such.
   TTree mytree("tree", "tree");
   double mass_x, track0_chi2_x, track1_chi2_x;

   mytree.Branch("track0_chi2", &track0_chi2_x, "track0_chi2/D");
   mytree.Branch("track1_chi2", &track1_chi2_x, "track1_chi2/D");
   mytree.Branch("mass", &mass_x, "mass/D");
   for (int i = 0; i < 50; i++) {
      track0_chi2_x = gRandom->Landau(1, 0.5);
      track1_chi2_x = gRandom->Landau(1, 0.5);
      mass_x = gRandom->Gaus(20, 0.5);
      mytree.Fill();
   }

   double chi2cutval = 1.0;
   constexpr double massmin = 0;
   constexpr double massmax = 40;

   // Now use roofit
   // observables from ttree
   RooRealVar mymass("mass", "mass", massmin, massmax);
   RooRealVar track0_chi2("track0_chi2", "track0_chi2", -10., 90);
   RooRealVar track1_chi2("track1_chi2", "track1_chi2", -10., 90);

   // get the datasets
   RooDataSet *data_unbinned =
      new RooDataSet("mass_example", "mass example", &mytree, RooArgSet(mymass, track0_chi2, track1_chi2));
   std::unique_ptr<RooDataHist> data(data_unbinned->binnedClone("data"));

   for (int i = 0; i < 3; ++i) {
      // Check with root:
      TH1F test_hist(Form("h%i", i), "histo", 10, massmin, massmax);
      chi2cutval += 0.5;

      TCut chi2_test_cut = Form("max(track0_chi2,track1_chi2)<%f", chi2cutval);

      Long64_t drawnEvents = mytree.Draw(Form("mass>>h%i", i), chi2_test_cut /*&& mass_cut*/);
      ASSERT_NE(drawnEvents, 0l);
      ASSERT_EQ(test_hist.Integral(), drawnEvents);

      // For unbinned data, reducing should be equivalent to the tree.
      std::unique_ptr<RooDataSet> data_unbinned_reduced(
         static_cast<RooDataSet *>(data_unbinned->reduce(RooFit::Cut(chi2_test_cut))));
      EXPECT_DOUBLE_EQ(data_unbinned_reduced->sumEntries(), test_hist.Integral());
      EXPECT_EQ(data_unbinned_reduced->numEntries(), test_hist.Integral());

      // When using binned data, reducing and expecting the ame number of entries as in the unbinned case is not
      // possible, since information is lost if entries to the left and right of the cut end up in the same bin.
      // Therefore, can only test <=
      std::unique_ptr<RooDataHist> reduced_binned_data(
         static_cast<RooDataHist *>(data->reduce(RooFit::Cut(chi2_test_cut))));
      if (floor(chi2cutval) == chi2cutval)
         EXPECT_FLOAT_EQ(reduced_binned_data->sumEntries(), test_hist.Integral());
      else
         EXPECT_LE(reduced_binned_data->sumEntries(), test_hist.Integral());
   }
}

/// ROOT-10845 IsOnHeap() always returned false.
TEST(RooDataSet, IsOnHeap)
{
   auto setp = new RooDataSet();
   EXPECT_TRUE(setp->IsOnHeap());

   RooDataSet setStack;
   EXPECT_FALSE(setStack.IsOnHeap());
}

/// ROOT-10935. Cannot read a category from a text file if states are given as index instead of label.
TEST(RooDataSet, ReadCategory)
{
   RooCategory cat("cat", "cat", {{"One", 1}, {"Two", 2}, {"Three", 3}});
   cat.setRange("OneTwo", "One,Two");

   RooRealVar x("x", "x", 0., 10.);

   constexpr auto filename = "datasetWithCategory.txt";
   std::ofstream file(filename);
   file << "1. One\n"
        << "2. Two\n"
        << "3. 3" << std::endl;

   auto dataset = RooDataSet::read(filename, RooArgList(x, cat));
   EXPECT_EQ(dataset->numEntries(), 3);
   for (int i = 1; i < 4; ++i) {
      EXPECT_EQ(static_cast<RooRealVar *>(dataset->get(i - 1)->find("x"))->getVal(), i);
      EXPECT_EQ(static_cast<RooCategory *>(dataset->get(i - 1)->find("cat"))->getIndex(), i);
   }

   gSystem->Unlink(filename);
}

/// ROOT-8173. Reading negative exponents from file goes wrong.
TEST(RooDataSet, ReadNegativeExponent)
{
   RooRealVar x("x", "x", 0., 10.);

   constexpr auto filename = "datasetWithCategory.txt";
   std::ofstream file(filename);
   file << "2.E-1\n"
        << "2.E-2\n"
        << "2.E-3" << std::endl;

   auto dataset = RooDataSet::read(filename, RooArgList(x));
   ASSERT_EQ(dataset->numEntries(), 3);
   const double solutions[] = {2.E-1, 2.E-2, 2.E-3};
   for (int i = 0; i < 3; ++i) {
      EXPECT_EQ(static_cast<RooRealVar *>(dataset->get(i)->find("x"))->getVal(), solutions[i]);
   }

   gSystem->Unlink(filename);
}

/// root-project/root#6408: Importing from tree deletes the TFile with the original
TEST(RooDataSet, CrashAfterImportFromTree)
{
   TTree *tree = new TTree("tree", "tree");
   double var = 1;
   tree->Branch("var", &var, "var/D");
   var = 1;
   tree->Fill();
   var = 2;
   tree->Fill();

   auto roovar = std::make_unique<RooRealVar>("var", "var", 0, 10);
   auto output_file = std::make_unique<TFile>("test.root", "RECREATE", "output_file");

   ASSERT_TRUE(output_file->IsOpen());
   auto data_set = std::make_unique<RooDataSet>("data_set", "data_set", tree, RooArgSet(*roovar));

   // Would crash, since the TFile would be deleted by importing:
   ASSERT_TRUE(output_file->IsOpen());

   EXPECT_EQ(data_set->sumEntries(), 2.);
   EXPECT_EQ(data_set->numEntries(), 2);
   EXPECT_EQ(static_cast<RooRealVar *>(data_set->get(0)->find("var"))->getVal(), 1.);
   EXPECT_EQ(static_cast<RooRealVar *>(data_set->get(1)->find("var"))->getVal(), 2.);
}

// root-project/root#6951: Broken weights after reducing RooDataSet created with RooAbsPdf::generate()
TEST(RooDataSet, ReduceWithCompositeDataStore)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   RooWorkspace ws{};
   ws.factory("Gaussian::gauss(x[-10,10], mean[3,-10,10], sigma[1,0.1,10])");
   auto &gauss = *ws.pdf("gauss");
   auto &x = *ws.var("x");

   std::size_t nEvents = 10;

   // Generate toy dataset
   std::unique_ptr<RooDataSet> dataSetPtr{gauss.generate(x, nEvents)};
   auto &dataSet = *dataSetPtr;

   // Generate a new dataset with weight column
   RooDataSet dataSetWeighted("dataSetWeighted", "dataSetWeighted", x, RooFit::WeightVar());
   for (std::size_t i = 0; i < nEvents; ++i) {
      dataSetWeighted.add(*dataSet.get(), 0.5);
   }

   // Make a new dataset that uses the RooCompositeDataStore backend
   RooCategory sample("sample", "sample");
   sample.defineType("physics");
   RooDataSet dataSetComposite("hmaster", "hmaster", x, RooFit::Index(sample),
                               RooFit::Link({{"physics", &dataSetWeighted}}));

   // Reduce the dataset with the RooCompositeDataStore
   std::unique_ptr<RooAbsData> dataSetReducedPtr{dataSetComposite.reduce("true")};
   auto &dataSetReduced = static_cast<RooDataSet &>(*dataSetReducedPtr);

   // Get the first row of all datasets
   dataSet.get(0);
   dataSetWeighted.get(0);
   dataSetComposite.get(0);
   dataSetReduced.get(0);

   // Make sure weights didn't get lost after reducing the dataset
   EXPECT_EQ(dataSetWeighted.weight(), dataSetComposite.weight());
   EXPECT_EQ(dataSetComposite.weight(), dataSetReduced.weight());
}

// Make sure that CutRange() also considers the variables not selected by
// SelectVars() in RooAbsData::reduce().
//
// Covers JIRA issue ROOT-8040.
TEST(RooDataSet, ReduceWithSelectVarsAndCutRange)
{
   using namespace RooFit;

   RooWorkspace ws;
   ws.factory("a[5,0,100]");
   ws.factory("b[10,0,100]");
   ws.defineSet("obs", "a,b");

   RooDataSet d("data", "data", *ws.set("obs"));
   ws.var("a")->setVal(5);
   ws.var("b")->setVal(5);
   d.add(*ws.set("obs"));
   ws.var("a")->setVal(15);
   ws.var("b")->setVal(15);
   d.add(*ws.set("obs"));
   ws.var("a")->setVal(5);
   ws.var("b")->setVal(15);
   d.add(*ws.set("obs"));
   ws.var("a")->setVal(15);
   ws.var("b")->setVal(5);
   d.add(*ws.set("obs"));

   ws.var("a")->setRange("myRange", 0, 10);
   ws.var("b")->setRange("myRange", 0, 10);

   std::unique_ptr<RooAbsData> reduced{d.reduce(SelectVars(*ws.var("a")), CutRange("myRange"))};

   EXPECT_EQ(reduced->numEntries(), 1);
}

// Test that importing a RooDataHist to a RooDataSet works and that it gives
// the right weight() and weightSquared().
TEST(RooDataSet, ImportDataHist)
{
   RooRealVar x{"x", "x", 0, 3};
   x.setBins(3);

   RooDataHist dh{"dh", "dh", x};

   dh.set(0, 10, 5);
   dh.set(1, 20, 15);
   dh.set(2, 30, 20);

   RooDataSet ds{"ds", "ds", x, RooFit::Import(dh)};

   for (int i = 0; i < x.numBins(); ++i) {
      dh.get(i);
      ds.get(i);
      EXPECT_FLOAT_EQ(ds.weight(), dh.weight()) << "weight() is off in bin " << i;
      EXPECT_FLOAT_EQ(ds.weightSquared(), dh.weightSquared()) << "weightSquared() is off in bin " << i;
   }
}

// Test that splitting a RooDataSet by index category does preserve the weight
// errors. Covers GitHub issue #12453.
TEST(RooDataSet, SplitDataSetWithWeightErrors)
{
   using namespace RooFit;

   RooRealVar x{"x", "x", 0, 10};
   RooCategory cat{"cat", "cat", {{"sample_0", 0}}};
   RooRealVar weight{"weight", "weight", 1.0};

   RooDataSet data1{"data", "data", {x, cat, weight}, WeightVar(weight), StoreError(weight)};

   data1.add({x, cat}, 2.0, 0.3);

   std::unique_ptr<TList> dataList{data1.split(cat, true)};
   auto &data2 = static_cast<RooDataSet &>(*dataList->At(0));

   data1.Print();

   data1.get(0);
   data2.get(0);
   EXPECT_DOUBLE_EQ(data2.weightError(), data1.weightError());
}

// The version number of the RooVectorDataStore::RealFullVector was increased
// in the 6.30 development cycle. The RealFullVector is used to store columns
// with errors, both symmetric and asymmetric.
//
// The reference file was created with the following code with ROOT 6.26.10:
//
// ```c++
// using namespace RooFit;
//
// RooRealVar x{"x", "x", 0, 10};
// RooRealVar y{"y", "y", 0, 10};
//
// RooDataSet data{"data", "data", {x, y}, StoreError(x), StoreAsymError(y)};
//
// x.setVal(5.0);
// x.setError(2.0);
// y.setVal(9.0);
// y.setAsymError(-4.0, 3.0);
//
// data.add({x, y});
//
// x.setVal(7.0);
// x.setError(3.0);
// y.setVal(4.0);
// y.setAsymError(-2.0, 1.0);
//
// data.add({x, y});
//
// std::unique_ptr<TFile> file{TFile::Open("dataSet_with_errors_6_26_10.root", "RECREATE")};
//
// file->WriteObject(&data, data.GetName());
// ```
TEST(RooDataSet, ReadDataSetWithErrors626)
{
   std::unique_ptr<TFile> file{TFile::Open("dataSet_with_errors_6_26_10.root", "READ")};

   auto data = file->Get<RooDataSet>("data");

   auto &x = static_cast<RooRealVar &>((*data->get())["x"]);
   auto &y = static_cast<RooRealVar &>((*data->get())["y"]);

   // Make sure the dataset is really using a RooVectorDataStore
   EXPECT_TRUE(dynamic_cast<RooVectorDataStore const *>(data->store()));

   data->get(0);

   EXPECT_DOUBLE_EQ(x.getVal(), 5.0);
   EXPECT_DOUBLE_EQ(x.getError(), 2.0);

   EXPECT_DOUBLE_EQ(y.getVal(), 9.0);
   EXPECT_DOUBLE_EQ(y.getAsymErrorLo(), -4.0);
   EXPECT_DOUBLE_EQ(y.getAsymErrorHi(), 3.0);

   data->get(1);

   EXPECT_DOUBLE_EQ(x.getVal(), 7.0);
   EXPECT_DOUBLE_EQ(x.getError(), 3.0);

   EXPECT_DOUBLE_EQ(y.getVal(), 4.0);
   EXPECT_DOUBLE_EQ(y.getAsymErrorLo(), -2.0);
   EXPECT_DOUBLE_EQ(y.getAsymErrorHi(), 1.0);
}
