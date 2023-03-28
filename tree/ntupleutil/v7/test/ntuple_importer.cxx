#include <ROOT/RNTupleImporter.hxx>

#include <TFile.h>
#include <TTree.h>
#include <TChain.h>

#include <cstdio>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "CustomStructUtil.hxx"
#include "ntupleutil_test.hxx"

using ROOT::Experimental::RNTupleImporter;
using ROOT::Experimental::RNTupleReader;

TEST(RNTupleImporter, Empty)
{
   FileRaii fileGuard("test_ntuple_importer_empty.root");
   {
      std::unique_ptr<TFile> file(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
      auto tree = std::make_unique<TTree>("tree", "");
      tree->Write();
   }

   auto importer = RNTupleImporter::Create(fileGuard.GetPath(), "tree", fileGuard.GetPath());
   importer->SetIsQuiet(true);
   EXPECT_THROW(importer->Import(), ROOT::Experimental::RException);
   importer->SetNTupleName("ntuple");
   importer->Import();
   auto reader = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   EXPECT_EQ(0U, reader->GetNEntries());
   EXPECT_THROW(importer->Import(), ROOT::Experimental::RException);
}

TEST(RNTupleImporter, CreateFromTree)
{
   FileRaii fileGuard("test_ntuple_importer_empty.root");
   {
      std::unique_ptr<TFile> file(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
      auto tree = std::make_unique<TTree>("tree", "");
      tree->Write();
   }

   std::unique_ptr<TFile> file(TFile::Open(fileGuard.GetPath().c_str()));
   auto tree = file->Get<TTree>("tree");

   auto importer = RNTupleImporter::Create(tree, fileGuard.GetPath());
   importer->SetIsQuiet(true);
   EXPECT_THROW(importer->Import(), ROOT::Experimental::RException);
   importer->SetNTupleName("ntuple");
   importer->Import();
   auto reader = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   EXPECT_EQ(0U, reader->GetNEntries());
   EXPECT_THROW(importer->Import(), ROOT::Experimental::RException);
}

TEST(RNTupleImporter, CreateFromChain)
{
   FileRaii fileGuard1("test_ntuple_create_from_chain_1.root");
   {
      std::unique_ptr<TFile> file(TFile::Open(fileGuard1.GetPath().c_str(), "RECREATE"));
      auto tree = std::make_unique<TTree>("tree", "");
      Int_t a = 42;
      // For single-leaf branches, use branch name, not leaf name
      tree->Branch("a", &a);
      tree->Fill();
      tree->Write();
   }

   FileRaii fileGuard2("test_ntuple_create_from_chain_2.root");
   {
      std::unique_ptr<TFile> file(TFile::Open(fileGuard2.GetPath().c_str(), "RECREATE"));
      auto tree = std::make_unique<TTree>("tree", "");
      Int_t a = 43;
      // For single-leaf branches, use branch name, not leaf name
      tree->Branch("a", &a);
      tree->Fill();
      tree->Write();
   }

   TChain *chain = new TChain("tree");
   chain->Add(fileGuard1.GetPath().c_str());
   chain->Add(fileGuard2.GetPath().c_str());

   auto importer = RNTupleImporter::Create(chain, fileGuard1.GetPath());
   importer->SetIsQuiet(true);
   EXPECT_THROW(importer->Import(), ROOT::Experimental::RException);
   importer->SetNTupleName("ntuple");
   importer->Import();

   auto reader = RNTupleReader::Open("ntuple", fileGuard1.GetPath());
   auto viewA = reader->GetView<std::int32_t>("a");

   EXPECT_EQ(2U, reader->GetNEntries());
   EXPECT_EQ(42, viewA(0));
   EXPECT_EQ(43, viewA(1));

   EXPECT_THROW(importer->Import(), ROOT::Experimental::RException);
}

TEST(RNTupleImporter, Simple)
{
   FileRaii fileGuard("test_ntuple_importer_simple.root");
   {
      std::unique_ptr<TFile> file(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
      auto tree = std::make_unique<TTree>("tree", "");
      bool myBool = true;
      Char_t myInt8 = -8;
      UChar_t myUInt8 = 8;
      Short_t myInt16 = -16;
      UShort_t myUInt16 = 16;
      Int_t myInt32 = -32;
      UInt_t myUInt32 = 32;
      Long64_t myInt64 = -64;
      ULong64_t myUInt64 = 64;
      Float_t myFloat = 32.0;
      Double_t myDouble = 64.0;
      // TODO(jblomer): Float16_t, Double32_t
      tree->Branch("myBool", &myBool);
      tree->Branch("myInt8", &myInt8);
      tree->Branch("myUInt8", &myUInt8);
      tree->Branch("myInt16", &myInt16);
      tree->Branch("myUInt16", &myUInt16);
      tree->Branch("myInt32", &myInt32);
      tree->Branch("myUInt32", &myUInt32);
      tree->Branch("myInt64", &myInt64);
      tree->Branch("myUInt64", &myUInt64);
      tree->Branch("myFloat", &myFloat);
      tree->Branch("myDouble", &myDouble);
      tree->Fill();
      tree->Write();
   }

   auto importer = RNTupleImporter::Create(fileGuard.GetPath(), "tree", fileGuard.GetPath());
   importer->SetIsQuiet(true);
   importer->SetNTupleName("ntuple");
   importer->Import();
   auto reader = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   EXPECT_EQ(1U, reader->GetNEntries());
   reader->LoadEntry(0);
   EXPECT_TRUE(*reader->GetModel()->Get<bool>("myBool"));
   EXPECT_EQ(-8, *reader->GetModel()->Get<char>("myInt8"));
   EXPECT_EQ(8U, *reader->GetModel()->Get<std::uint8_t>("myUInt8"));
   EXPECT_EQ(-16, *reader->GetModel()->Get<std::int16_t>("myInt16"));
   EXPECT_EQ(16U, *reader->GetModel()->Get<std::uint16_t>("myUInt16"));
   EXPECT_EQ(-32, *reader->GetModel()->Get<std::int32_t>("myInt32"));
   EXPECT_EQ(32U, *reader->GetModel()->Get<std::uint32_t>("myUInt32"));
   EXPECT_EQ(-64, *reader->GetModel()->Get<std::int64_t>("myInt64"));
   EXPECT_EQ(64U, *reader->GetModel()->Get<std::uint64_t>("myUInt64"));
   EXPECT_FLOAT_EQ(32.0, *reader->GetModel()->Get<float>("myFloat"));
   EXPECT_FLOAT_EQ(64.0, *reader->GetModel()->Get<double>("myDouble"));
}

TEST(RNTupleImporter, FieldName)
{
   FileRaii fileGuard("test_ntuple_importer_field_name.root");
   {
      std::unique_ptr<TFile> file(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
      auto tree = std::make_unique<TTree>("tree", "");
      Int_t a = 42;
      // For single-leaf branches, use branch name, not leaf name
      tree->Branch("a", &a, "b/I");
      tree->Fill();
      tree->Write();
   }

   auto importer = RNTupleImporter::Create(fileGuard.GetPath(), "tree", fileGuard.GetPath());
   importer->SetIsQuiet(true);
   importer->SetNTupleName("ntuple");
   importer->Import();
   auto reader = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   EXPECT_EQ(1U, reader->GetNEntries());
   reader->LoadEntry(0);
   EXPECT_EQ(42, *reader->GetModel()->Get<std::int32_t>("a"));
}

TEST(RNTupleImporter, CString)
{
   FileRaii fileGuard("test_ntuple_importer_cstring.root");
   {
      std::unique_ptr<TFile> file(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
      auto tree = std::make_unique<TTree>("tree", "");
      const char *myString = "R";
      tree->Branch("myString", const_cast<char *>(myString), "myString/C");
      tree->Fill();
      myString = "";
      tree->SetBranchAddress("myString", const_cast<char *>(myString));
      tree->Fill();
      myString = "ROOT RNTuple";
      tree->SetBranchAddress("myString", const_cast<char *>(myString));
      tree->Fill();
      tree->Write();
   }

   auto importer = RNTupleImporter::Create(fileGuard.GetPath(), "tree", fileGuard.GetPath());
   importer->SetIsQuiet(true);
   importer->SetNTupleName("ntuple");
   importer->Import();
   auto reader = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   EXPECT_EQ(3U, reader->GetNEntries());
   reader->LoadEntry(0);
   EXPECT_EQ(std::string("R"), *reader->GetModel()->Get<std::string>("myString"));
   reader->LoadEntry(1);
   EXPECT_EQ(std::string(""), *reader->GetModel()->Get<std::string>("myString"));
   reader->LoadEntry(2);
   EXPECT_EQ(std::string("ROOT RNTuple"), *reader->GetModel()->Get<std::string>("myString"));
}

TEST(RNTupleImporter, Leaflist)
{
   FileRaii fileGuard("test_ntuple_importer_leaflist.root");
   {
      std::unique_ptr<TFile> file(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
      auto tree = std::make_unique<TTree>("tree", "");
      struct {
         Int_t a = 1;
         Int_t b = 2;
      } leafList;
      tree->Branch("branch", &leafList, "a/I:b/I");
      tree->Fill();
      tree->Write();
   }

   auto importer = RNTupleImporter::Create(fileGuard.GetPath(), "tree", fileGuard.GetPath());
   importer->SetIsQuiet(true);
   importer->SetNTupleName("ntuple");
   importer->Import();

   auto reader = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   EXPECT_EQ(1U, reader->GetNEntries());
   // Field "branch" is an anonymous record, we cannot go through the default model here
   auto viewA = reader->GetView<std::int32_t>("branch.a");
   auto viewB = reader->GetView<std::int32_t>("branch.b");
   EXPECT_EQ(1, viewA(0));
   EXPECT_EQ(2, viewB(0));
}

TEST(RNTupleImporter, FixedSizeArray)
{
   FileRaii fileGuard("test_ntuple_importer_fixed_size_array.root");
   {
      std::unique_ptr<TFile> file(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
      auto tree = std::make_unique<TTree>("tree", "");
      Int_t a[1] = {42};
      Int_t b[2] = {1, 2};
      char c[4] = {'R', 'O', 'O', 'T'};
      tree->Branch("a", a, "a[1]/I");
      tree->Branch("b", b, "b[2]/I");
      tree->Branch("c", c, "c[4]/C");
      struct {
         Int_t a = 1;
         Int_t b[2] = {2, 3};
         Int_t c = 4;
      } leafList;
      tree->Branch("branch", &leafList, "a/I:b[2]/I:c/I");
      tree->Fill();
      tree->Write();
   }

   auto importer = RNTupleImporter::Create(fileGuard.GetPath(), "tree", fileGuard.GetPath());
   importer->SetIsQuiet(true);
   importer->SetNTupleName("ntuple");
   importer->Import();

   auto reader = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   EXPECT_EQ(1U, reader->GetNEntries());
   auto viewA = reader->GetView<std::int32_t>("a");
   auto viewB = reader->GetView<std::array<std::int32_t, 2>>("b");
   auto viewC = reader->GetView<std::array<char, 4>>("c");
   EXPECT_EQ(42, viewA(0));
   EXPECT_EQ(2U, viewB(0).size());
   EXPECT_EQ(1, viewB(0)[0]);
   EXPECT_EQ(2, viewB(0)[1]);
   EXPECT_EQ(4U, viewC(0).size());
   EXPECT_EQ('R', viewC(0)[0]);
   EXPECT_EQ('O', viewC(0)[1]);
   EXPECT_EQ('O', viewC(0)[2]);
   EXPECT_EQ('T', viewC(0)[3]);
   auto viewBranchA = reader->GetView<std::int32_t>("branch.a");
   auto viewBranchB = reader->GetView<std::array<std::int32_t, 2>>("branch.b");
   auto viewBranchC = reader->GetView<std::int32_t>("branch.c");
   EXPECT_EQ(1, viewBranchA(0));
   EXPECT_EQ(2U, viewBranchB(0).size());
   EXPECT_EQ(2, viewBranchB(0)[0]);
   EXPECT_EQ(3, viewBranchB(0)[1]);
   EXPECT_EQ(4, viewBranchC(0));
}

TEST(RNTupleImporter, LeafCountArray)
{
   FileRaii fileGuard("test_ntuple_importer_leaf_count_array.root");
   {
      std::unique_ptr<TFile> file(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
      auto tree = std::make_unique<TTree>("tree", "");
      Int_t nmuons = 1;
      Int_t begin = 1;
      Int_t njets;
      float jet_pt[2];
      Int_t middle = 2;
      float jet_eta[2];
      Int_t end = 3;
      float muon_pt[1];
      tree->Branch("nmuons", &nmuons);
      tree->Branch("begin", &begin);
      tree->Branch("njets", &njets);
      tree->Branch("jet_pt", jet_pt, "jet_pt[njets]");
      tree->Branch("middle", &middle);
      tree->Branch("jet_eta", jet_eta, "jet_eta[njets]");
      tree->Branch("end", &end);
      tree->Branch("muon_pt", muon_pt, "muon_pt[nmuons]");
      njets = 1;
      jet_pt[0] = 1.0;
      jet_eta[0] = 2.0;
      muon_pt[0] = 10.0;
      tree->Fill();
      njets = 0;
      muon_pt[0] = 11.0;
      tree->Fill();
      njets = 2;
      jet_pt[0] = 3.0;
      jet_eta[0] = 4.0;
      jet_pt[1] = 5.0;
      jet_eta[1] = 6.0;
      muon_pt[0] = 12.0;
      tree->Fill();
      tree->Write();
   }

   auto importer = RNTupleImporter::Create(fileGuard.GetPath(), "tree", fileGuard.GetPath());
   importer->SetIsQuiet(true);
   importer->SetNTupleName("ntuple");
   importer->Import();

   auto reader = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   EXPECT_EQ(3U, reader->GetNEntries());
   auto viewBegin = reader->GetView<std::int32_t>("begin");
   auto viewMiddle = reader->GetView<std::int32_t>("middle");
   auto viewEnd = reader->GetView<std::int32_t>("end");
   EXPECT_EQ(1, viewBegin(0));
   EXPECT_EQ(2, viewMiddle(0));
   EXPECT_EQ(3, viewEnd(0));
   auto viewJets = reader->GetViewCollection("_collection0");
   auto viewJetPt = viewJets.GetView<float>("jet_pt");
   auto viewJetEta = viewJets.GetView<float>("jet_eta");
   auto viewMuons = reader->GetViewCollection("_collection1");
   auto viewMuonPt = viewMuons.GetView<float>("muon_pt");
   auto viewProjectedNjets = reader->GetView<ROOT::Experimental::RNTupleCardinality>("njets");
   auto viewProjectedJetPt = reader->GetView<ROOT::RVec<float>>("jet_pt");
   auto viewProjectedJetEta = reader->GetView<ROOT::RVec<float>>("jet_eta");
   auto viewProjectedNmuons = reader->GetView<ROOT::Experimental::RNTupleCardinality>("nmuons");
   auto viewProjectedMuonPt = reader->GetView<ROOT::RVec<float>>("muon_pt");

   // Entry 0: 1 jet, 1 muon
   EXPECT_EQ(1, viewJets(0));
   EXPECT_FLOAT_EQ(1.0, viewJetPt(0));
   EXPECT_FLOAT_EQ(2.0, viewJetEta(0));
   EXPECT_EQ(1, viewMuons(0));
   EXPECT_FLOAT_EQ(10.0, viewMuonPt(0));
   EXPECT_EQ(1U, viewProjectedNjets(0));
   EXPECT_EQ(1U, viewProjectedJetPt(0).size());
   EXPECT_FLOAT_EQ(1.0, viewProjectedJetPt(0).at(0));
   EXPECT_EQ(1U, viewProjectedJetEta(0).size());
   EXPECT_FLOAT_EQ(2.0, viewProjectedJetEta(0).at(0));
   EXPECT_EQ(1U, viewProjectedNmuons(0));
   EXPECT_EQ(1U, viewProjectedMuonPt(0).size());
   EXPECT_FLOAT_EQ(10.0, viewProjectedMuonPt(0).at(0));

   // Entry 1: 0 jets, 1 muon
   EXPECT_EQ(0, viewJets(1));
   EXPECT_EQ(1, viewMuons(1));
   EXPECT_FLOAT_EQ(11.0, viewMuonPt(1));
   EXPECT_EQ(0U, viewProjectedNjets(1));
   EXPECT_EQ(0U, viewProjectedJetPt(1).size());
   EXPECT_EQ(0U, viewProjectedJetEta(1).size());
   EXPECT_EQ(1U, viewProjectedNmuons(1));
   EXPECT_EQ(1U, viewProjectedMuonPt(1).size());
   EXPECT_FLOAT_EQ(11.0, viewProjectedMuonPt(1).at(0));

   // Entry 2: 2 jets, 1 muon
   EXPECT_EQ(2, viewJets(2));
   EXPECT_FLOAT_EQ(3.0, viewJetPt(1));
   EXPECT_FLOAT_EQ(4.0, viewJetEta(1));
   EXPECT_FLOAT_EQ(5.0, viewJetPt(2));
   EXPECT_FLOAT_EQ(6.0, viewJetEta(2));
   EXPECT_EQ(1, viewMuons(2));
   EXPECT_FLOAT_EQ(12.0, viewMuonPt(2));
   EXPECT_EQ(2U, viewProjectedNjets(2));
   EXPECT_EQ(2U, viewProjectedJetPt(2).size());
   EXPECT_FLOAT_EQ(3.0, viewProjectedJetPt(2).at(0));
   EXPECT_FLOAT_EQ(5.0, viewProjectedJetPt(2).at(1));
   EXPECT_EQ(2U, viewProjectedJetEta(2).size());
   EXPECT_FLOAT_EQ(4.0, viewProjectedJetEta(2).at(0));
   EXPECT_FLOAT_EQ(6.0, viewProjectedJetEta(2).at(1));
   EXPECT_EQ(1U, viewProjectedNmuons(2));
   EXPECT_EQ(1U, viewProjectedMuonPt(2).size());
   EXPECT_FLOAT_EQ(12.0, viewProjectedMuonPt(2).at(0));
}

TEST(RNTupleImporter, STL)
{
   FileRaii fileGuard("test_ntuple_importer_stl.root");
   {
      std::unique_ptr<TFile> file(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
      auto tree = std::make_unique<TTree>("tree", "");
      auto vec = new std::vector<float>{1.0, 2.0};
      auto pair = new std::pair<float, float>{3.0, 4.0};
      auto tuple = new std::tuple<int, float, bool>{5, 6.0, true};
      tree->Branch("vec", &vec);
      tree->Branch("pair", &pair);
      tree->Branch("tuple", &tuple);
      tree->Fill();
      tree->Write();
      delete vec;
      delete pair;
      delete tuple;
   }

   auto importer = RNTupleImporter::Create(fileGuard.GetPath(), "tree", fileGuard.GetPath());
   importer->SetIsQuiet(true);
   importer->SetNTupleName("ntuple");
   importer->Import();
   auto reader = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   EXPECT_EQ(1U, reader->GetNEntries());
   reader->LoadEntry(0);

   auto vec = reader->GetModel()->Get<std::vector<float>>("vec");
   EXPECT_EQ(2U, vec->size());
   EXPECT_FLOAT_EQ(1.0, vec->at(0));
   EXPECT_FLOAT_EQ(2.0, vec->at(1));
   auto pair = reader->GetModel()->Get<std::pair<float, float>>("pair");
   EXPECT_FLOAT_EQ(3.0, pair->first);
   EXPECT_FLOAT_EQ(4.0, pair->second);
   auto tuple = reader->GetModel()->Get<std::tuple<int, float, bool>>("tuple");
   EXPECT_EQ(5, std::get<0>(*tuple));
   EXPECT_FLOAT_EQ(6.0, std::get<1>(*tuple));
   EXPECT_TRUE(std::get<2>(*tuple));
}

TEST(RNTupleImporter, CustomClass)
{
   FileRaii fileGuard("test_ntuple_importer_custom_class.root");
   {
      std::unique_ptr<TFile> file(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
      auto tree = std::make_unique<TTree>("tree", "");
      CustomStructUtil *object = nullptr;
      tree->Branch("object", &object);
      object->base = 13;
      object->a = 1.0;
      object->v1.emplace_back(2.0);
      object->v1.emplace_back(3.0);
      object->nnlo.push_back(std::vector<float>{42.0, 43.0});
      object->nnlo.push_back(std::vector<float>());
      object->nnlo.push_back(std::vector<float>{137.0});
      object->s = "ROOT";
      tree->Fill();
      tree->Write();
   }

   auto importer = RNTupleImporter::Create(fileGuard.GetPath(), "tree", fileGuard.GetPath());
   importer->SetIsQuiet(true);
   importer->SetNTupleName("ntuple");
   importer->Import();

   auto reader = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   EXPECT_EQ(1U, reader->GetNEntries());
   reader->LoadEntry(0);
   auto object = reader->GetModel()->Get<CustomStructUtil>("object");
   EXPECT_EQ(13, object->base);
   EXPECT_FLOAT_EQ(1.0, object->a);
   EXPECT_EQ(2U, object->v1.size());
   EXPECT_FLOAT_EQ(2.0, object->v1[0]);
   EXPECT_FLOAT_EQ(3.0, object->v1[1]);
   EXPECT_EQ(std::string("ROOT"), object->s);
   EXPECT_EQ(3U, object->nnlo.size());
   EXPECT_EQ(2U, object->nnlo[0].size());
   EXPECT_FLOAT_EQ(42.0, object->nnlo[0][0]);
   EXPECT_FLOAT_EQ(43.0, object->nnlo[0][1]);
   EXPECT_EQ(0U, object->nnlo[1].size());
   EXPECT_EQ(1U, object->nnlo[2].size());
   EXPECT_FLOAT_EQ(137.0, object->nnlo[2][0]);
}

TEST(RNTupleImporter, ComplexClass)
{
   int splitlevels[] = {0, 1, 99};
   for (auto lvl : splitlevels) {
      FileRaii fileGuard("test_ntuple_importer_complex_class.root");
      {
         std::unique_ptr<TFile> file(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
         auto tree = std::make_unique<TTree>("tree", "");
         ComplexStructUtil *object = nullptr;
         tree->Branch("object", &object, 32000, lvl);
         object->Init1();
         tree->Fill();
         object->Init2();
         tree->Fill();
         object->Init3();
         tree->Fill();
         tree->Write();
      }

      auto importer = RNTupleImporter::Create(fileGuard.GetPath(), "tree", fileGuard.GetPath());
      importer->SetIsQuiet(true);
      importer->SetNTupleName("ntuple");
      importer->Import();

      auto reader = RNTupleReader::Open("ntuple", fileGuard.GetPath());
      EXPECT_EQ(3U, reader->GetNEntries());
      auto object = reader->GetModel()->Get<ComplexStructUtil>("object");
      ComplexStructUtil reference;
      reader->LoadEntry(0);
      reference.Init1();
      EXPECT_EQ(reference, *object);
      reader->LoadEntry(1);
      reference.Init2();
      EXPECT_EQ(reference, *object);
      reader->LoadEntry(2);
      reference.Init3();
      EXPECT_EQ(reference, *object);
   }
}

TEST(RNTupleImporter, CollectionProxyClass)
{
   int splitlevels[] = {0, 1, 99};
   for (auto lvl : splitlevels) {
      FileRaii fileGuard("test_ntuple_importer_collection_proxy_class.root");
      {
         std::unique_ptr<TFile> file(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
         auto tree = std::make_unique<TTree>("tree", "");
         std::vector<BaseUtil> objectVec;
         tree->Branch("objectVec", &objectVec, 32000, lvl);

         for (int i = 0; i < 3; ++i) {
            BaseUtil b;
            b.base = i;
            objectVec.emplace_back(b);
         }

         tree->Fill();
         tree->Write();
      }

      auto importer = RNTupleImporter::Create(fileGuard.GetPath(), "tree", fileGuard.GetPath());
      importer->SetIsQuiet(true);
      importer->SetNTupleName("ntuple");
      importer->Import();

      auto reader = RNTupleReader::Open("ntuple", fileGuard.GetPath());
      EXPECT_EQ(1U, reader->GetNEntries());
      auto objectVec = reader->GetModel()->Get<std::vector<BaseUtil>>("objectVec");

      reader->LoadEntry(0);
      EXPECT_EQ(3, objectVec->size());

      for (int i = 0; i < 3; ++i) {
         EXPECT_EQ(i, objectVec->at(i).base);
      }
   }
}

TEST(RNTUpleImporter, MaxEntries)
{
   FileRaii fileGuard("test_ntuple_importer_max_entries.root");
   {
      std::unique_ptr<TFile> file(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
      auto tree = std::make_unique<TTree>("tree", "");
      Int_t a = 42;
      // For single-leaf branches, use branch name, not leaf name
      tree->Branch("a", &a);

      for (int i = 0; i < 5; ++i) {
         tree->Fill();
      }

      tree->Write();
   }

   auto importer = RNTupleImporter::Create(fileGuard.GetPath(), "tree", fileGuard.GetPath());

   // Base case, we don't do anything with `SetMaxEntries`.
   importer->SetIsQuiet(true);
   importer->SetNTupleName("ntuple1");
   importer->Import();

   auto reader = RNTupleReader::Open("ntuple1", fileGuard.GetPath());
   EXPECT_EQ(5U, reader->GetNEntries());

   // We only want to import 3 entries, which should happen.
   importer->SetMaxEntries(3);
   importer->SetNTupleName("ntuple2");
   importer->Import();

   reader = RNTupleReader::Open("ntuple2", fileGuard.GetPath());
   EXPECT_EQ(3U, reader->GetNEntries());

   // Now we want to import 15 entries, while the original tree only has 5 entries.
   importer->SetMaxEntries(15);
   importer->SetNTupleName("ntuple3");
   importer->Import();

   reader = RNTupleReader::Open("ntuple3", fileGuard.GetPath());
   EXPECT_EQ(5U, reader->GetNEntries());

   // Negative fMaxEntry values should be ignored.
   importer->SetMaxEntries(-15);
   importer->SetNTupleName("ntuple4");
   importer->Import();

   reader = RNTupleReader::Open("ntuple4", fileGuard.GetPath());
   EXPECT_EQ(5U, reader->GetNEntries());
}
