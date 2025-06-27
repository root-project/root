#include "ROOT/TestSupport.hxx"
#include "ROOT/RDataFrame.hxx"

#include "ROOT/RNTuple.hxx"
#include "ROOT/RNTupleModel.hxx"
#include "ROOT/RNTupleWriter.hxx"
#include "ROOT/RNTupleReader.hxx"

#include "TROOT.h"
#include "TSystem.h"

#include "gtest/gtest.h"

#include "NTupleStruct.hxx"

#include <TFile.h>

using ROOT::RNTupleModel;
using ROOT::RNTupleReader;
using ROOT::RNTupleWriter;

using namespace ROOT::RDF;

class FileRAII {
private:
   std::string fPath;

public:
   explicit FileRAII(const std::string &path) : fPath(path) {}
   FileRAII(FileRAII &&) = default;
   FileRAII(const FileRAII &) = delete;
   FileRAII &operator=(FileRAII &&) = default;
   FileRAII &operator=(const FileRAII &) = delete;
   ~FileRAII() { std::remove(fPath.c_str()); }
   std::string GetPath() const { return fPath; }
};

TEST(RDFSnapshotRNTuple, FromScratchTemplated)
{
   FileRAII fileGuard{"RDFSnapshotRNTuple_from_scratch_templated.root"};
   const std::vector<std::string> columns = {"x"};

   auto df = ROOT::RDataFrame(25ull).Define("x", [] { return 10; });

   RSnapshotOptions opts;
   opts.fOutputFormat = ROOT::RDF::ESnapshotOutputFormat::kRNTuple;

   auto sdf = df.Snapshot("ntuple", fileGuard.GetPath(), columns, opts);

   EXPECT_EQ(columns, sdf->GetColumnNames());

   // Verify we actually snapshotted to an RNTuple.
   auto ntuple = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   EXPECT_EQ(25ull, ntuple->GetNEntries());

   auto x = ntuple->GetView<int>("x");
   for (const auto i : ntuple->GetEntryRange()) {
      EXPECT_EQ(10, x(i));
   }
}

TEST(RDFSnapshotRNTuple, FromScratchJITted)
{
   FileRAII fileGuard{"RDFSnapshotRNTuple_from_scratch_jitted.root"};
   const std::vector<std::string> columns = {"x"};

   auto df = ROOT::RDataFrame(25ull).Define("x", [] { return 10; });

   RSnapshotOptions opts;
   opts.fOutputFormat = ROOT::RDF::ESnapshotOutputFormat::kRNTuple;

   auto sdf = df.Snapshot("ntuple", fileGuard.GetPath(), "x", opts);

   EXPECT_EQ(columns, sdf->GetColumnNames());

   // Verify we actually snapshotted to an RNTuple.
   auto ntuple = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   EXPECT_EQ(25ull, ntuple->GetNEntries());

   auto x = ntuple->GetView<int>("x");
   for (const auto i : ntuple->GetEntryRange()) {
      EXPECT_EQ(10, x(i));
   }
}

void BookLazySnapshot(std::string_view filename)
{
   auto d = ROOT::RDataFrame(1);
   ROOT::RDF::RSnapshotOptions opts;
   opts.fOutputFormat = ROOT::RDF::ESnapshotOutputFormat::kRNTuple;
   opts.fLazy = true;
   d.Snapshot("t", filename, {"rdfentry_"}, opts);
}

TEST(RDFSnapshotRNTuple, LazyNotTriggered)
{
   FileRAII fileGuard{"lazysnapshotnottriggered_shouldnotbecreated.root"};
   ROOT_EXPECT_WARNING(BookLazySnapshot(fileGuard.GetPath()), "Snapshot",
                       "A lazy Snapshot action was booked but never triggered.");
   EXPECT_TRUE(gSystem->AccessPathName(fileGuard.GetPath().c_str()));
}

TEST(RDFSnapshotRNTuple, Compression)
{
   FileRAII fileGuard{"RDFSnapshotRNTuple_compression.root"};
   const std::vector<std::string> columns = {"x"};

   auto df = ROOT::RDataFrame(25ull).Define("x", [] { return 10; });

   RSnapshotOptions opts;
   opts.fOutputFormat = ROOT::RDF::ESnapshotOutputFormat::kRNTuple;
   opts.fCompressionAlgorithm = ROOT::RCompressionSetting::EAlgorithm::kLZ4;
   opts.fCompressionLevel = 4;

   auto sdf = df.Snapshot("ntuple", fileGuard.GetPath(), "x", opts);

   EXPECT_EQ(columns, sdf->GetColumnNames());

   auto reader = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   auto compSettings = *reader->GetDescriptor().GetClusterDescriptor(0).GetColumnRange(0).GetCompressionSettings();
   EXPECT_EQ(404, compSettings);
}

class RDFSnapshotRNTupleTest : public ::testing::Test {
protected:
   const std::string fFileName = "RDFSnapshotRNTuple.root";
   const std::string fNtplName = "ntuple";
   RSnapshotOptions fSnapshotOpts;

   void SetUp() override
   {
      fSnapshotOpts.fOutputFormat = ESnapshotOutputFormat::kRNTuple;

      auto model = RNTupleModel::Create();
      *model->MakeField<float>("pt") = 42.f;
      *model->MakeField<std::string>("tag") = "xyz";
      auto fldNnlo = model->MakeField<std::vector<std::vector<float>>>("nnlo");
      fldNnlo->push_back(std::vector<float>());
      fldNnlo->push_back(std::vector<float>{1.0});
      fldNnlo->push_back(std::vector<float>{1.0, 2.0, 4.0, 8.0});
      *model->MakeField<ROOT::RVecI>("rvec") = ROOT::RVecI{1, 2, 3};
      auto fldElectron = model->MakeField<Electron>("electron");
      fldElectron->pt = 137.0;
      auto fldElectrons = model->MakeField<std::vector<Electron>>("electrons");
      fldElectrons->push_back(*fldElectron);
      fldElectrons->push_back(*fldElectron);
      auto fldJets = model->MakeField<std::vector<Jet>>("jets");
      fldJets->push_back(Jet{*fldElectrons});
      {
         auto ntuple = RNTupleWriter::Recreate(std::move(model), fNtplName, fFileName);
         ntuple->Fill();
      }
   }

   void TearDown() override { std::remove(fFileName.c_str()); }
};

void SnapshotToDefaultOutput(ROOT::RDataFrame &df, std::string_view filename)
{
   df.Define("x", [] { return 10; }).Snapshot("ntuple", filename, {"pt", "x"});
}

TEST_F(RDFSnapshotRNTupleTest, DefaultFormatWarning)
{
   FileRAII fileGuard{"RDFSnapshotRNTuple_snap_default_format_warning.root"};

   auto df = ROOT::RDataFrame(fNtplName, fFileName);

   ROOT_EXPECT_WARNING(SnapshotToDefaultOutput(df, fileGuard.GetPath()), "Snapshot",
                       "The default Snapshot output data format is TTree, but the input data format is RNTuple. If you "
                       "want to Snapshot to RNTuple or suppress this warning, set the appropriate fOutputFormat option "
                       "in RSnapshotOptions. Note that this current default behaviour might change in the future.");
}

TEST_F(RDFSnapshotRNTupleTest, DefaultToRNTupleTemplated)
{
   FileRAII fileGuard{"RDFSnapshotRNTuple_snap_templated.root"};

   auto df = ROOT::RDataFrame(fNtplName, fFileName);
   auto sdf = df.Define("x", [] { return 10; }).Snapshot("ntuple", fileGuard.GetPath(), {"pt", "x"}, fSnapshotOpts);

   auto ntuple = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   EXPECT_EQ(1ull, ntuple->GetNEntries());

   auto pt = ntuple->GetView<float>("pt");
   auto x = ntuple->GetView<int>("x");

   EXPECT_FLOAT_EQ(42.0, pt(0));
   EXPECT_EQ(10, x(0));
}

TEST_F(RDFSnapshotRNTupleTest, DefaultToRNTupleJITted)
{
   FileRAII fileGuard{"RDFSnapshotRNTuple_snap_jitted.root"};

   auto df = ROOT::RDataFrame(fNtplName, fFileName);
   auto sdf = df.Define("x", [] { return 10; }).Snapshot("ntuple", fileGuard.GetPath(), {"pt", "x"}, fSnapshotOpts);

   auto ntuple = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   EXPECT_EQ(1ull, ntuple->GetNEntries());

   auto pt = ntuple->GetView<float>("pt");
   auto x = ntuple->GetView<int>("x");

   EXPECT_FLOAT_EQ(42.0, pt(0));
   EXPECT_EQ(10, x(0));
}

TEST_F(RDFSnapshotRNTupleTest, ToTTreeTemplated)
{
   FileRAII fileGuard{"RDFSnapshotRNTuple_to_ttree_templated.root"};

   auto df = ROOT::RDataFrame(fNtplName, fFileName);

   fSnapshotOpts.fOutputFormat = ROOT::RDF::ESnapshotOutputFormat::kTTree;

   auto sdf = df.Define("x", [] { return 10; }).Snapshot("tree", fileGuard.GetPath(), {"pt", "x"}, fSnapshotOpts);

   TFile file(fileGuard.GetPath().c_str());
   auto tree = file.Get<TTree>("tree");
   EXPECT_EQ(1ull, tree->GetEntries());

   float pt;
   int x;

   tree->SetBranchAddress("pt", &pt);
   tree->SetBranchAddress("x", &x);

   tree->GetEntry(0);

   EXPECT_FLOAT_EQ(42.0, pt);
   EXPECT_EQ(10, x);
}

TEST_F(RDFSnapshotRNTupleTest, ToTTreeJITted)
{
   FileRAII fileGuard{"RDFSnapshotRNTuple_to_ttree_jitted.root"};

   auto df = ROOT::RDataFrame(fNtplName, fFileName);

   fSnapshotOpts.fOutputFormat = ROOT::RDF::ESnapshotOutputFormat::kTTree;

   auto sdf = df.Define("x", [] { return 10; }).Snapshot("tree", fileGuard.GetPath(), {"pt", "x"}, fSnapshotOpts);

   TFile file(fileGuard.GetPath().c_str());
   auto tree = file.Get<TTree>("tree");
   EXPECT_EQ(1ull, tree->GetEntries());

   float pt;
   int x;

   tree->SetBranchAddress("pt", &pt);
   tree->SetBranchAddress("x", &x);

   tree->GetEntry(0);

   EXPECT_FLOAT_EQ(42.0, pt);
   EXPECT_EQ(10, x);
}

TEST_F(RDFSnapshotRNTupleTest, ScalarFields)
{
   FileRAII fileGuard{"RDFSnapshotRNTuple_scalar_fields.root"};
   auto df = ROOT::RDataFrame(fNtplName, fFileName);
   auto sdf = df.Snapshot("ntuple", fileGuard.GetPath(), "pt", fSnapshotOpts);

   // Verify we actually snapshotted to an RNTuple.
   auto ntuple = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   EXPECT_EQ(1ull, ntuple->GetNEntries());

   std::vector<std::string> expected = {"pt"};
   EXPECT_EQ(expected, sdf->GetColumnNames());

   auto maxPt = sdf->Max<float>("pt").GetValue();
   EXPECT_FLOAT_EQ(42.f, maxPt);
}

TEST_F(RDFSnapshotRNTupleTest, VectorFields)
{
   FileRAII fileGuard{"RDFSnapshotRNTuple_vector_fields.root"};
   auto df = ROOT::RDataFrame(fNtplName, fFileName);
   auto sdf = df.Snapshot("ntuple", fileGuard.GetPath(), "nnlo", fSnapshotOpts);

   // Verify we actually snapshotted to an RNTuple.
   auto ntuple = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   EXPECT_EQ(1ull, ntuple->GetNEntries());

   std::vector<std::string> expected = {"nnlo"};
   EXPECT_EQ(expected, sdf->GetColumnNames());

   auto nnloMax = sdf->Define("nnloMax",
                              [](const ROOT::RVec<ROOT::RVec<float>> &nnlo) {
                                 auto innerMax = ROOT::VecOps::Map(
                                    nnlo, [](ROOT::RVec<float> innerNnlo) { return ROOT::VecOps::Max(innerNnlo); });
                                 return ROOT::VecOps::Max(innerMax);
                              },
                              {"nnlo"})
                     .Max("nnloMax")
                     .GetValue();
   EXPECT_FLOAT_EQ(8.f, nnloMax);
}

TEST_F(RDFSnapshotRNTupleTest, ComplexFields)
{
   FileRAII fileGuard{"RDFSnapshotRNTuple_complex_fields.root"};
   auto df = ROOT::RDataFrame(fNtplName, fFileName);
   auto sdf = df.Snapshot("ntuple", fileGuard.GetPath(), "electrons", fSnapshotOpts);

   // Verify we actually snapshotted to an RNTuple.
   auto ntuple = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   EXPECT_EQ(1ull, ntuple->GetNEntries());

   std::vector<std::string> expected = {"electrons", "electrons.pt"};
   EXPECT_EQ(expected, sdf->GetColumnNames());

   auto electronsPtMax =
      sdf->Define("electronsPtMax", [](const ROOT::RVec<float> &electronPt) { return ROOT::VecOps::Max(electronPt); },
                  {"electrons.pt"})
         .Max("electronsPtMax")
         .GetValue();
   EXPECT_FLOAT_EQ(137.f, electronsPtMax);
}

TEST_F(RDFSnapshotRNTupleTest, InnerFields)
{
   FileRAII fileGuard{"RDFSnapshotRNTuple_inner_fields.root"};

   auto df = ROOT::RDataFrame(fNtplName, fFileName);
   auto sdf1 = df.Snapshot("ntuple", fileGuard.GetPath(), "electron.pt", fSnapshotOpts);

   // Verify we actually snapshotted to an RNTuple.
   auto ntuple1 = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   EXPECT_EQ(1ull, ntuple1->GetNEntries());

   std::vector<std::string> expected = {"electron_pt"};
   EXPECT_EQ(expected, sdf1->GetColumnNames());

   auto electronPtMax = sdf1->Max("electron_pt").GetValue();
   EXPECT_FLOAT_EQ(137.f, electronPtMax);

   auto sdf2 = df.Snapshot("ntuple", fileGuard.GetPath(), "jets.electrons", fSnapshotOpts);

   // Verify we actually snapshotted to an RNTuple.
   auto ntuple2 = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   EXPECT_EQ(1ull, ntuple2->GetNEntries());

   expected = {"jets_electrons", "jets_electrons.pt"};
   EXPECT_EQ(expected, sdf2->GetColumnNames());

   auto jetsElectronsPtMax =
      sdf2
         ->Define("jetsElectronsPtMax",
                  [](const ROOT::RVec<ROOT::RVec<float>> &jetsElectronsPt) {
                     auto innerMax = ROOT::VecOps::Map(jetsElectronsPt, [](const ROOT::RVec<float> &electronsPt) {
                        return ROOT::VecOps::Max(electronsPt);
                     });
                     return ROOT::VecOps::Max(innerMax);
                  },
                  {"jets_electrons.pt"})
         .Max("jetsElectronsPtMax")
         .GetValue();
   EXPECT_FLOAT_EQ(137.f, jetsElectronsPtMax);
}

TEST_F(RDFSnapshotRNTupleTest, AllFields)
{
   FileRAII fileGuard{"RDFSnapshotRNTuple_all_fields.root"};

   auto df = ROOT::RDataFrame(fNtplName, fFileName);
   auto sdf = df.Snapshot("ntuple", fileGuard.GetPath(), "", fSnapshotOpts);

   // Verify we actually snapshotted to an RNTuple.
   auto ntuple = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   EXPECT_EQ(1ull, ntuple->GetNEntries());

   EXPECT_EQ(df.GetColumnNames(), sdf->GetColumnNames());
}

TEST_F(RDFSnapshotRNTupleTest, WithDefines)
{
   FileRAII fileGuard{"RDFSnapshotRNTuple_with_defines.root"};

   auto df = ROOT::RDataFrame(fNtplName, fFileName);
   auto sdf = df.Define("x", [] { return 10; }).Snapshot("ntuple", fileGuard.GetPath(), "", fSnapshotOpts);

   // Verify we actually snapshotted to an RNTuple.
   auto ntuple = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   EXPECT_EQ(1ull, ntuple->GetNEntries());

   std::vector<std::string> expected = df.GetColumnNames();
   expected.push_back("x");
   EXPECT_EQ(expected, sdf->GetColumnNames());
}

TEST(RDFSnapshotRNTuple, WithFilters)
{
   FileRAII fileGuardInput{"RDFSnapshotRNTuple_defines_and_filters.root"};
   FileRAII fileGuardSnapshot{"snap_ntuple_filtered.root"};

   RSnapshotOptions opts;
   opts.fOutputFormat = ROOT::RDF::ESnapshotOutputFormat::kRNTuple;

   {
      auto df = ROOT::RDataFrame(10ull).DefineSlotEntry("x", [](unsigned int, std::uint64_t entry) { return entry; });
      df.Snapshot("ntuple", fileGuardInput.GetPath(), "x", opts);
   }

   auto df = ROOT::RDataFrame("ntuple", fileGuardInput.GetPath()).Filter("x % 2 == 0");
   auto sdf = df.Snapshot("ntuple", fileGuardSnapshot.GetPath(), "", opts);

   auto ntuple = RNTupleReader::Open("ntuple", fileGuardSnapshot.GetPath());
   EXPECT_EQ(5ull, ntuple->GetNEntries());

   auto x = ntuple->GetView<std::uint64_t>("x");
   for (const auto i : ntuple->GetEntryRange()) {
      EXPECT_FLOAT_EQ(i * 2, x(i));
   }
}

TEST(RDFSnapshotRNTuple, UpdateDifferentName)
{
   FileRAII fileGuard{"RDFSnapshotRNTuple_update_different_name.root"};

   RSnapshotOptions opts;
   opts.fOutputFormat = ROOT::RDF::ESnapshotOutputFormat::kRNTuple;

   {
      auto df = ROOT::RDataFrame(25ull).Define("x", [] { return 10; });
      auto sdf = df.Snapshot("ntuple", fileGuard.GetPath(), "x", opts);
   }

   auto df = ROOT::RDataFrame("ntuple", fileGuard.GetPath());

   opts.fMode = "UPDATE";

   auto sdf = df.Define("y", [] { return 42; }).Snapshot("ntuple_snap", fileGuard.GetPath(), "", opts);

   std::vector<std::string> expected = {"x", "y"};
   EXPECT_EQ(expected, sdf->GetColumnNames());

   auto ntupleOriginal = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   EXPECT_EQ(25ull, ntupleOriginal->GetNEntries());

   auto ntupleSnap = RNTupleReader::Open("ntuple_snap", fileGuard.GetPath());
   EXPECT_EQ(25ull, ntupleSnap->GetNEntries());
}

TEST(RDFSnapshotRNTuple, UpdateSameName)
{
   FileRAII fileGuard{"RDFSnapshotRNTuple_update_same_name.root"};

   RSnapshotOptions opts;
   opts.fOutputFormat = ROOT::RDF::ESnapshotOutputFormat::kRNTuple;

   {
      auto df = ROOT::RDataFrame(25ull).Define("x", [] { return 10; });
      auto sdf = df.Snapshot("ntuple", fileGuard.GetPath(), "x", opts);

      // Verify we actually snapshotted to an RNTuple.
      auto ntuple = RNTupleReader::Open("ntuple", fileGuard.GetPath());
      EXPECT_EQ(25ull, ntuple->GetNEntries());
   }

   auto df = ROOT::RDataFrame("ntuple", fileGuard.GetPath());

   opts.fMode = "UPDATE";

   try {
      auto sdf = df.Define("y", [] { return 42; }).Snapshot("ntuple", fileGuard.GetPath(), {"x", "y"}, opts);
      FAIL() << "snapshotting in \"UPDATE\" mode to the same ntuple name without `fOverwriteIfExists` is not allowed ";
   } catch (const std::invalid_argument &err) {
      EXPECT_STREQ(err.what(),
                   "Snapshot: RNTuple \"ntuple\" already present in file "
                   "\"RDFSnapshotRNTuple_update_same_name.root\". If you want to delete the original "
                   "ntuple and write another, please set the 'fOverwriteIfExists' option to true in RSnapshotOptions.");
   }

   opts.fOverwriteIfExists = true;
   auto sdf = df.Define("y", [] { return 42; }).Snapshot("ntuple", fileGuard.GetPath(), "", opts);

   // Verify we actually snapshotted to an RNTuple.
   auto ntuple = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   EXPECT_EQ(25ull, ntuple->GetNEntries());

   std::vector<std::string> expected = {"x", "y"};
   EXPECT_EQ(expected, sdf->GetColumnNames());
}

TEST(RDFSnapshotRNTuple, TDirectory)
{
   FileRAII fileGuard{"RDFSnapshotRNTuple_snap_tdirectory.root"};

   auto df = ROOT::RDataFrame(1);

   RSnapshotOptions opts;
   opts.fOutputFormat = ESnapshotOutputFormat::kRNTuple;

   df.Define("x", [] { return 10; }).Snapshot("dir/ntuple", fileGuard.GetPath(), {"x"}, opts);

   // Check that we can open the snapshotted file through RNTupleReader...
   auto ntuple = RNTupleReader::Open("dir/ntuple", fileGuard.GetPath());
   EXPECT_EQ(1ull, ntuple->GetNEntries());

   // ... and also create an RDF from scratch with it
   auto sdf = ROOT::RDataFrame("dir/ntuple", fileGuard.GetPath());
   std::vector<std::string> expected = {"x"};
   EXPECT_EQ(expected, sdf.GetColumnNames());
}

void WriteTestTree(const std::string &tname, const std::string &fname)
{
   TFile file(fname.c_str(), "RECREATE");
   TTree t(tname.c_str(), tname.c_str());
   float pt;
   t.Branch("pt", &pt);

   pt = 42.0;
   t.Fill();

   t.Write();
}

TEST(RDFSnapshotRNTuple, DisallowFromTTreeTemplated)
{
   const auto treename = "tree";
   FileRAII fileGuard{"RDFSnapshotRNTuple_disallow_from_ttree_templated.root"};

   WriteTestTree(treename, fileGuard.GetPath());

   auto df = ROOT::RDataFrame(treename, fileGuard.GetPath());

   RSnapshotOptions opts;
   opts.fOutputFormat = ROOT::RDF::ESnapshotOutputFormat::kRNTuple;

   try {
      auto sdf = df.Define("x", [] { return 10; }).Snapshot("ntuple", fileGuard.GetPath(), {"pt", "x"}, opts);
      FAIL() << "snapshotting from RNTuple to TTree is not (yet) possible";
   } catch (const std::runtime_error &err) {
      EXPECT_STREQ(err.what(), "Snapshotting from TTree to RNTuple is not yet supported. The current recommended way "
                               "to convert TTrees to RNTuple is through the RNTupleImporter.");
   }
}

TEST(RDFSnapshotRNTuple, DisallowFromTTreeJITted)
{
   const auto treename = "tree";
   FileRAII fileGuard{"RDFSnapshotRNTuple_disallow_from_ttree_jitted.root"};

   WriteTestTree(treename, fileGuard.GetPath());

   auto df = ROOT::RDataFrame(treename, fileGuard.GetPath());

   RSnapshotOptions opts;
   opts.fOutputFormat = ROOT::RDF::ESnapshotOutputFormat::kRNTuple;

   try {
      auto sdf = df.Define("x", [] { return 10; }).Snapshot("ntuple", fileGuard.GetPath(), {"pt", "x"}, opts);
      FAIL() << "snapshotting from RNTuple to TTree is not (yet) possible";
   } catch (const std::runtime_error &err) {
      EXPECT_STREQ(err.what(), "Snapshotting from TTree to RNTuple is not yet supported. The current recommended way "
                               "to convert TTrees to RNTuple is through the RNTupleImporter.");
   }
}

#ifdef R__USE_IMT
struct TIMTEnabler {
   TIMTEnabler(unsigned int nSlots) { ROOT::EnableImplicitMT(nSlots); }
   ~TIMTEnabler() { ROOT::DisableImplicitMT(); }
};

TEST(RDFSnapshotRNTuple, ThrowIfMT)
{
   TIMTEnabler _(4);

   FileRAII fileGuard{"RDFSnapshotRNTuple_throw_if_mt.root"};

   auto df = ROOT::RDataFrame(25ull).Define("x", [] { return 10; });

   RSnapshotOptions opts;
   opts.fOutputFormat = ROOT::RDF::ESnapshotOutputFormat::kRNTuple;

   try {
      auto sdf = df.Snapshot("ntuple", fileGuard.GetPath(), {"x"}, opts);
      *sdf;
      FAIL() << "MT snapshotting to RNTuple is not supported yet";
   } catch (const std::runtime_error &err) {
      EXPECT_STREQ(err.what(), "Snapshot: Snapshotting to RNTuple with IMT enabled is not supported yet.");
   }
}
#endif // R__USE_IMT
