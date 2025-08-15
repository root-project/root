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
#include <TTree.h>

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

template <typename T>
void expect_vec_eq(const ROOT::RVec<T> &v1, const ROOT::RVec<T> &v2)
{
   ASSERT_EQ(v1.size(), v2.size()) << "Vectors 'v1' and 'v2' are of unequal length";
   for (std::size_t i = 0ull; i < v1.size(); ++i) {
      if constexpr (std::is_floating_point_v<T>)
         EXPECT_FLOAT_EQ(v1[i], v2[i]) << "Vectors 'v1' and 'v2' differ at index " << i;
      else
         EXPECT_EQ(v1[i], v2[i]) << "Vectors 'v1' and 'v2' differ at index " << i;
   }
}

TEST(RDFSnapshotRNTuple, FromScratch)
{
   FileRAII fileGuard{"RDFSnapshotRNTuple_from_scratch.root"};
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

TEST(RDFSnapshotRNTuple, LazyTriggered)
{
   FileRAII fileGuard{"RDFSnapshotRNTuple_lazy.root"};
   auto d = ROOT::RDataFrame(1);
   ROOT::RDF::RSnapshotOptions opts;
   opts.fOutputFormat = ROOT::RDF::ESnapshotOutputFormat::kRNTuple;
   opts.fLazy = true;
   auto r = d.Snapshot("t", fileGuard.GetPath(), {"rdfentry_"}, opts);
   *r;
   r = {};
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

TEST_F(RDFSnapshotRNTupleTest, DefaultToRNTuple)
{
   FileRAII fileGuard{"RDFSnapshotRNTuple_snap.root"};

   auto df = ROOT::RDataFrame(fNtplName, fFileName);
   auto sdf = df.Define("x", [] { return 10; }).Snapshot("ntuple", fileGuard.GetPath(), {"pt", "x"}, fSnapshotOpts);

   auto ntuple = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   EXPECT_EQ(1ull, ntuple->GetNEntries());

   auto pt = ntuple->GetView<float>("pt");
   auto x = ntuple->GetView<int>("x");

   EXPECT_FLOAT_EQ(42.0, pt(0));
   EXPECT_EQ(10, x(0));
}

TEST_F(RDFSnapshotRNTupleTest, ToTTree)
{
   FileRAII fileGuard{"RDFSnapshotRNTuple_to_ttree.root"};

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

   std::vector<std::string> expected = {"electron_pt"};

   auto df = ROOT::RDataFrame(fNtplName, fFileName);
   {
      auto sdf1 = df.Snapshot("ntuple", fileGuard.GetPath(), "electron.pt", fSnapshotOpts);

      // Verify we actually snapshotted to an RNTuple.
      auto ntuple1 = RNTupleReader::Open("ntuple", fileGuard.GetPath());
      EXPECT_EQ(1ull, ntuple1->GetNEntries());

      EXPECT_EQ(expected, sdf1->GetColumnNames());

      auto electronPtMax = sdf1->Max("electron_pt").GetValue();
      EXPECT_FLOAT_EQ(137.f, electronPtMax);
   }
   {
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

class RDFSnapshotRNTupleFromTTreeTest : public ::testing::Test {
protected:
   const std::string fFileName = "RDFSnapshotRNTuple_ttree_fixture.root";
   const std::string fTreeName = "tree";

   void SetUp() override
   {
      TFile file(fFileName.c_str(), "RECREATE");
      TTree t(fTreeName.c_str(), fTreeName.c_str());

      float pt = 42.f;
      std::vector<float> photons{1.f, 2.f, 3.f};
      Electron electron{137.f};
      Jet jets;
      jets.electrons.emplace_back(Electron{122.f});
      jets.electrons.emplace_back(Electron{125.f});
      jets.electrons.emplace_back(Electron{129.f});

      Int_t nmuons = 1;
      float muon_pt[3] = {10.f, 20.f, 30.f};

      struct {
         Int_t x = 1;
         Int_t y = 2;
      } point;

      t.Branch("pt", &pt);
      t.Branch("photons", &photons);
      t.Branch("electron", &electron);
      t.Branch("jets", &jets);
      t.Branch("nmuons", &nmuons);
      t.Branch("muon_pt", muon_pt, "muon_pt[nmuons]");
      t.Branch("point", &point, "x/I:y/I");

      t.Fill();
      t.Write();
   }

   void TearDown() override { gSystem->Unlink(fFileName.c_str()); }

   void TestFromTTree(const std::string &fname, bool vector2RVec = false)
   {
      FileRAII fileGuard{fname};

      auto df = ROOT::RDataFrame(fTreeName, fFileName);

      {
         RSnapshotOptions opts;
         opts.fOutputFormat = ROOT::RDF::ESnapshotOutputFormat::kRNTuple;
         opts.fVector2RVec = vector2RVec;

         // FIXME(fdegeus): snapshotting leaflist branches as-is (i.e. without explicitly providing their leafs) is not
         // supported, because we have no way of reconstructing the memory layout of the branch itself from only the
         // TTree's on-disk information without JITting. For RNTuple, we would be able to do this using anonymous record
         // fields, however. Once this is implemented, this test should be changed to check the result of snapshotting
         // "point" fully.
         auto sdf = df.Define("x", [] { return 10; })
                       .Snapshot("ntuple", fileGuard.GetPath(),
                                 {"x", "pt", "photons", "electron", "jets", "muon_pt", "point.x", "point.y"}, opts);

         auto x = sdf->Take<int>("x");
         auto pt = sdf->Take<float>("pt");
         auto photons = sdf->Take<ROOT::RVec<float>>("photons");
         auto electron = sdf->Take<Electron>("electron");
         auto jet_electrons = sdf->Take<ROOT::RVec<Electron>>("jets.electrons");
         auto nMuons = sdf->Take<int>("nmuons");
         auto muonPt = sdf->Take<ROOT::RVec<float>>("muon_pt");
         auto pointX = sdf->Take<int>("point_x");
         auto pointY = sdf->Take<int>("point_y");

         ASSERT_EQ(1UL, x->size());
         ASSERT_EQ(1UL, pt->size());
         ASSERT_EQ(1UL, photons->size());
         ASSERT_EQ(1UL, electron->size());
         ASSERT_EQ(1UL, jet_electrons->size());
         ASSERT_EQ(1UL, nMuons->size());
         ASSERT_EQ(1UL, muonPt->size());
         ASSERT_EQ(1UL, pointX->size());
         ASSERT_EQ(1UL, pointY->size());

         EXPECT_EQ(10, x->front());
         EXPECT_EQ(42.f, pt->front());
         expect_vec_eq<float>({1.f, 2.f, 3.f}, photons->front());
         EXPECT_EQ(Electron{137.f}, electron->front());
         expect_vec_eq({Electron{122.f}, Electron{125.f}, Electron{129.f}}, jet_electrons->front());
         EXPECT_EQ(1, nMuons->front());
         expect_vec_eq({10.f}, muonPt->front());
         EXPECT_EQ(1, pointX->front());
         EXPECT_EQ(2, pointY->front());
      }

      auto reader = RNTupleReader::Open("ntuple", fileGuard.GetPath());

      auto &descriptor = reader->GetDescriptor();

      int nTopLevelFields = std::distance(descriptor.GetTopLevelFields().begin(), descriptor.GetTopLevelFields().end());
      EXPECT_EQ(9, nTopLevelFields);
      EXPECT_EQ("std::int32_t", descriptor.GetFieldDescriptor(descriptor.FindFieldId("x")).GetTypeName());
      EXPECT_EQ("float", descriptor.GetFieldDescriptor(descriptor.FindFieldId("pt")).GetTypeName());
      if (vector2RVec) {
         EXPECT_EQ("ROOT::VecOps::RVec<float>",
                   descriptor.GetFieldDescriptor(descriptor.FindFieldId("photons")).GetTypeName());
      } else {
         EXPECT_EQ("std::vector<float>",
                   descriptor.GetFieldDescriptor(descriptor.FindFieldId("photons")).GetTypeName());
      }
      EXPECT_EQ("Electron", descriptor.GetFieldDescriptor(descriptor.FindFieldId("electron")).GetTypeName());
      auto jetsId = descriptor.FindFieldId("jets");
      EXPECT_EQ("Jet", descriptor.GetFieldDescriptor(jetsId).GetTypeName());
      auto electronsId = descriptor.FindFieldId("electrons", jetsId);
      EXPECT_EQ("std::vector<Electron>", descriptor.GetFieldDescriptor(electronsId).GetTypeName());
      EXPECT_EQ("std::int32_t", descriptor.GetFieldDescriptor(descriptor.FindFieldId("nmuons")).GetTypeName());
      EXPECT_EQ("ROOT::VecOps::RVec<float>",
                descriptor.GetFieldDescriptor(descriptor.FindFieldId("muon_pt")).GetTypeName());
      EXPECT_EQ("std::int32_t", descriptor.GetFieldDescriptor(descriptor.FindFieldId("point_x")).GetTypeName());
      EXPECT_EQ("std::int32_t", descriptor.GetFieldDescriptor(descriptor.FindFieldId("point_y")).GetTypeName());
      // sanity check to make sure we don't snapshot the internal RDF size columns
      EXPECT_EQ(ROOT::kInvalidDescriptorId, descriptor.FindFieldId("R_rdf_sizeof_photons"));

      auto x = reader->GetView<int>("x");
      auto pt = reader->GetView<float>("pt");
      auto photons = reader->GetView<ROOT::RVec<float>>("photons");
      auto electron = reader->GetView<Electron>("electron");
      auto jet_electrons = reader->GetView<ROOT::RVec<Electron>>("jets.electrons");
      auto nMuons = reader->GetView<int>("nmuons");
      auto muonPt = reader->GetView<ROOT::RVec<float>>("muon_pt");
      auto pointX = reader->GetView<int>("point_x");
      auto pointY = reader->GetView<int>("point_y");

      EXPECT_EQ(10, x(0));
      EXPECT_EQ(42.f, pt(0));
      expect_vec_eq<float>({1.f, 2.f, 3.f}, photons(0));
      EXPECT_EQ(Electron{137.f}, electron(0));
      expect_vec_eq({Electron{122.f}, Electron{125.f}, Electron{129.f}}, jet_electrons(0));
      EXPECT_EQ(1, nMuons(0));
      expect_vec_eq({10.f}, muonPt(0));
      EXPECT_EQ(1, pointX(0));
      EXPECT_EQ(2, pointY(0));
   }
};

TEST_F(RDFSnapshotRNTupleFromTTreeTest, FromTTree)
{
   TestFromTTree("RDFSnapshotRNTuple_from_ttree.root");
}

TEST_F(RDFSnapshotRNTupleFromTTreeTest, FromTTreeNoVector2RVec)
{
   TestFromTTree("RDFSnapshotRNTuple_from_ttree_novec2rvec.root", /*vector2RVec=*/false);
}

#ifdef R__USE_IMT
struct TIMTEnabler {
   TIMTEnabler(unsigned int nSlots) { ROOT::EnableImplicitMT(nSlots); }
   ~TIMTEnabler() { ROOT::DisableImplicitMT(); }
};

TEST(RDFSnapshotRNTuple, WithMT)
{
   TIMTEnabler _(4);

   FileRAII fileGuard{"RDFSnapshotRNTuple_mt.root"};

   auto df = ROOT::RDataFrame(25ull).Define("x", [](ULong64_t e) { return e; }, {"rdfentry_"});

   RSnapshotOptions opts;
   opts.fOutputFormat = ROOT::RDF::ESnapshotOutputFormat::kRNTuple;

   auto sdf = df.Snapshot("ntuple", fileGuard.GetPath(), {"x"}, opts);
   *sdf;

   auto sum = sdf->Sum<std::uint64_t>("x");
   EXPECT_EQ(300, sum.GetValue());

   auto reader = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   EXPECT_EQ(25, reader->GetNEntries());
   // There should be more than one cluster, but this is not guaranteed because of scheduling...
}
#endif // R__USE_IMT
