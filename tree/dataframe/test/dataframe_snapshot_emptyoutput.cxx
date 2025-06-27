#include "ROOT/TestSupport.hxx"
#include "gtest/gtest.h"

// We only use this header to generate the dictionary for RVec<ROOT::Math::PtEtaPhiMVector>
#include "DummyHeader.hxx"

#include <ROOT/InternalTreeUtils.hxx>
#include <ROOT/RDataFrame.hxx>

#include <thread>

#include <TFile.h>

class RDFManyBranchTypesTTree : public ::testing::TestWithParam<std::pair<bool, bool>> {
protected:
   inline constexpr static auto fgNEvents = 10u;
   inline constexpr static auto fgFileName{"rdfmanybranchtypesttree.root"};
   inline constexpr static auto fgTreeName{"manybranchtypes"};
   unsigned int fNSlots;

   RDFManyBranchTypesTTree() : fNSlots(GetParam().first ? std::min(4u, std::thread::hardware_concurrency()) : 1u)
   {
      if (GetParam().first)
         ROOT::EnableImplicitMT(fNSlots);

      auto curEvent = 0u;
      TFile f(fgFileName, "recreate");
      TTree t(fgTreeName, fgTreeName);

      // doubles, floats
      const unsigned int fixedSize = 4u;
      float fixedSizeArr[fixedSize];
      t.Branch("fixedSizeArr", fixedSizeArr, ("fixedSizeArr[" + std::to_string(fixedSize) + "]/F").c_str());
      unsigned int size = 0u;
      t.Branch("size", &size);
      double *varSizeArr = new double[fgNEvents * 100u];
      t.Branch("varSizeArr", varSizeArr, "varSizeArr[size]/D");

      // bools. std::vector<bool> makes bool treatment in RDF special
      bool fixedSizeBoolArr[fixedSize];
      t.Branch("fixedSizeBoolArr", fixedSizeBoolArr, ("fixedSizeBoolArr[" + std::to_string(fixedSize) + "]/O").c_str());
      bool *varSizeBoolArr = new bool[fgNEvents * 100u];
      t.Branch("varSizeBoolArr", varSizeBoolArr, "varSizeBoolArr[size]/O");

      // std::vector
      std::vector<std::int64_t> veci{};
      t.Branch("veci", &veci);

      // ROOT::RVec
      ROOT::RVecI rveci{};
      t.Branch("rveci", &rveci);

      // Custom class, unsplit
      ROOT::RVec<ROOT::Math::PtEtaPhiMVector> lorentzVectors{};
      t.Branch("lorentzVectorsUnsplit", &lorentzVectors, 32000, 0);

      // Custom class, default
      ROOT::Math::PtEtaPhiMVector ptetaphimvector{11., 22., 33., 44.};
      t.Branch("ptetaphimvector", &ptetaphimvector);

      // We want to test the case where the application
      // gets an empty TTree as input, with the branches set up.
      if (GetParam().second) {
         for (auto i : ROOT::TSeqU(fgNEvents)) {

            // fixed-size arrays
            for (auto j : ROOT::TSeqU(4)) {
               fixedSizeArr[j] = curEvent * j;
               fixedSizeBoolArr[j] = j % 2 == 0;
            }

            // dynamic-sized arrays
            size = (i + 1) * 100u;
            for (auto j : ROOT::TSeqU(size)) {
               varSizeArr[j] = curEvent * j;
               varSizeBoolArr[j] = j % 2 == 0;
            }

            // std::vector, ROOT::RVec
            veci.clear();
            rveci.clear();
            for (auto k : ROOT::TSeqU(i)) {
               veci.push_back(k);
               rveci.push_back(k);
               lorentzVectors.push_back(ROOT::Math::PtEtaPhiMVector{1.f, 2.f, 3.f, 4.f});
            }

            t.Fill();
            ++curEvent;
         }
      }

      f.Write();

      delete[] varSizeArr;
      delete[] varSizeBoolArr;
   }
   ~RDFManyBranchTypesTTree() override
   {
      if (GetParam().first)
         ROOT::DisableImplicitMT();

      std::remove(fgFileName);
   }
};

template <typename T0, typename T1>
void expect_vec_eq(const T0 &v1, const T1 &v2)
{
   ASSERT_EQ(v1.size(), v2.size()) << "Vectors 'v1' and 'v2' are of unequal length";
   for (std::size_t i = 0ull; i < v1.size(); ++i) {
      EXPECT_EQ(v1[i], v2[i]) << "Vectors 'v1' and 'v2' differ at index " << i;
   }
}

TEST_P(RDFManyBranchTypesTTree, SnapshotEmptyTTree)
{
   const auto outputTreeNameExpected{"outTreeExpected"};
   const auto outputFileNameExpected{"rdfmanybranchtypesttree_snapshotemptytree_expected.root"};
   const auto outputTreeNameActual{"outTreeActual"};
   const auto outputFileNameActual{"rdfmanybranchtypesttree_snapshotemptytree_actual.root"};
   const auto topLevelBranchNames = [&]() {
      TFile f{fgFileName};
      auto *t = f.Get<TTree>(fgTreeName);
      return ROOT::Internal::TreeUtils::GetTopLevelBranchNames(*t);
   }();
   {
      ROOT::RDataFrame init{fgTreeName, fgFileName};
      ROOT::RDF::RNode df = init;
      ROOT::RDF::RSnapshotOptions opts{};
      opts.fVector2RVec = false;
      opts.fLazy = true;

      // In case the input TTree is not empty, this will fill the output TTree with all events and branches
      auto snapdf_expected = df.Snapshot(outputTreeNameExpected, outputFileNameExpected, topLevelBranchNames, opts);

      // If the input TTree was not empty, we want to simulate the situation
      // where no events pass any filter so the output TTree will be empty.
      if (GetParam().second)
         df = df.Filter([] { return false; });

      auto snap_actual = df.Snapshot(outputTreeNameActual, outputFileNameActual, topLevelBranchNames, opts);

      // Trigger both snapshots
      *snapdf_expected;
   }

   // Check that branch names and types in all output cases are the same as the original
   TFile file_original{fgFileName};
   auto *tree_original = file_original.Get<TTree>(fgTreeName);

   auto branchnames_original = ROOT::Internal::TreeUtils::GetTopLevelBranchNames(*tree_original);
   auto nbranches_original = branchnames_original.size();
   std::vector<std::string> branchtypes_original(nbranches_original);
   for (decltype(nbranches_original) i = 0; i < nbranches_original; i++) {
      branchtypes_original[i] = ROOT::Internal::RDF::GetBranchOrLeafTypeName(*tree_original, branchnames_original[i]);
   }
   std::sort(branchnames_original.begin(), branchnames_original.end());
   std::sort(branchtypes_original.begin(), branchtypes_original.end());

   TFile file_expected{outputFileNameExpected};
   auto *tree_expected = file_expected.Get<TTree>(outputTreeNameExpected);

   auto branchnames_expected = ROOT::Internal::TreeUtils::GetTopLevelBranchNames(*tree_expected);
   auto nbranches_expected = branchnames_expected.size();
   std::vector<std::string> branchtypes_expected(nbranches_expected);
   for (decltype(nbranches_expected) i = 0; i < nbranches_expected; i++) {
      branchtypes_expected[i] = ROOT::Internal::RDF::GetBranchOrLeafTypeName(*tree_expected, branchnames_expected[i]);
   }
   std::sort(branchnames_expected.begin(), branchnames_expected.end());
   std::sort(branchtypes_expected.begin(), branchtypes_expected.end());

   TFile file_actual{outputFileNameActual};
   auto *tree_actual = file_actual.Get<TTree>(outputTreeNameActual);

   auto branchnames_actual = ROOT::Internal::TreeUtils::GetTopLevelBranchNames(*tree_actual);
   auto nbranches_actual = branchnames_actual.size();
   std::vector<std::string> branchtypes_actual(nbranches_actual);
   for (decltype(nbranches_actual) i = 0; i < nbranches_actual; i++) {
      branchtypes_actual[i] = ROOT::Internal::RDF::GetBranchOrLeafTypeName(*tree_actual, branchnames_actual[i]);
   }
   std::sort(branchnames_actual.begin(), branchnames_actual.end());
   std::sort(branchtypes_actual.begin(), branchtypes_actual.end());

   expect_vec_eq(branchnames_original, branchnames_expected);
   expect_vec_eq(branchtypes_original, branchtypes_expected);
   expect_vec_eq(branchnames_expected, branchnames_actual);
   expect_vec_eq(branchtypes_expected, branchtypes_actual);
}

class RDFMoreBranchesThanCompilerLimits : public ::testing::TestWithParam<bool> {
protected:
   inline constexpr static auto fgNEvents = 10u;
   inline constexpr static auto fgFileName{"rdfmorebranchesthancompilerlimits.root"};
   inline constexpr static auto fgTreeName{"rdfmorebranchesthancompilerlimits"};
   unsigned int fNSlots;

   RDFMoreBranchesThanCompilerLimits() : fNSlots(GetParam() ? std::min(4u, std::thread::hardware_concurrency()) : 1u)
   {
      if (GetParam())
         ROOT::EnableImplicitMT(fNSlots);
      TFile f{fgFileName, "recreate"};
      TTree t{fgTreeName, fgTreeName};

      // Compilers usually have a default template depth of 1024
      std::size_t nBranches{1025};
      std::vector<int> values(nBranches, 42);
      for (decltype(nBranches) i = 0; i < nBranches; i++) {
         auto branchName = "br_" + std::to_string(i);
         t.Branch(branchName.c_str(), &values[i]);
      }
      t.Fill();
      f.Write();
   }

   ~RDFMoreBranchesThanCompilerLimits() override
   {
      if (GetParam())
         ROOT::DisableImplicitMT();

      std::remove(fgFileName);
   }
};

TEST_P(RDFMoreBranchesThanCompilerLimits, SnapshotEmptyTTree)
{
   const auto outputTreeName{"outTree"};
   const auto outputFileName{"rdfmorebranchesthancompilerlimits_snapshotemptytree.root"};
   {
      ROOT::RDataFrame init{fgTreeName, fgFileName};
      ROOT::RDF::RNode df = init;
      // Simulate the situation where no entries pass the filter and we get an empty TTree in output
      df = df.Filter([] { return false; });
      df.Snapshot(outputTreeName, outputFileName);
   }

   // Check that branch names and types in all output cases are the same as the original
   TFile originalFile{fgFileName};
   auto *originalTree = originalFile.Get<TTree>(fgTreeName);

   auto originalBranchNames = ROOT::Internal::TreeUtils::GetTopLevelBranchNames(*originalTree);
   auto originalNBranches = originalBranchNames.size();
   std::vector<std::string> originalBranchTypes(originalNBranches);
   for (decltype(originalNBranches) i = 0; i < originalNBranches; i++) {
      originalBranchTypes[i] = ROOT::Internal::RDF::GetBranchOrLeafTypeName(*originalTree, originalBranchNames[i]);
   }
   std::sort(originalBranchNames.begin(), originalBranchNames.end());
   std::sort(originalBranchTypes.begin(), originalBranchTypes.end());

   TFile outputFile{outputFileName};
   auto *outputTree = outputFile.Get<TTree>(outputTreeName);

   auto outputBranchNames = ROOT::Internal::TreeUtils::GetTopLevelBranchNames(*outputTree);
   auto outputNBranches = outputBranchNames.size();
   std::vector<std::string> outputBranchTypes(outputNBranches);
   for (decltype(outputNBranches) i = 0; i < outputNBranches; i++) {
      outputBranchTypes[i] = ROOT::Internal::RDF::GetBranchOrLeafTypeName(*outputTree, outputBranchNames[i]);
   }
   std::sort(outputBranchNames.begin(), outputBranchNames.end());
   std::sort(outputBranchTypes.begin(), outputBranchTypes.end());

   expect_vec_eq(originalBranchNames, outputBranchNames);
   expect_vec_eq(originalBranchTypes, outputBranchTypes);
}

// instantiate single-thread tests
INSTANTIATE_TEST_SUITE_P(Seq, RDFManyBranchTypesTTree,
                         ::testing::Values(std::make_pair(false, false), std::make_pair(false, true)));
INSTANTIATE_TEST_SUITE_P(Seq, RDFMoreBranchesThanCompilerLimits, ::testing::Values(false));

// instantiate multi-thread tests
#ifdef R__USE_IMT
INSTANTIATE_TEST_SUITE_P(MT, RDFManyBranchTypesTTree,
                         ::testing::Values(std::make_pair(true, false), std::make_pair(true, true)));
INSTANTIATE_TEST_SUITE_P(MT, RDFMoreBranchesThanCompilerLimits, ::testing::Values(false));
#endif
