#include <gtest/gtest.h>

#include <ROOT/TestSupport.hxx>

#include <ROOT/RDataFrame.hxx>

#include <TFile.h>
#include <TTree.h>

#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleWriter.hxx>

#include <vector>
#include <array>

#include "ClassWithSequenceContainers.hxx"

using ROOT::RNTupleModel;
using ROOT::RNTupleWriter;

struct ClassWithSequenceContainersData {
   std::array<float, 3> fArrFl{};
   std::array<std::array<float, 3>, 3> fArrArrFl{};
   std::array<std::vector<float>, 3> fArrVecFl{};

   std::vector<float> fVecFl{};
   std::vector<std::array<float, 3>> fVecArrFl{};
   std::vector<std::vector<float>> fVecVecFl{};

   ClassWithSequenceContainers fClassWithArrays{};
   std::vector<ClassWithSequenceContainers> fVecClassWithArrays{};
};

std::vector<ClassWithSequenceContainersData> generateClassWithSequenceContainersData()
{
   // ClassWithSequenceContainers members
   std::array<float, 3> topArrFl{1.1, 2.2, 3.3};
   std::array<std::array<float, 3>, 3> topArrArrFl{{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}, {7.7, 8.8, 9.9}}};
   std::array<std::vector<float>, 3> topArrVecFl{{{11.11}, {12.12, 13.13}, {14.14, 15.15, 16.16}}};

   std::vector<float> topVecFl{17.17, 18.18, 19.19};
   std::vector<std::array<float, 3>> topVecArrFl{{{21.21, 22.22, 23.23},
                                                  {24.24, 25.25, 26.26},
                                                  {27.27, 28.28, 29.29},
                                                  {31.31, 32.32, 33.33},
                                                  {34.34, 35.35, 36.36},
                                                  {37.37, 38.38, 39.39}}};
   std::vector<std::vector<float>> topVecVecFl{{}, {41.41}, {42.42, 43.43}, {44.44, 45.45, 46.46}};

   // Class object
   ClassWithSequenceContainers classWithArrays(0, topArrFl, topArrArrFl, topArrVecFl, topVecFl, topVecArrFl,
                                               topVecVecFl);

   // std::vector of class objects
   std::vector<ClassWithSequenceContainers> vecClassWithArrays;
   vecClassWithArrays.reserve(5);
   for (int i = 1; i < 6; i++) {
      vecClassWithArrays.emplace_back(i, topArrFl, topArrArrFl, topArrVecFl, topVecFl, topVecArrFl, topVecVecFl);
   }
   return std::vector{ClassWithSequenceContainersData{topArrFl, topArrArrFl, topArrVecFl, topVecFl, topVecArrFl,
                                                      topVecVecFl, classWithArrays, vecClassWithArrays}};
}

std::vector<ClassWithSequenceContainersData> generateClassWithSequenceContainersDataPlusOne()
{
   auto data = generateClassWithSequenceContainersData();
   for (auto &entry : data) {
      for (auto &val : entry.fArrFl) {
         val += 1;
      }
      for (auto &arr : entry.fArrArrFl) {
         for (auto &val : arr) {
            val += 1;
         }
      }
      for (auto &vec : entry.fArrVecFl) {
         for (auto &val : vec) {
            val += 1;
         }
      }
      for (auto &val : entry.fVecFl) {
         val += 1;
      }
      for (auto &arr : entry.fVecArrFl) {
         for (auto &val : arr) {
            val += 1;
         }
      }
      for (auto &vec : entry.fVecVecFl) {
         for (auto &val : vec) {
            val += 1;
         }
      }
      auto classWithArraysPlusOne = [](ClassWithSequenceContainers &obj) {
         obj.fObjIndex += 1;
         for (auto &val : obj.fArrFl) {
            val += 1;
         }
         for (auto &arr : obj.fArrArrFl) {
            for (auto &val : arr) {
               val += 1;
            }
         }
         for (auto &vec : obj.fArrVecFl) {
            for (auto &val : vec) {
               val += 1;
            }
         }
         for (auto &val : obj.fVecFl) {
            val += 1;
         }
         for (auto &arr : obj.fVecArrFl) {
            for (auto &val : arr) {
               val += 1;
            }
         }
         for (auto &vec : obj.fVecVecFl) {
            for (auto &val : vec) {
               val += 1;
            }
         }
      };
      classWithArraysPlusOne(entry.fClassWithArrays);
      for (auto &obj : entry.fVecClassWithArrays) {
         classWithArraysPlusOne(obj);
      }
   }
   return data;
}

class ClassWithSequenceContainersTest : public ::testing::TestWithParam<bool> {
protected:
   constexpr static const char *fFileName = "root_dataframe_sequence_containers_ClassWithSequenceContainers.root";
   constexpr static const char *fDatasetName = "root_dataframe_sequence_containers_ClassWithSequenceContainers";

   void WriteTTree()
   {
      auto f = std::make_unique<TFile>(fFileName, "RECREATE");
      auto t = std::make_unique<TTree>(fDatasetName, fDatasetName);

      auto data = generateClassWithSequenceContainersData();

      // Branches
      t->Branch("topArrFl", &data[0].fArrFl);
      // Not supported: std::array<T, N> with T being a class type as top-level branch
      // t->Branch("topArrArrFl", &data.topArrArrFl);
      // t->Branch("topArrVecFl", &data.topArrVecFl);
      t->Branch("topVecFl", &data[0].fVecFl);
      // Not supported: Could not find the real data member '_M_elems[3]' when constructing the branch 'topVecArrFl'
      // t->Branch("topVecArrFl", &data.fVecArrFl);
      t->Branch("topVecVecFl", &data[0].fVecVecFl);
      t->Branch("classWithArrays", &data[0].fClassWithArrays);
      t->Branch("vecClassWithArrays", &data[0].fVecClassWithArrays);

      t->Fill();

      f->Write();
   }

   void WriteRNTuple()
   {
      auto model = RNTupleModel::Create();
      auto topArrFl = model->MakeField<std::array<float, 3>>("topArrFl");
      auto topArrArrFl = model->MakeField<std::array<std::array<float, 3>, 3>>("topArrArrFl");
      auto topArrVecFl = model->MakeField<std::array<std::vector<float>, 3>>("topArrVecFl");

      auto topVecFl = model->MakeField<std::vector<float>>("topVecFl");
      auto topVecArrFl = model->MakeField<std::vector<std::array<float, 3>>>("topVecArrFl");
      auto topVecVecFl = model->MakeField<std::vector<std::vector<float>>>("topVecVecFl");

      auto classWithArrays = model->MakeField<ClassWithSequenceContainers>("classWithArrays");
      auto vecClassWithArrays = model->MakeField<std::vector<ClassWithSequenceContainers>>("vecClassWithArrays");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), fDatasetName, fFileName);
      auto data = generateClassWithSequenceContainersData();
      for (const auto &entry : data) {
         *topArrFl = entry.fArrFl;
         *topArrArrFl = entry.fArrArrFl;
         *topArrVecFl = entry.fArrVecFl;
         *topVecFl = entry.fVecFl;
         *topVecArrFl = entry.fVecArrFl;
         *topVecVecFl = entry.fVecVecFl;
         *classWithArrays = entry.fClassWithArrays;
         *vecClassWithArrays = entry.fVecClassWithArrays;
         ntuple->Fill();
      }
   }

   ClassWithSequenceContainersTest()
   {
      if (GetParam()) {
         WriteTTree();
      } else {
         WriteRNTuple();
      }
   }

   ~ClassWithSequenceContainersTest() override { std::remove(fFileName); }
};

template <typename T, typename U>
void check_1d_coll(const T &coll1, const U &coll2)
{
   ASSERT_EQ(coll1.size(), coll2.size());
   for (size_t i = 0; i < coll1.size(); ++i) {
      EXPECT_FLOAT_EQ(coll1[i], coll2[i]) << " at index " << i;
   }
}

template <typename T, typename U>
void check_2d_coll(const T &coll1, const U &coll2)
{
   ASSERT_EQ(coll1.size(), coll2.size());
   for (size_t i = 0; i < coll1.size(); ++i) {
      check_1d_coll(coll1[i], coll2[i]);
   }
}

void check_class_with_arrays(const ClassWithSequenceContainers &obj1, const ClassWithSequenceContainers &obj2)
{
   EXPECT_EQ(obj1.fObjIndex, obj2.fObjIndex);
   check_1d_coll(obj1.fArrFl, obj2.fArrFl);
   check_2d_coll(obj1.fArrArrFl, obj2.fArrArrFl);
   check_2d_coll(obj1.fArrVecFl, obj2.fArrVecFl);
   check_1d_coll(obj1.fVecFl, obj2.fVecFl);
   // fVecArrFl is not supported: Could not find the real data member '_M_elems[3]' when constructing the branch
   // 'fVecArrFl'
   // check_2d_coll(obj1.fVecArrFl, obj2.fVecArrFl);
   check_2d_coll(obj1.fVecVecFl, obj2.fVecVecFl);
}

template <typename T, typename U>
void check_coll_class_with_arrays(const T &coll1, const U &coll2)
{
   ASSERT_EQ(coll1.size(), coll2.size());
   for (size_t i = 0; i < coll1.size(); ++i) {
      check_class_with_arrays(coll1[i], coll2[i]);
   }
}

TEST_P(ClassWithSequenceContainersTest, ExpectedTypes)
{
   // TODO: This test currently assumes spellings of column types when the
   // data format is TTree
   if (!GetParam())
      return; // The expected type names are different between the TTree and RNTuple data sources

   ROOT::RDataFrame df{fDatasetName, fFileName};

   const std::unordered_map<std::string, std::string> expectedColTypes{
      {"topArrFl", "ROOT::VecOps::RVec<Float_t>"},
      // Not supported: std::array<T, N> with T being a class type as top-level branch
      // {"topArrArrFl", "ROOT::VecOps::RVec<ROOT::VecOps::RVec<Float_t>>"},
      // {"topArrVecFl", "ROOT::VecOps::RVec<ROOT::VecOps::RVec<Float_t>>"},
      {"topVecFl", "ROOT::VecOps::RVec<float>"},
      // Not supported: Could not find the real data member '_M_elems[3]' when constructing the branch 'topVecArrFl'
      // {"topVecArrFl", "ROOT::VecOps::RVec<std::vector<float>>"},
      {"topVecVecFl", "ROOT::VecOps::RVec<vector<float>>"},
      {"classWithArrays", "ClassWithSequenceContainers"},
      {"classWithArrays.fObjIndex", "UInt_t"},
      {"classWithArrays.fArrFl[3]", "ROOT::VecOps::RVec<Float_t>"},
      // TODO: array of array is currently not properly handled
      // {"classWithArrays.fArrArrFl[3][3]", "ROOT::VecOps::RVec<array<float,3>>"},
      {"classWithArrays.fArrVecFl[3]", "ROOT::VecOps::RVec<vector<float>>"},
      {"classWithArrays.fVecFl", "ROOT::VecOps::RVec<float>"},
      {"classWithArrays.fVecVecFl", "ROOT::VecOps::RVec<vector<float>>"},
      {"vecClassWithArrays", "ROOT::VecOps::RVec<ClassWithSequenceContainers>"},
      {"vecClassWithArrays.fArrFl[3]", "ROOT::VecOps::RVec<std::array<Float_t, 3>>"},
      {"vecClassWithArrays.fArrVecFl[3]", "ROOT::VecOps::RVec<std::array<vector<float>, 3>>"},
      {"vecClassWithArrays.fVecFl", "ROOT::VecOps::RVec<vector<float>>"},
      {"vecClassWithArrays.fVecVecFl", "ROOT::VecOps::RVec<vector<vector<float> >>"},
   };
   for (const auto &[colName, expectedType] : expectedColTypes) {
      EXPECT_EQ(df.GetColumnType(colName), expectedType) << " for column " << colName;
   }
}

TEST_P(ClassWithSequenceContainersTest, TakeExpectedTypes)
{
#ifndef NDEBUG
   // The following warning is only for debugging purposes with the TTree data source. It happens in this test
   // because we are reading a class partially, i.e. the branch type is std::vector<ClassWithSequenceContainers> but
   // we are only reading the data member fArrFl with the column name "classWithArrays.fArrFl[3]" as an
   // RVec<std::array<float, 3>>.
   ROOT::TestSupport::CheckDiagsRAII diagRAII;
   diagRAII.optionalDiag(
      kWarning, "RTreeColumnReader::Get",
      "hangs from a non-split branch. A copy is being performed in order to properly read the content.", false);
#endif

   ROOT::RDataFrame df{fDatasetName, fFileName};

   auto data = generateClassWithSequenceContainersData();

   const std::string classWithArraysArrFlColName = GetParam() ? "classWithArrays.fArrFl[3]" : "classWithArrays.fArrFl";
   const std::string vecClassWithArraysArrFlColName =
      GetParam() ? "vecClassWithArrays.fArrFl[3]" : "vecClassWithArrays.fArrFl";

   // Take each column individually and check the content
   // In this test, use the types as expected by the test "ExpectedTypes"
   auto takeTopArrFl = df.Take<ROOT::VecOps::RVec<Float_t>>("topArrFl");
   // Not supported: std::array<T, N> with T being a class type as top-level branch
   // auto takeTopArrArrFl = df.Take<ROOT::VecOps::RVec<Float_t>>("topArrArrFl");
   // auto takeTopArrVecFl = df.Take<ROOT::VecOps::RVec<Float_t>>("topArrVecFl");
   auto takeTopVecFl = df.Take<ROOT::VecOps::RVec<float>>("topVecFl");
   // Not supported: Could not find the real data member '_M_elems[3]' when constructing the branch 'topVecArrFl'
   // auto takeTopVecArrFl = df.Take<ROOT::VecOps::RVec<std::array<float, 3>>>("topVecArrFl");
   auto takeTopVecVecFl = df.Take<ROOT::VecOps::RVec<std::vector<float>>>("topVecVecFl");
   auto takeClassWithArrays = df.Take<ClassWithSequenceContainers>("classWithArrays");
   auto takeClassWithArrays_fObjIndex = df.Take<UInt_t>("classWithArrays.fObjIndex");
   auto takeClassWithArrays_fArrFl = df.Take<ROOT::VecOps::RVec<Float_t>>(classWithArraysArrFlColName);
   // TODO: array of array is currently not properly handled
   // auto takeClassWithArrays_fArrArrFl = df.Take<std::array<std::array<float, 3>,
   // 3>>>("classWithArrays.fArrArrFl[3][3]");
   // TODO: array of vector as a data member is currently not properly handled
   // auto takeClassWithArrays_fArrVecFl =
   // df.Take<ROOT::VecOps::RVec<std::vector<float>>>("classWithArrays.fArrVecFl[3]");
   auto takeClassWithArrays_fVecFl = df.Take<ROOT::VecOps::RVec<float>>("classWithArrays.fVecFl");
   // RNTuple currently fails with the following operation with error
   // RNTupleDS: Could not create field with type "ROOT::VecOps::RVec<std::vector<float>>" for column
   // "classWithArrays.fVecVecFl"
   ROOT::RDF::RResultPtr<std::vector<ROOT::VecOps::RVec<std::vector<float>>>> takeClassWithArrays_fVecVecFl;
   if (GetParam()) {
      takeClassWithArrays_fVecVecFl = df.Take<ROOT::VecOps::RVec<std::vector<float>>>("classWithArrays.fVecVecFl");
   }
   auto takeVecClassWithArrays = df.Take<ROOT::VecOps::RVec<ClassWithSequenceContainers>>("vecClassWithArrays");
   // RNTuple currently fails with the following operation with error
   // RNTupleDS: Could not create field with type "std::array<float,3>" for column "classWithArrays.fArrFl"
   ROOT::RDF::RResultPtr<std::vector<ROOT::VecOps::RVec<std::array<Float_t, 3>>>> takeVecClassWithArrays_fArrFl;
   if (GetParam()) {
      takeVecClassWithArrays_fArrFl =
         df.Take<ROOT::VecOps::RVec<std::array<Float_t, 3>>>(vecClassWithArraysArrFlColName);
   }
   // TODO: array of vector as a data member is currently not properly handled
   // auto takeVecClassWithArrays_fArrVecFl =
   // df.Take<ROOT::VecOps::RVec<std::array<std::vector<float>,3>>>("vecClassWithArrays.fArrVecFl[3]");
   // TODO: vector of vector throws `std::bad_alloc` currently
   // auto takeVecClassWithArrays_fVecFl = df.Take<ROOT::VecOps::RVec<std::vector<float>>>("vecClassWithArrays.fVecFl");
   // auto takeVecClassWithArrays_fVecVecFl =
   // df.Take<ROOT::VecOps::RVec<std::vector<std::vector<float>>>>("vecClassWithArrays.fVecVecFl");

   auto nEvents = takeTopArrFl->size();
   EXPECT_EQ(nEvents, 1);

   for (decltype(nEvents) i = 0; i < nEvents; ++i) {
      check_1d_coll(takeTopArrFl->at(i), data[i].fArrFl);
      check_1d_coll(takeTopVecFl->at(i), data[i].fVecFl);
      check_2d_coll(takeTopVecVecFl->at(i), data[i].fVecVecFl);
      check_class_with_arrays(takeClassWithArrays->at(i), data[i].fClassWithArrays);
      EXPECT_EQ(takeClassWithArrays_fObjIndex->at(i), data[i].fClassWithArrays.fObjIndex);
      check_1d_coll(takeClassWithArrays_fArrFl->at(i), data[i].fClassWithArrays.fArrFl);
      check_1d_coll(takeClassWithArrays_fVecFl->at(i), data[i].fClassWithArrays.fVecFl);
      if (GetParam()) {
         check_2d_coll(takeClassWithArrays_fVecVecFl->at(i), data[i].fClassWithArrays.fVecVecFl);
      }
      check_coll_class_with_arrays(takeVecClassWithArrays->at(i), data[i].fVecClassWithArrays);
      if (GetParam()) {
         std::vector<std::array<float, 3>> expectedVecArrFl(data[i].fVecClassWithArrays.size());
         for (size_t j = 0; j < data[i].fVecClassWithArrays.size(); ++j) {
            expectedVecArrFl[j] = data[i].fVecClassWithArrays[j].fArrFl;
         }
         check_2d_coll(takeVecClassWithArrays_fArrFl->at(i), expectedVecArrFl);
      }
   }
}

TEST_P(ClassWithSequenceContainersTest, TakeOriginalTypes)
{
   ROOT::RDataFrame df{fDatasetName, fFileName};

   const std::string classWithArraysArrFlColName = GetParam() ? "classWithArrays.fArrFl[3]" : "classWithArrays.fArrFl";
   const std::string vecClassWithArraysArrFlColName =
      GetParam() ? "vecClassWithArrays.fArrFl[3]" : "vecClassWithArrays.fArrFl";

   auto data = generateClassWithSequenceContainersData();

   // Take each column individually and check the content
   // In this test, call Take using only original types as written in the EDM
   auto takeTopArrFl = df.Take<std::array<float, 3>>("topArrFl");
   // Not supported: std::array<T, N> with T being a class type as top-level branch
   // auto takeTopArrArrFl = df.Take<ROOT::VecOps::RVec<Float_t>>("topArrArrFl");
   // auto takeTopArrVecFl = df.Take<ROOT::VecOps::RVec<Float_t>>("topArrVecFl");
   auto takeTopVecFl = df.Take<std::vector<float>>("topVecFl");
   // Not supported: Could not find the real data member '_M_elems[3]' when constructing the branch 'topVecArrFl'
   // auto takeTopVecArrFl = df.Take<ROOT::VecOps::RVec<std::array<float, 3>>>("topVecArrFl");
   auto takeTopVecVecFl = df.Take<std::vector<std::vector<float>>>("topVecVecFl");
   auto takeClassWithArrays = df.Take<ClassWithSequenceContainers>("classWithArrays");
   auto takeClassWithArrays_fObjIndex = df.Take<unsigned int>("classWithArrays.fObjIndex");
   // RNTuple currently fails with the following operation with error
   // RNTupleDS: Could not create field with type "std::array<float,3>" for column "classWithArrays.fArrFl"
   ROOT::RDF::RResultPtr<std::vector<std::array<float, 3>>> takeClassWithArrays_fArrFl;
   if (GetParam()) {
      takeClassWithArrays_fArrFl = df.Take<std::array<float, 3>>(classWithArraysArrFlColName);
   }
   // TODO: array of array is currently not properly handled
   // auto takeClassWithArrays_fArrArrFl = df.Take<std::array<std::array<float, 3>,
   // 3>>>("classWithArrays.fArrArrFl[3][3]");
   // TODO: array of vector as a data member is currently not properly handled
   // auto takeClassWithArrays_fArrVecFl =
   // df.Take<ROOT::VecOps::RVec<std::vector<float>>>("classWithArrays.fArrVecFl[3]");
   // RNTuple currently fails with the following operation with error
   // RNTupleDS: Could not create field with type "std::vector<float>" for column "classWithArrays.fVecFl"
   ROOT::RDF::RResultPtr<std::vector<std::vector<float>>> takeClassWithArrays_fVecFl;
   if (GetParam()) {
      takeClassWithArrays_fVecFl = df.Take<std::vector<float>>("classWithArrays.fVecFl");
   }
   // RNTuple currently fails with the following operation with error
   // RNTupleDS: Could not create field with type "std::vector<std::vector<float>>" for column
   // "classWithArrays.fVecVecFl"
   ROOT::RDF::RResultPtr<std::vector<std::vector<std::vector<float>>>> takeClassWithArrays_fVecVecFl;
   if (GetParam()) {
      takeClassWithArrays_fVecVecFl = df.Take<std::vector<std::vector<float>>>("classWithArrays.fVecVecFl");
   }
   auto takeVecClassWithArrays = df.Take<std::vector<ClassWithSequenceContainers>>("vecClassWithArrays");
   // RNTuple currently fails with the following operation with error
   // RNTupleDS: Could not create field with type "std::vector<std::array<float,3>>" for column
   // "vecClassWithArrays.fArrFl"
   ROOT::RDF::RResultPtr<std::vector<std::vector<std::array<float, 3>>>> takeVecClassWithArrays_fArrFl;
   if (GetParam()) {
      takeVecClassWithArrays_fArrFl = df.Take<std::vector<std::array<float, 3>>>(vecClassWithArraysArrFlColName);
   }
   // TODO: array of vector as a data member is currently not properly handled
   // auto takeVecClassWithArrays_fArrVecFl =
   // df.Take<ROOT::VecOps::RVec<std::array<std::vector<float>,3>>>("vecClassWithArrays.fArrVecFl[3]");
   // TODO: vector of vector throws `std::bad_alloc` currently
   // auto takeVecClassWithArrays_fVecFl = df.Take<ROOT::VecOps::RVec<std::vector<float>>>("vecClassWithArrays.fVecFl");
   // auto takeVecClassWithArrays_fVecVecFl =
   // df.Take<ROOT::VecOps::RVec<std::vector<std::vector<float>>>>("vecClassWithArrays.fVecVecFl");

   auto nEvents = takeTopArrFl->size();
   EXPECT_EQ(nEvents, 1);

   for (decltype(nEvents) i = 0; i < nEvents; ++i) {
      check_1d_coll(takeTopArrFl->at(i), data[i].fArrFl);
      check_1d_coll(takeTopVecFl->at(i), data[i].fVecFl);
      check_2d_coll(takeTopVecVecFl->at(i), data[i].fVecVecFl);
      check_class_with_arrays(takeClassWithArrays->at(i), data[i].fClassWithArrays);
      EXPECT_EQ(takeClassWithArrays_fObjIndex->at(i), data[i].fClassWithArrays.fObjIndex);
      if (GetParam()) {
         check_1d_coll(takeClassWithArrays_fArrFl->at(i), data[i].fClassWithArrays.fArrFl);
         check_1d_coll(takeClassWithArrays_fVecFl->at(i), data[i].fClassWithArrays.fVecFl);
         check_2d_coll(takeClassWithArrays_fVecVecFl->at(i), data[i].fClassWithArrays.fVecVecFl);
      }
      check_coll_class_with_arrays(takeVecClassWithArrays->at(i), data[i].fVecClassWithArrays);
      if (GetParam()) {
         std::vector<std::array<float, 3>> expectedVecArrFl(data[i].fVecClassWithArrays.size());
         for (size_t j = 0; j < data[i].fVecClassWithArrays.size(); ++j) {
            expectedVecArrFl[j] = data[i].fVecClassWithArrays[j].fArrFl;
         }
         check_2d_coll(takeVecClassWithArrays_fArrFl->at(i), expectedVecArrFl);
      }
   }
}

TEST_P(ClassWithSequenceContainersTest, TemplatedOps)
{
   ROOT::RDataFrame df{fDatasetName, fFileName};
   ROOT::RDF::RNode node = df;

   const std::string classWithArraysArrFlColName = GetParam() ? "classWithArrays.fArrFl[3]" : "classWithArrays.fArrFl";
   const std::string vecClassWithArraysArrFlColName =
      GetParam() ? "vecClassWithArrays.fArrFl[3]" : "vecClassWithArrays.fArrFl";

   auto data = generateClassWithSequenceContainersDataPlusOne();

   node = node.Define("topArrFl_plus_1",
                      [](const std::array<float, 3> &arr) {
                         std::array<float, 3> result;
                         for (size_t i = 0; i < arr.size(); ++i) {
                            result[i] = arr[i] + 1.0f;
                         }
                         return result;
                      },
                      {"topArrFl"});
   node = node.Define("topVecFl_plus_1",
                      [](const std::vector<float> &vec) {
                         std::vector<float> result(vec.size());
                         for (size_t i = 0; i < vec.size(); ++i) {
                            result[i] = vec[i] + 1.0f;
                         }
                         return result;
                      },
                      {"topVecFl"});
   node = node.Define("topVecVecFl_plus_1",
                      [](const std::vector<std::vector<float>> &vecvec) {
                         std::vector<std::vector<float>> result(vecvec.size());
                         for (size_t i = 0; i < vecvec.size(); ++i) {
                            result[i].resize(vecvec[i].size());
                            for (size_t j = 0; j < vecvec[i].size(); ++j) {
                               result[i][j] = vecvec[i][j] + 1.0f;
                            }
                         }
                         return result;
                      },
                      {"topVecVecFl"});
   node = node.Define("classWithArrays_plus_1",
                      [](const ClassWithSequenceContainers &obj) {
                         ClassWithSequenceContainers result = obj;
                         result.fObjIndex += 1;
                         for (auto &val : result.fArrFl) {
                            val += 1;
                         }
                         for (auto &arr : result.fArrArrFl) {
                            for (auto &val : arr) {
                               val += 1;
                            }
                         }
                         for (auto &vec : result.fArrVecFl) {
                            for (auto &val : vec) {
                               val += 1;
                            }
                         }
                         for (auto &val : result.fVecFl) {
                            val += 1;
                         }
                         for (auto &arr : result.fVecArrFl) {
                            for (auto &val : arr) {
                               val += 1;
                            }
                         }
                         for (auto &vec : result.fVecVecFl) {
                            for (auto &val : vec) {
                               val += 1;
                            }
                         }
                         return result;
                      },
                      {"classWithArrays"});
   node = node.Define("vecClassWithArrays_plus_1",
                      [](const std::vector<ClassWithSequenceContainers> &vec) {
                         std::vector<ClassWithSequenceContainers> result = vec;
                         for (auto &obj : result) {
                            obj.fObjIndex += 1;
                            for (auto &val : obj.fArrFl) {
                               val += 1;
                            }
                            for (auto &arr : obj.fArrArrFl) {
                               for (auto &val : arr) {
                                  val += 1;
                               }
                            }
                            for (auto &vvec : obj.fArrVecFl) {
                               for (auto &val : vvec) {
                                  val += 1;
                               }
                            }
                            for (auto &val : obj.fVecFl) {
                               val += 1;
                            }
                            for (auto &arr : obj.fVecArrFl) {
                               for (auto &val : arr) {
                                  val += 1;
                               }
                            }
                            for (auto &vvec : obj.fVecVecFl) {
                               for (auto &val : vvec) {
                                  val += 1;
                               }
                            }
                         }
                         return result;
                      },
                      {"vecClassWithArrays"});
   // Also create modified values for data member columns
   node = node.Define("classWithArrays_fObjIndex_plus_1", [](unsigned int objIndex) { return objIndex + 1; },
                      {"classWithArrays.fObjIndex"});
   // RNTuple currently fails with the following operation with error
   // RNTupleDS: Could not create field with type "std::array<float,3>" for column "classWithArrays.fArrFl"
   if (GetParam()) {
      node = node.Define("classWithArrays_fArrFl_plus_1",
                         [](const std::array<float, 3> &arr) {
                            std::array<float, 3> result;
                            for (size_t i = 0; i < arr.size(); ++i) {
                               result[i] = arr[i] + 1.0f;
                            }
                            return result;
                         },
                         {classWithArraysArrFlColName});
   }
   // RNTuple currently fails with the following operation with error
   // RNTupleDS: Could not create field with type "std::vector<float>" for column "classWithArrays.fVecFl"
   if (GetParam()) {
      node = node.Define("classWithArrays_fVecFl_plus_1",
                         [](const std::vector<float> &vec) {
                            std::vector<float> result(vec.size());
                            for (size_t i = 0; i < vec.size(); ++i) {
                               result[i] = vec[i] + 1.0f;
                            }
                            return result;
                         },
                         {"classWithArrays.fVecFl"});
   }
   // RNTuple currently fails with the following operation with error
   // RNTupleDS: Could not create field with type "std::vector<std::vector<float>>" for column
   // "classWithArrays.fVecVecFl"
   if (GetParam()) {
      node = node.Define("classWithArrays_fVecVecFl_plus_1",
                         [](const std::vector<std::vector<float>> &vecVecFl) {
                            std::vector<std::vector<float>> result(vecVecFl.size());
                            for (size_t i = 0; i < vecVecFl.size(); ++i) {
                               result[i].resize(vecVecFl[i].size());
                               for (size_t j = 0; j < vecVecFl[i].size(); ++j) {
                                  result[i][j] = vecVecFl[i][j] + 1.0f;
                               }
                            }
                            return result;
                         },
                         {"classWithArrays.fVecVecFl"});
   }
   // RNTuple currently fails with the following operation with error
   // RNTupleDS: Could not create field with type "std::vector<std::array<float,3>>" for column
   // "vecClassWithArrays.fArrFl"
   if (GetParam()) {
      node = node.Define("vecClassWithArrays_fArrFl_plus_1",
                         [](const std::vector<std::array<float, 3>> &vecArrFl) {
                            std::vector<std::array<float, 3>> result(vecArrFl.size());
                            for (size_t i = 0; i < vecArrFl.size(); ++i) {
                               for (size_t j = 0; j < vecArrFl[i].size(); ++j) {
                                  result[i][j] = vecArrFl[i][j] + 1.0f;
                               }
                            }
                            return result;
                         },
                         {vecClassWithArraysArrFlColName});
   }
   // Take each column individually and check the content
   auto takeTopArrFl = node.Take<std::array<float, 3>>("topArrFl_plus_1");
   // Not supported: std::array<T, N> with T being a class type as top-level branch
   // auto takeTopArrArrFl = node.Take<ROOT::VecOps::RVec<Float_t>>("topArrArrFl");
   // auto takeTopArrVecFl = node.Take<ROOT::VecOps::RVec<Float_t>>("topArrVecFl");
   auto takeTopVecFl = node.Take<std::vector<float>>("topVecFl_plus_1");
   // Not supported: Could not find the real data member '_M_elems[3]' when constructing the branch 'topVecArrFl'
   // auto takeTopVecArrFl = node.Take<ROOT::VecOps::RVec<std::array<float, 3>>>("topVecArrFl");
   auto takeTopVecVecFl = node.Take<std::vector<std::vector<float>>>("topVecVecFl_plus_1");
   auto takeClassWithArrays = node.Take<ClassWithSequenceContainers>("classWithArrays_plus_1");
   auto takeClassWithArrays_fObjIndex = node.Take<UInt_t>("classWithArrays_fObjIndex_plus_1");
   // RNTuple currently fails with the following operation with error
   // RNTupleDS: Could not create field with type "std::array<float,3>" for column "classWithArrays.fArrFl"
   ROOT::RDF::RResultPtr<std::vector<std::array<float, 3>>> takeClassWithArrays_fArrFl;
   if (GetParam()) {
      takeClassWithArrays_fArrFl = node.Take<std::array<float, 3>>("classWithArrays_fArrFl_plus_1");
   }
   // TODO: array of array is currently not properly handled
   // auto takeClassWithArrays_fArrArrFl = node.Take<std::array<std::array<float, 3>,
   // 3>>>("classWithArrays.fArrArrFl[3][3]");
   // TODO: array of vector as a data member is currently not properly handled
   // auto takeClassWithArrays_fArrVecFl =
   // node.Take<ROOT::VecOps::RVec<std::vector<float>>>("classWithArrays.fArrVecFl[3]");
   // RNTuple currently fails with the following operation with error
   // RNTupleDS: Could not create field with type "std::vector<float>" for column "classWithArrays.fVecFl"
   ROOT::RDF::RResultPtr<std::vector<std::vector<float>>> takeClassWithArrays_fVecFl;
   if (GetParam()) {
      takeClassWithArrays_fVecFl = node.Take<std::vector<float>>("classWithArrays_fVecFl_plus_1");
   }
   // RNTuple currently fails with the following operation with error
   // RNTupleDS: Could not create field with type "std::vector<std::vector<float>>" for column
   // "classWithArrays.fVecVecFl"
   ROOT::RDF::RResultPtr<std::vector<std::vector<std::vector<float>>>> takeClassWithArrays_fVecVecFl;
   if (GetParam()) {
      takeClassWithArrays_fVecVecFl = node.Take<std::vector<std::vector<float>>>("classWithArrays_fVecVecFl_plus_1");
   }
   auto takeVecClassWithArrays = node.Take<std::vector<ClassWithSequenceContainers>>("vecClassWithArrays_plus_1");
   // RNTuple currently fails with the following operation with error
   // RNTupleDS: Could not create field with type "std::vector<std::array<float,3>>" for column
   // "vecClassWithArrays.fArrFl"
   ROOT::RDF::RResultPtr<std::vector<std::vector<std::array<float, 3>>>> takeVecClassWithArrays_fArrFl;
   if (GetParam()) {
      takeVecClassWithArrays_fArrFl = node.Take<std::vector<std::array<float, 3>>>("vecClassWithArrays_fArrFl_plus_1");
   }
   // TODO: array of vector as a data member is currently not properly handled
   // auto takeVecClassWithArrays_fArrVecFl =
   // node.Take<ROOT::VecOps::RVec<std::array<std::vector<float>,3>>>("vecClassWithArrays.fArrVecFl[3]");
   // TODO: vector of vector throws `std::bad_alloc` currently
   // auto takeVecClassWithArrays_fVecFl =
   // node.Take<ROOT::VecOps::RVec<std::vector<float>>>("vecClassWithArrays.fVecFl"); auto
   // takeVecClassWithArrays_fVecVecFl =
   // node.Take<ROOT::VecOps::RVec<std::vector<std::vector<float>>>>("vecClassWithArrays.fVecVecFl");

   auto nEvents = takeTopArrFl->size();
   EXPECT_EQ(nEvents, 1);

   for (decltype(nEvents) i = 0; i < nEvents; ++i) {
      check_1d_coll(takeTopArrFl->at(i), data[i].fArrFl);
      check_1d_coll(takeTopVecFl->at(i), data[i].fVecFl);
      check_2d_coll(takeTopVecVecFl->at(i), data[i].fVecVecFl);
      check_class_with_arrays(takeClassWithArrays->at(i), data[i].fClassWithArrays);
      EXPECT_EQ(takeClassWithArrays_fObjIndex->at(i), data[i].fClassWithArrays.fObjIndex);
      if (GetParam()) {
         check_1d_coll(takeClassWithArrays_fArrFl->at(i), data[i].fClassWithArrays.fArrFl);
         check_1d_coll(takeClassWithArrays_fVecFl->at(i), data[i].fClassWithArrays.fVecFl);
         check_2d_coll(takeClassWithArrays_fVecVecFl->at(i), data[i].fClassWithArrays.fVecVecFl);
      }
      check_coll_class_with_arrays(takeVecClassWithArrays->at(i), data[i].fVecClassWithArrays);
      if (GetParam()) {
         std::vector<std::array<float, 3>> expectedVecArrFl(data[i].fVecClassWithArrays.size());
         for (size_t j = 0; j < data[i].fVecClassWithArrays.size(); ++j) {
            expectedVecArrFl[j] = data[i].fVecClassWithArrays[j].fArrFl;
         }
         check_2d_coll(takeVecClassWithArrays_fArrFl->at(i), expectedVecArrFl);
      }
   }
}

TEST_P(ClassWithSequenceContainersTest, JittedOps)
{
   ROOT::RDataFrame df{fDatasetName, fFileName};
   ROOT::RDF::RNode node = df;

   const std::string classWithArraysArrFlColName = GetParam() ? "classWithArrays.fArrFl[3]" : "classWithArrays.fArrFl";
   const std::string vecClassWithArraysArrFlColName =
      GetParam() ? "vecClassWithArrays.fArrFl[3]" : "vecClassWithArrays.fArrFl";

   auto data = generateClassWithSequenceContainersDataPlusOne();

   // all the next define calls should be written with jitted code in strings instead of C++ lambdas
   node = node.Define("topArrFl_plus_1",
                      R"CODE(
                         std::array<float, 3> result;
                         for (size_t i = 0; i < topArrFl.size(); ++i) {
                            result[i] = topArrFl[i] + 1.0f;
                         }
                         return result;
                         )CODE");
   node = node.Define("topVecFl_plus_1",
                      R"CODE(
                         std::vector<float> result(topVecFl.size());
                         for (size_t i = 0; i < topVecFl.size(); ++i) {
                            result[i] = topVecFl[i] + 1.0f;
                         }
                         return result;
                         )CODE");
   node = node.Define("topVecVecFl_plus_1",
                      R"CODE(
                         std::vector<std::vector<float>> result(topVecVecFl.size());
                         for (size_t i = 0; i < topVecVecFl.size(); ++i) {
                            result[i].resize(topVecVecFl[i].size());
                            for (size_t j = 0; j < topVecVecFl[i].size(); ++j) {
                               result[i][j] = topVecVecFl[i][j] + 1.0f;
                            }
                         }
                         return result;
                         )CODE");
   node = node.Define("classWithArrays_plus_1",
                      R"CODE(
                         ClassWithSequenceContainers result = classWithArrays;
                         result.fObjIndex += 1;
                         for (auto &val : result.fArrFl) {
                            val += 1;
                         }
                         for (auto &arr : result.fArrArrFl) {
                            for (auto &val : arr) {
                               val += 1;
                            }
                         }
                         for (auto &vec : result.fArrVecFl) {
                            for (auto &val : vec) {
                               val += 1;
                            }
                         }
                         for (auto &val : result.fVecFl) {
                            val += 1;
                         }
                         for (auto &arr : result.fVecArrFl) {
                            for (auto &val : arr) {
                               val += 1;
                            }
                         }
                         for (auto &vec : result.fVecVecFl) {
                            for (auto &val : vec) {
                               val += 1;
                            }
                         }
                         return result;
                        )CODE");
   node = node.Define("vecClassWithArrays_plus_1",
                      R"CODE(
                         ROOT::VecOps::RVec<ClassWithSequenceContainers> result = vecClassWithArrays;
                         for (auto &obj : result) {
                            obj.fObjIndex += 1;
                            for (auto &val : obj.fArrFl) {
                               val += 1;
                            }
                            for (auto &arr : obj.fArrArrFl) {
                               for (auto &val : arr) {
                                  val += 1;
                               }
                            }
                            for (auto &vvec : obj.fArrVecFl) {
                               for (auto &val : vvec) {
                                  val += 1;
                               }
                            }
                            for (auto &val : obj.fVecFl) {
                               val += 1;
                            }
                            for (auto &arr : obj.fVecArrFl) {
                               for (auto &val : arr) {
                                  val += 1;
                               }
                            }
                            for (auto &vvec : obj.fVecVecFl) {
                               for (auto &val : vvec) {
                                  val += 1;
                               }
                            }
                         }
                         return result;
                        )CODE");
   // Also create modified values for data member columns
   node = node.Define("classWithArrays_fObjIndex_plus_1", "return classWithArrays.fObjIndex + 1;");
   // Using a branch with an invalid C++ name will break the jitted execution
   // node = node.Alias("classWithArrays_fArrFl", "classWithArrays.fArrFl[3]");
   // node = node.Define("classWithArrays_fArrFl_plus_1",
   //                    R"CODE(
   //                       std::array<float, 3> result;
   //                       for (size_t i = 0; i < classWithArrays_fArrFl.size(); ++i) {
   //                          result[i] = classWithArrays_fArrFl[i] + 1.0f;
   //                       }
   //                       return result;
   //                   )CODE");
   node = node.Define("classWithArrays_fVecFl_plus_1",
                      R"CODE(
                         std::vector<float> result(classWithArrays.fVecFl.size());
                         for (size_t i = 0; i < classWithArrays.fVecFl.size(); ++i) {
                            result[i] = classWithArrays.fVecFl[i] + 1.0f;
                         }
                         return result;
                     )CODE");
   node = node.Define("classWithArrays_fVecVecFl_plus_1",
                      R"CODE(
                         std::vector<std::vector<float>> result(classWithArrays.fVecVecFl.size());
                         for (size_t i = 0; i < classWithArrays.fVecVecFl.size(); ++i) {
                            result[i].resize(classWithArrays.fVecVecFl[i].size());
                            for (size_t j = 0; j < classWithArrays.fVecVecFl[i].size(); ++j) {
                               result[i][j] = classWithArrays.fVecVecFl[i][j] + 1.0f;
                            }
                         }
                         return result;
                         )CODE");
   // Using a branch with an invalid C++ name will break the jitted execution
   // node = node.Alias("vecClassWithArrays_fArrFl", "vecClassWithArrays.fArrFl[3]");
   // node = node.Define("vecClassWithArrays_fArrFl_plus_1",
   //                    R"CODE(
   //                       std::vector<std::array<float, 3>> result(vecClassWithArrays_fArrFl.size());
   //                       for (size_t i = 0; i < vecClassWithArrays_fArrFl.size(); ++i) {
   //                          for (size_t j = 0; j < vecClassWithArrays_fArrFl[i].size(); ++j) {
   //                             result[i][j] = vecClassWithArrays_fArrFl[i][j] + 1.0f;
   //                          }
   //                       }
   //                       return result;
   //                      )CODE");

   // Take each column individually and check the content
   auto takeTopArrFl = node.Take<std::array<float, 3>>("topArrFl_plus_1");
   // Not supported: std::array<T, N> with T being a class type as top-level branch
   // auto takeTopArrArrFl = node.Take<ROOT::VecOps::RVec<Float_t>>("topArrArrFl");
   // auto takeTopArrVecFl = node.Take<ROOT::VecOps::RVec<Float_t>>("topArrVecFl");
   auto takeTopVecFl = node.Take<std::vector<float>>("topVecFl_plus_1");
   // Not supported: Could not find the real data member '_M_elems[3]' when constructing the branch 'topVecArrFl'
   // auto takeTopVecArrFl = node.Take<ROOT::VecOps::RVec<std::array<float, 3>>>("topVecArrFl");
   auto takeTopVecVecFl = node.Take<std::vector<std::vector<float>>>("topVecVecFl_plus_1");
   auto takeClassWithArrays = node.Take<ClassWithSequenceContainers>("classWithArrays_plus_1");
   auto takeClassWithArrays_fObjIndex = node.Take<UInt_t>("classWithArrays_fObjIndex_plus_1");
   // Using a branch with an invalid C++ name will break the jitted execution
   // auto takeClassWithArrays_fArrFl = node.Take<std::array<float, 3>>("classWithArrays_fArrFl_plus_1");
   // TODO: array of array is currently not properly handled
   // auto takeClassWithArrays_fArrArrFl = node.Take<std::array<std::array<float, 3>,
   // 3>>>("classWithArrays.fArrArrFl[3][3]");
   // TODO: array of vector as a data member is currently not properly handled
   // auto takeClassWithArrays_fArrVecFl =
   // node.Take<ROOT::VecOps::RVec<std::vector<float>>>("classWithArrays.fArrVecFl[3]");
   auto takeClassWithArrays_fVecFl = node.Take<std::vector<float>>("classWithArrays_fVecFl_plus_1");
   auto takeClassWithArrays_fVecVecFl = node.Take<std::vector<std::vector<float>>>("classWithArrays_fVecVecFl_plus_1");
   auto takeVecClassWithArrays =
      node.Take<ROOT::VecOps::RVec<ClassWithSequenceContainers>>("vecClassWithArrays_plus_1");
   // Using a branch with an invalid C++ name will break the jitted execution
   // auto takeVecClassWithArrays_fArrFl =
   //    node.Take<std::vector<std::array<Float_t, 3>>>("vecClassWithArrays_fArrFl_plus_1");
   // TODO: array of vector as a data member is currently not properly handled
   // auto takeVecClassWithArrays_fArrVecFl =
   // node.Take<ROOT::VecOps::RVec<std::array<std::vector<float>,3>>>("vecClassWithArrays.fArrVecFl[3]");
   // TODO: vector of vector throws `std::bad_alloc` currently
   // auto takeVecClassWithArrays_fVecFl =
   // node.Take<ROOT::VecOps::RVec<std::vector<float>>>("vecClassWithArrays.fVecFl"); auto
   // takeVecClassWithArrays_fVecVecFl =
   // node.Take<ROOT::VecOps::RVec<std::vector<std::vector<float>>>>("vecClassWithArrays.fVecVecFl");

   auto nEvents = takeTopArrFl->size();
   EXPECT_EQ(nEvents, 1);

   for (decltype(nEvents) i = 0; i < nEvents; ++i) {
      check_1d_coll(takeTopArrFl->at(i), data[i].fArrFl);
      check_1d_coll(takeTopVecFl->at(i), data[i].fVecFl);
      check_2d_coll(takeTopVecVecFl->at(i), data[i].fVecVecFl);
      check_class_with_arrays(takeClassWithArrays->at(i), data[i].fClassWithArrays);
      EXPECT_EQ(takeClassWithArrays_fObjIndex->at(i), data[i].fClassWithArrays.fObjIndex);
      // Using a branch with an invalid C++ name will break the jitted execution
      // check_1d_coll(takeClassWithArrays_fArrFl->at(i), data[i].fClassWithArrays.fArrFl);
      check_1d_coll(takeClassWithArrays_fVecFl->at(i), data[i].fClassWithArrays.fVecFl);
      check_2d_coll(takeClassWithArrays_fVecVecFl->at(i), data[i].fClassWithArrays.fVecVecFl);
      check_coll_class_with_arrays(takeVecClassWithArrays->at(i), data[i].fVecClassWithArrays);
      // Using a branch with an invalid C++ name will break the jitted execution
      // std::vector<std::array<float, 3>> expectedVecArrFl(data[i].fVecClassWithArrays.size());
      // for (size_t j = 0; j < data[i].fVecClassWithArrays.size(); ++j) {
      //    expectedVecArrFl[j] = data[i].fVecClassWithArrays[j].fArrFl;
      // }
      // check_2d_coll(takeVecClassWithArrays_fArrFl->at(i), expectedVecArrFl);
   }
}

INSTANTIATE_TEST_SUITE_P(Run, ClassWithSequenceContainersTest, ::testing::Values(true, false));

int main(int argc, char **argv)
{
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}
