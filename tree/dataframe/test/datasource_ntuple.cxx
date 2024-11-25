#include <ROOT/RDataFrame.hxx>
#include <ROOT/RNTupleDS.hxx>
#include <ROOT/RVec.hxx>

#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleWriter.hxx>
#include <ROOT/RPageStorage.hxx>

#include <NTupleStruct.hxx>

#include <gtest/gtest.h>

#include "ClassWithArrays.h"

#include <limits>

using ROOT::Experimental::RNTupleDS;
using ROOT::Experimental::RNTupleModel;
using ROOT::Experimental::RNTupleWriter;
using ROOT::Experimental::Internal::RPageSource;

namespace {

class FileRAII {
private:
   std::string fPath;

public:
   explicit FileRAII(const std::string &path) : fPath(path) {}
   FileRAII(const FileRAII &) = delete;
   FileRAII &operator=(const FileRAII &) = delete;
   ~FileRAII() { std::remove(fPath.c_str()); }
   std::string GetPath() const { return fPath; }
};

} // namespace

template <typename V1, typename V2>
void EXPECT_VEC_EQ(const V1 &v1, const V2 &v2)
{
   ASSERT_EQ(v1.size(), v2.size());
   for (std::size_t i = 0ul; i < v1.size(); ++i) {
      EXPECT_EQ(v1[i], v2[i]);
   }
}

class RNTupleDSTest : public ::testing::Test {
protected:
   std::string fFileName = "RNTupleDS_test.root";
   std::string fNtplName = "ntuple";

   void SetUp() override {
      auto model = RNTupleModel::Create();
      *model->MakeField<float>("pt") = 42;
      *model->MakeField<float>("energy") = 7;
      *model->MakeField<std::string>("tag") = "xyz";
      *model->MakeField<std::vector<float>>("jets") = std::vector<float>{1.f, 2.f};
      auto fldNnlo = model->MakeField<std::vector<std::vector<float>>>("nnlo");
      fldNnlo->push_back(std::vector<float>());
      fldNnlo->push_back(std::vector<float>{1.0});
      fldNnlo->push_back(std::vector<float>{1.0, 2.0, 4.0, 8.0});
      *model->MakeField<ROOT::RVecI>("rvec") = ROOT::RVecI{1, 2, 3};
      auto fldElectron = model->MakeField<Electron>("electron");
      fldElectron->pt = 137.0;
      auto fldVecElectron = model->MakeField<std::vector<Electron>>("VecElectron");
      fldVecElectron->push_back(*fldElectron);
      fldVecElectron->push_back(*fldElectron);
      {
         auto ntuple = RNTupleWriter::Recreate(std::move(model), fNtplName, fFileName);
         ntuple->Fill();
      }
   }

   void TearDown() override {
      std::remove(fFileName.c_str());
   }
};

TEST_F(RNTupleDSTest, ColTypeNames)
{
   RNTupleDS ds(fNtplName, fFileName);

   auto colNames = ds.GetColumnNames();
   ASSERT_EQ(15, colNames.size());

   EXPECT_TRUE(ds.HasColumn("pt"));
   EXPECT_TRUE(ds.HasColumn("energy"));
   EXPECT_TRUE(ds.HasColumn("rvec"));
   EXPECT_TRUE(ds.HasColumn("R_rdf_sizeof_nnlo"));
   EXPECT_TRUE(ds.HasColumn("electron"));
   EXPECT_TRUE(ds.HasColumn("electron.pt"));
   EXPECT_TRUE(ds.HasColumn("VecElectron"));
   EXPECT_TRUE(ds.HasColumn("R_rdf_sizeof_VecElectron"));
   EXPECT_TRUE(ds.HasColumn("VecElectron.pt"));
   EXPECT_TRUE(ds.HasColumn("R_rdf_sizeof_VecElectron.pt"));
   EXPECT_FALSE(ds.HasColumn("Address"));

   EXPECT_STREQ("std::string", ds.GetTypeName("tag").c_str());
   EXPECT_STREQ("float", ds.GetTypeName("energy").c_str());
   EXPECT_STREQ("std::size_t", ds.GetTypeName("R_rdf_sizeof_jets").c_str());
   EXPECT_STREQ("ROOT::VecOps::RVec<std::int32_t>", ds.GetTypeName("rvec").c_str());
}

TEST_F(RNTupleDSTest, NFiles)
{
   RNTupleDS ds(fNtplName, fFileName);

   EXPECT_EQ(1, ds.GetNFiles());
}

TEST_F(RNTupleDSTest, CardinalityColumn)
{
   auto df = ROOT::RDF::Experimental::FromRNTuple(fNtplName, fFileName);

   // Check that the special column #<collection> works without jitting...
   auto identity = [](std::size_t sz) { return sz; };
   auto max_njets = df.Define("njets", identity, {"R_rdf_sizeof_jets"}).Max<std::size_t>("njets");
   auto max_njets2 = df.Max<std::size_t>("#jets");
   auto max_rvec = df.Max<std::size_t>("#rvec");
   EXPECT_EQ(*max_njets, *max_njets2);
   EXPECT_EQ(*max_njets, 2);
   EXPECT_EQ(*max_rvec, 3);

   // ...and with jitting
   auto max_njets_jitted = df.Define("njets", "R_rdf_sizeof_jets").Max<std::size_t>("njets");
   auto max_njets_jitted2 = df.Define("njets", "#jets").Max<std::size_t>("njets");
   auto max_njets_jitted3 = df.Max("#jets");
   auto max_rvec2 = df.Max("#rvec");
   EXPECT_EQ(*max_njets_jitted, *max_njets_jitted2);
   EXPECT_EQ(*max_njets_jitted3, *max_njets_jitted2);
   EXPECT_EQ(2, *max_njets_jitted);
   EXPECT_EQ(3, *max_rvec2);
}

static void ReadTest(const std::string &name, const std::string &fname)
{
   auto df = ROOT::RDF::Experimental::FromRNTuple(name, fname);

   auto count = df.Count();
   auto sumpt = df.Sum<float>("pt");
   auto tag = df.Take<std::string>("tag");
   auto njets = df.Take<std::size_t>("R_rdf_sizeof_jets");
   auto sumjets = df.Sum<ROOT::RVec<float>>("jets");
   auto sumnnlosize = df.Sum<ROOT::RVec<std::size_t>>("R_rdf_sizeof_nnlo");
   auto sumvec = [](float red, const ROOT::RVec<ROOT::RVec<float>> &nnlo) {
      auto sum = 0.f;
      for (auto &v : nnlo)
         for (auto e : v)
            sum += e;
      return red + sum;
   };
   auto sumnnlo = df.Aggregate(sumvec, std::plus<float>{}, "nnlo", 0.f);
   auto rvec = df.Take<ROOT::RVecI>("rvec");
   auto vectorasrvec = df.Take<ROOT::RVecF>("jets");
   auto sumElectronPt = df.Aggregate([](float &acc, const Electron &e) { acc += e.pt; },
                                     [](float a, float b) { return a + b; }, "electron");
   auto sumVecElectronPt = df.Aggregate(
      [](float &acc, const ROOT::RVec<Electron> &ve) {
         for (const auto &e : ve)
            acc += e.pt;
      },
      [](float a, float b) { return a + b; }, "VecElectron");

   EXPECT_EQ(1ull, count.GetValue());
   EXPECT_DOUBLE_EQ(42.f, sumpt.GetValue());
   EXPECT_EQ(1ull, tag.GetValue().size());
   EXPECT_EQ(std::string("xyz"), tag.GetValue()[0]);
   EXPECT_EQ(1ull, njets.GetValue().size());
   EXPECT_EQ(2u, njets.GetValue()[0]);
   EXPECT_EQ(3.f, sumjets.GetValue());
   EXPECT_EQ(16.f, sumnnlo.GetValue());
   EXPECT_EQ(5u, sumnnlosize.GetValue());
   EXPECT_TRUE(All(rvec->at(0) == ROOT::RVecI{1, 2, 3}));
   EXPECT_TRUE(All(vectorasrvec->at(0) == ROOT::RVecF{1.f, 2.f}));
   EXPECT_FLOAT_EQ(137.0, sumElectronPt.GetValue());
   EXPECT_FLOAT_EQ(2. * 137.0, sumVecElectronPt.GetValue());
}

static void ChainTest(const std::string &name, const std::string &fname)
{
   auto df1 = ROOT::RDataFrame(std::make_unique<RNTupleDS>(name, std::vector<std::string>{fname}));
   EXPECT_DOUBLE_EQ(42.0, df1.Sum<float>("pt").GetValue());

   auto df2 = ROOT::RDataFrame(std::make_unique<RNTupleDS>(name, std::vector<std::string>{fname, fname}));
   EXPECT_DOUBLE_EQ(84.0, df2.Sum<float>("pt").GetValue());

   std::vector<std::string> fileNames(1000, fname);
   auto df1000 = ROOT::RDataFrame(std::make_unique<RNTupleDS>(name, fileNames));
   EXPECT_DOUBLE_EQ(42000.0, df1000.Sum<float>("pt").GetValue());

   FileRAII guardFile1("RNTupleDS_test_chain_1.root");
   FileRAII guardFile2("RNTupleDS_test_chain_2.root");
   FileRAII guardFile3("RNTupleDS_test_chain_3.root");

   {
      auto model = RNTupleModel::Create();
      auto fldElectron = model->MakeField<Electron>("e");
      auto writer = RNTupleWriter::Recreate(std::move(model), "chain", guardFile1.GetPath());
      fldElectron->pt = 1.0;
      writer->Fill();
   }
   {
      auto model = RNTupleModel::Create();
      // Add dummy field to ensure that the Electron fields in the files have different field IDs
      model->MakeField<int>("dummy1");
      auto fldElectron = model->MakeField<Electron>("e");
      auto writer = RNTupleWriter::Recreate(std::move(model), "chain", guardFile2.GetPath());
      // empty file in chain, no entries
   }
   {
      auto model = RNTupleModel::Create();
      model->MakeField<int>("dummy1");
      model->MakeField<int>("dummy2");
      auto fldElectron = model->MakeField<Electron>("e");
      auto writer = RNTupleWriter::Recreate(std::move(model), "chain", guardFile3.GetPath());
      fldElectron->pt = 2.0;
      writer->Fill();
      fldElectron->pt = 3.0;
      writer->Fill();
   }

   auto df3 = ROOT::RDataFrame(std::make_unique<RNTupleDS>(
      "chain", std::vector<std::string>{guardFile1.GetPath(), guardFile2.GetPath(), guardFile3.GetPath()}));
   EXPECT_EQ(3, df3.Describe().GetNFiles());
   auto sumElectronPt =
      df3.Aggregate([](float &acc, const Electron &e) { acc += e.pt; }, [](float a, float b) { return a + b; }, "e");
   EXPECT_FLOAT_EQ(6.0, sumElectronPt.GetValue());
   // Trigger the event loop again
   EXPECT_FLOAT_EQ(6.0, sumElectronPt.GetValue());
}

TEST_F(RNTupleDSTest, Read)
{
   ReadTest(fNtplName, fFileName);
}

TEST_F(RNTupleDSTest, Chain)
{
   ChainTest(fNtplName, fFileName);
}

#ifdef R__USE_IMT
struct IMTRAII {
   IMTRAII() { ROOT::EnableImplicitMT(); }
   ~IMTRAII() { ROOT::DisableImplicitMT(); }
};

TEST_F(RNTupleDSTest, ReadMT)
{
   IMTRAII _;

   ReadTest(fNtplName, fFileName);
}

TEST_F(RNTupleDSTest, ChainMT)
{
   IMTRAII _;

   ChainTest(fNtplName, fFileName);
}

TEST_F(RNTupleDSTest, ChainTailScheduling)
{
   IMTRAII _;

   FileRAII guardFile1("RNTupleDS_test_chain_tail_scheduling_1.root");
   FileRAII guardFile2("RNTupleDS_test_chain_tail_scheduling_2.root");
   FileRAII guardFile3("RNTupleDS_test_chain_tail_scheduling_3.root");

   {
      auto model = RNTupleModel::Create();
      auto ptrX = model->MakeField<int>("x");
      auto writer = RNTupleWriter::Recreate(std::move(model), "chain", guardFile1.GetPath());
      for (unsigned i = 0; i < 2; ++i) {
         *ptrX = i;
         writer->Fill();
         writer->CommitCluster();
      }
   }
   {
      auto model = RNTupleModel::Create();
      model->MakeField<int>("x");
      auto writer = RNTupleWriter::Recreate(std::move(model), "chain", guardFile2.GetPath());
      // Empty file
   }
   {
      auto model = RNTupleModel::Create();
      auto ptrX = model->MakeField<int>("x");
      auto writer = RNTupleWriter::Recreate(std::move(model), "chain", guardFile3.GetPath());
      for (unsigned i = 0; i < 11; ++i) {
         *ptrX = i;
         writer->Fill();
         writer->CommitCluster();
      }
   }

   auto df = ROOT::RDataFrame(std::make_unique<RNTupleDS>(
      "chain", std::vector<std::string>{guardFile1.GetPath(), guardFile2.GetPath(), guardFile3.GetPath()}));
   EXPECT_EQ(3, df.Describe().GetNFiles());
   auto sumX = df.Aggregate([](int &acc, int x) { acc += x; }, [](int a, int b) { return a + b; }, "x");
   EXPECT_EQ(56, sumX.GetValue());
}
#endif

const static std::array<ROOT::RVec<std::array<ROOT::RVecI, 3>>, 3> arraysDatasetCol4El{
   ROOT::RVec<std::array<ROOT::RVecI, 3>>{
      {ROOT::RVecI{1, 2}, ROOT::RVecI{4, 5, 6}, ROOT::RVecI{42, 43, 44, 45}},
   },
   ROOT::RVec<std::array<ROOT::RVecI, 3>>{
      {ROOT::RVecI{55, 66}, ROOT::RVecI{-18, 33, std::numeric_limits<std::int32_t>::max()},
       ROOT::RVecI{42, std::numeric_limits<std::int32_t>::min(), 44, 1888}},
      {ROOT::RVecI{10, 11}, ROOT::RVecI{-32, -33, -34}, ROOT::RVecI{2953, -20, 343212}},
   },
   ROOT::RVec<std::array<ROOT::RVecI, 3>>{
      {ROOT::RVecI{-32, -33, -34}, ROOT::RVecI{42, -43, 44, -45}, ROOT::RVecI{1}},
      {ROOT::RVecI{0, 0, std::numeric_limits<std::int32_t>::min(), 42}, ROOT::RVecI{-32, -33, -34},
       ROOT::RVecI{30000, 40000, 50000}},
      {ROOT::RVecI{0, 0, std::numeric_limits<std::int32_t>::min(), 42}, ROOT::RVecI{-32, -33, -34},
       ROOT::RVecI{std::numeric_limits<std::int32_t>::min(), std::numeric_limits<std::int32_t>::max(),
                   std::numeric_limits<std::int32_t>::min()}},
   }};

class RNTupleDSArraysDataset : public ::testing::Test {
protected:
   std::string fFileName = "rntupleds_arrays_dataset.root";
   std::string fNtplName = "ntuple";

   void SetUp() override
   {
      auto model = RNTupleModel::Create();
      auto col1_arr = model->MakeField<std::array<int, 3>>("col1_arr");
      auto col2_arr_rvec = model->MakeField<std::array<ROOT::RVecI, 3>>("col2_arr_rvec");
      auto col3_rvec_arr = model->MakeField<ROOT::RVec<std::array<int, 3>>>("col3_rvec_arr");
      auto col4_arr_rvec_arr_rvec =
         model->MakeField<std::array<ROOT::RVec<std::array<ROOT::RVecI, 3>>, 3>>("col4_arr_rvec_arr_rvec");
      auto col5_class_with_arrays = model->MakeField<ClassWithArrays>("col5_class_with_arrays");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), fNtplName, fFileName);
      for (int i = 1; i <= 5; i++) {

         *col1_arr = {1 * i, 2 * i, 3 * i};

         *col2_arr_rvec = {ROOT::RVecI{i}, ROOT::RVecI{1 * i, 2 * i, 3 * i},
                           ROOT::RVecI{-1 * i, -2 * i, -3 * i, -4 * i, -5 * i}};

         ROOT::RVec<std::array<int, 3>> rvecVal;
         rvecVal.reserve(i);
         for (auto j = 0; j < i; j++) {
            rvecVal.emplace_back(std::array<int, 3>{1 * i, 2 * i, 3 * i});
         }
         *col3_rvec_arr = rvecVal;

         *col4_arr_rvec_arr_rvec = arraysDatasetCol4El;

         std::array<float, 3> col5_member_1{42.f, std::numeric_limits<float>::max(), 0.f};
         std::array<ROOT::RVecF, 3> col5_member_2{ROOT::RVecF{1.f}, ROOT::RVecF{2.f, 3.f}, ROOT::RVecF{4.f, 5.f, 6.f}};
         ROOT::RVec<std::array<float, 3>> col5_member_3{col5_member_1, col5_member_1, col5_member_1, col5_member_1};
         *col5_class_with_arrays =
            ClassWithArrays{std::move(col5_member_1), std::move(col5_member_2), std::move(col5_member_3)};

         ntuple->Fill();
      }
   }

   void TearDown() override { std::remove(fFileName.c_str()); }
};

void ReadArraysTest(const std::string &name, const std::string &fname)
{
   // These tests use the columns that contain std::array data on disk as RVecs
   // reading them into RVecs and checking their values.
   auto df = ROOT::RDF::Experimental::FromRNTuple(name, fname);

   auto count = df.Count();

   auto col1Take = df.Take<ROOT::RVecI>("col1_arr");
   auto col2Take = df.Take<ROOT::RVec<ROOT::RVecI>>("col2_arr_rvec");
   auto col3Take = df.Take<ROOT::RVec<ROOT::RVecI>>("col3_rvec_arr");
   auto col4Take = df.Take<ROOT::RVec<ROOT::RVec<ROOT::RVec<ROOT::RVecI>>>>("col4_arr_rvec_arr_rvec");
   auto col5TakeArr = df.Take<ROOT::RVecF>("col5_class_with_arrays.fArr");
   auto col5TakeArrRVec = df.Take<ROOT::RVec<ROOT::RVecF>>("col5_class_with_arrays.fArrRVec");
   auto col5TakeRVecArr = df.Take<ROOT::RVec<ROOT::RVecF>>("col5_class_with_arrays.fRVecArr");

   EXPECT_EQ(5ull, count.GetValue());

   auto col1Val = col1Take.GetValue();
   EXPECT_EQ(col1Val.size(), 5);
   for (int i = 1; i <= 5; i++) {
      EXPECT_VEC_EQ(col1Val[i - 1], ROOT::RVecI{1 * i, 2 * i, 3 * i});
   }

   auto col2Val = col2Take.GetValue();
   EXPECT_EQ(col2Val.size(), 5);
   for (int i = 1; i <= 5; i++) {
      EXPECT_VEC_EQ(col2Val[i - 1][0], ROOT::RVecI{i});
      EXPECT_VEC_EQ(col2Val[i - 1][1], ROOT::RVecI{1 * i, 2 * i, 3 * i});
      EXPECT_VEC_EQ(col2Val[i - 1][2], ROOT::RVecI{-1 * i, -2 * i, -3 * i, -4 * i, -5 * i});
   }

   auto col3Val = col3Take.GetValue();
   EXPECT_EQ(col3Val.size(), 5);
   for (int i = 1; i <= 5; i++) {
      for (auto j = 0; j < i; j++) {
         EXPECT_VEC_EQ(col3Val[i - 1][j], std::array<int, 3>{1 * i, 2 * i, 3 * i});
      }
   }

   auto col4Val = col4Take.GetValue();
   EXPECT_EQ(col4Val.size(), 5);

   // Each event contains the values of arraysDatasetCol4El in column 4
   for (unsigned i = 0; i < 5; i++) {
      // Unpack the three top-level std::arrays
      const auto &arr1 = col4Val[i][0];
      const auto &arr2 = col4Val[i][1];
      const auto &arr3 = col4Val[i][2];

      // The first contains an RVec with only one element
      for (unsigned k = 0; k < arr1.size(); k++) {
         EXPECT_VEC_EQ(arr1[0][k], arraysDatasetCol4El[0][0][k]);
      }

      // The others contain more than one element in their sub-level RVecs
      for (unsigned j = 0; j < arr2.size(); j++) {
         for (unsigned k = 0; k < arr2[j].size(); k++) {
            EXPECT_VEC_EQ(arr2[j][k], arraysDatasetCol4El[1][j][k]);
         }
      }

      for (unsigned j = 0; j < arr3.size(); j++) {
         for (unsigned k = 0; k < arr3[j].size(); k++) {
            EXPECT_VEC_EQ(arr3[j][k], arraysDatasetCol4El[2][j][k]);
         }
      }
   }

   auto col5ValArr = col5TakeArr.GetValue();
   auto col5ValArrRVec = col5TakeArrRVec.GetValue();
   auto col5ValRVecArr = col5TakeRVecArr.GetValue();
   EXPECT_EQ(col5ValArr.size(), 5);
   EXPECT_EQ(col5ValArrRVec.size(), 5);
   EXPECT_EQ(col5ValRVecArr.size(), 5);

   std::array<float, 3> col5_member_1{42.f, std::numeric_limits<float>::max(), 0.f};
   for (const auto &arr : col5ValArr) {
      EXPECT_VEC_EQ(arr, col5_member_1);
   }

   std::array<ROOT::RVecF, 3> col5_member_2{ROOT::RVecF{1.f}, ROOT::RVecF{2.f, 3.f}, ROOT::RVecF{4.f, 5.f, 6.f}};
   for (const auto &arr : col5ValArrRVec) {
      for (unsigned i = 0; i < 3; i++) {
         EXPECT_VEC_EQ(arr[i], col5_member_2[i]);
      }
   }

   ROOT::RVec<std::array<float, 3>> col5_member_3{col5_member_1, col5_member_1, col5_member_1, col5_member_1};
   for (const auto &arr : col5ValRVecArr) {
      for (unsigned i = 0; i < 4; i++) {
         EXPECT_VEC_EQ(arr[i], col5_member_3[i]);
      }
   }
}

TEST_F(RNTupleDSArraysDataset, Read)
{
   ReadArraysTest(fNtplName, fFileName);
}

#ifdef R__USE_IMT

TEST_F(RNTupleDSArraysDataset, ReadMT)
{
   IMTRAII _;

   ReadArraysTest(fNtplName, fFileName);
}
#endif

void UseArraysAsRVec(const std::string &name, const std::string &fname)
{
   // These tests use the columns that contain std::array data on disk as RVecs
   // passing them as arguments to functions that expect RVecs.
   auto df = ROOT::RDF::Experimental::FromRNTuple(name, fname);

   auto df1 = df.Define("col1_short", [](const ROOT::RVecI &arr) { return ROOT::VecOps::Take(arr, 2); }, {"col1_arr"});
   auto take1 = df1.Take<ROOT::RVecI>("col1_short");

   // Exercise jitting
   auto df2 = df1.Define("sum_col2",
                         "int sum = 0; for (const auto &v: col3_rvec_arr) sum += ROOT::VecOps::Sum(v); return sum;");
   auto take2 = df2.Take<int>("sum_col2");

   auto take1Val = take1.GetValue();
   for (int i = 0; i < 5; i++) {
      EXPECT_VEC_EQ(take1Val[i], ROOT::RVecI{(i + 1), (i + 1) * 2});
   }

   std::array<int, 5> expectedVals{6, 24, 54, 96, 150};
   EXPECT_VEC_EQ(take2.GetValue(), expectedVals);
}

TEST_F(RNTupleDSArraysDataset, UseArrays)
{
   UseArraysAsRVec(fNtplName, fFileName);
}

#ifdef R__USE_IMT

TEST_F(RNTupleDSArraysDataset, UseArraysMT)
{
   IMTRAII _;

   UseArraysAsRVec(fNtplName, fFileName);
}
#endif

void UseArraySizeColumn(const std::string &name, const std::string &fname)
{
   // These tests use the columns that contain std::array data on disk as RVecs
   // checking the size of the collection with the R_rdf_sizeof_* columns
   auto df = ROOT::RDF::Experimental::FromRNTuple(name, fname);

   auto sizeOfCol1 = df.Take<std::size_t>("R_rdf_sizeof_col1_arr");
   // Use # here to exercise that too
   auto sizeOfCol2 = df.Take<ROOT::RVec<std::size_t>>("#col2_arr_rvec");
   auto sizeOfCol4 = df.Take<ROOT::RVec<ROOT::RVec<ROOT::RVec<std::size_t>>>>("R_rdf_sizeof_col4_arr_rvec_arr_rvec");

   EXPECT_VEC_EQ(sizeOfCol1.GetValue(), std::array<std::size_t, 5>{3, 3, 3, 3, 3});

   for (const auto &sizeVec : sizeOfCol2) {
      EXPECT_VEC_EQ(sizeVec, ROOT::RVecULL{1, 3, 5});
   }

   // The sizes of elements in col4 are
   // { { { 2, 3, 4 } }, { { 2, 3, 4 }, { 2, 3, 3 } }, { { 3, 4, 1 }, { 4, 3, 3 }, { 4, 3, 3 } } }
   auto sizeOfCol4Val = sizeOfCol4.GetValue();
   for (unsigned i = 0; i < 5; i++) {
      // Unpack the three top-level vectors
      const auto &rvec1 = sizeOfCol4Val[i][0];
      const auto &rvec2 = sizeOfCol4Val[i][1];
      const auto &rvec3 = sizeOfCol4Val[i][2];

      // The first contains an RVec with only one element
      EXPECT_VEC_EQ(rvec1[0], ROOT::RVecULL{2, 3, 4});

      // The others contain more than one element in their sub-level RVecs
      ROOT::RVec<ROOT::RVecULL> expectedRVecs2{{2, 3, 4}, {2, 3, 3}};
      for (unsigned j = 0; j < rvec2.size(); j++) {
         EXPECT_VEC_EQ(rvec2[j], expectedRVecs2[j]);
      }

      ROOT::RVec<ROOT::RVecULL> expectedRVecs3{{3, 4, 1}, {4, 3, 3}, {4, 3, 3}};
      for (unsigned j = 0; j < rvec3.size(); j++) {
         EXPECT_VEC_EQ(rvec3[j], expectedRVecs3[j]);
      }
   }
}

TEST_F(RNTupleDSArraysDataset, UseArraySizeColumn)
{
   UseArraySizeColumn(fNtplName, fFileName);
}

#ifdef R__USE_IMT

TEST_F(RNTupleDSArraysDataset, UseArraySizeColumnMT)
{
   IMTRAII _;

   UseArraySizeColumn(fNtplName, fFileName);
}
#endif
