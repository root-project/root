#include "CustomStruct.hxx"

#include <ROOT/RFieldVisitor.hxx>
#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleUtil.hxx>
#include <TFile.h>

#include "gtest/gtest.h"

#include <iostream>
#include <sstream>
#include <vector>

using ClusterSize_t = ROOT::Experimental::ClusterSize_t;
using RNTupleReader = ROOT::Experimental::RNTupleReader;
using RNTupleWriter = ROOT::Experimental::RNTupleWriter;
using RNTupleModel = ROOT::Experimental::RNTupleModel;
using RFieldBase = ROOT::Experimental::Detail::RFieldBase;
using RPrintVisitor = ROOT::Experimental::RPrintVisitor;
using RPrepareVisitor = ROOT::Experimental::RPrepareVisitor;

template <class T>
using RField = ROOT::Experimental::RField<T>;

namespace {

/**
 * An RAII wrapper around an open temporary file on disk. It cleans up the guarded file when the wrapper object
 * goes out of scope.
 */
class FileRaii {
private:
   std::string fPath;
public:
   explicit FileRaii(const std::string &path) : fPath(path) { }
   FileRaii(const FileRaii&) = delete;
   FileRaii& operator=(const FileRaii&) = delete;
   ~FileRaii() { std::remove(fPath.c_str()); }
   std::string GetPath() const { return fPath; }
};

} // anonymous namespace

TEST(RNtuplePrint, FullString)
{
   FileRaii fileGuard("test_ntuple_print_fullstring.root");
   {
      auto model = RNTupleModel::Create();
      auto fieldPt = model->MakeField<float>("pt");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "Staff", fileGuard.GetPath());
      *fieldPt = 5.0f;
      ntuple->Fill();
   }
   auto ntuple2 = RNTupleReader::Open("Staff", fileGuard.GetPath());
   std::ostringstream os;
   ntuple2->PrintInfo(ROOT::Experimental::ENTupleInfo::kSummary, os);
   std::string fString{std::string("")
       + "************************************ NTUPLE ************************************\n"
       + "* N-Tuple : Staff                                                              *\n"
       + "* Entries : 1                                                                  *\n"
       + "********************************************************************************\n"
       + "* Field 1   : pt (float)                                                       *\n"
       + "********************************************************************************\n"};
   EXPECT_EQ(fString, os.str());
}

TEST(RNtuplePrint, IntPrint)
{
   std::stringstream os;
   RPrintVisitor visitor(os);
   RField<int> testField("intTest");
   testField.AcceptVisitor(visitor, 1);
   std::string expected{std::string("")
       + "********************************************************************************\n"
       + "* Field 1   : intTest (std::int32_t)                                           *\n"};
   EXPECT_EQ(expected, os.str());
}

TEST(RNtuplePrint, FloatPrint)
{
   std::stringstream os;
   RPrintVisitor visitor(os);
   RField<float> testField("floatTest");
   testField.AcceptVisitor(visitor, 1);
   std::string expected{std::string("")
       + "********************************************************************************\n"
       + "* Field 1   : floatTest (float)                                                *\n"};
   EXPECT_EQ(expected, os.str());

}

TEST(RNtuplePrint, FloatTraverse)
{
   std::stringstream os;
   RPrintVisitor visitor(os, 'a');
   RField<float> testField("floatTest");
   testField.TraverseVisitor(visitor, 1);
   std::string expected{std::string("")
       + "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
       + "a Field 1   : floatTest (float)                                                a\n"};
   EXPECT_EQ(expected, os.str());
}

TEST(RNtuplePrint, VecAccept)
{
   std::stringstream os;
   RPrintVisitor visitor(os, 'a');
   RField<std::vector<float>> testField("floatTest");
   testField.AcceptVisitor(visitor, 1);
   std::string expected{std::string("")
       + "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
       + "a Field 1   : floatTest (std::vector<float>)                                   a\n"};
   EXPECT_EQ(expected, os.str());
}

TEST(RNtuplePrint, VecTraverse)
{
   std::stringstream os;
   RPrepareVisitor prepVisitor;
   RField<std::vector<float>> testField("floatVecTest");
   testField.TraverseVisitor(prepVisitor, 1);
   RPrintVisitor visitor(os, '$');
   visitor.SetDeepestLevel(prepVisitor.GetDeepestLevel());
   visitor.SetNumFields(prepVisitor.GetNumFields());
   testField.TraverseVisitor(visitor, 1);
   std::string expected{std::string("")
       + "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n"
       + "$ Field 1       : floatVecTest (std::vector<float>)                            $\n"
       + "$ |__Field 1.1  : float (float)                                                $\n"};
   EXPECT_EQ(expected, os.str());
}

TEST(RNtuplePrint, VecVecTraverse)
{
   std::stringstream os;
   RPrepareVisitor prepVisitor;
   RField<std::vector<std::vector<float>>> testField("floatVecVecTest");
   testField.TraverseVisitor(prepVisitor, 1);
   RPrintVisitor visitor(os, 'x');
   visitor.SetDeepestLevel(prepVisitor.GetDeepestLevel());
   visitor.SetNumFields(prepVisitor.GetNumFields());
   testField.TraverseVisitor(visitor, 1);
   std::string expected{std::string("")
       + "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n"
       + "x Field 1           : floatVecVecTest (std::vector<std::vector<float>>)        x\n"
       + "x |__Field 1.1      : std::vector<float> (std::vector<float>)                  x\n"
       + "x   |__Field 1.1.1  : float (float)                                            x\n"};
   EXPECT_EQ(expected, os.str());
}

TEST(RNtuplePrint, NarrowManyEntriesVecVecTraverse)
{
   std::stringstream os;
   RPrepareVisitor prepVisitor;
   RField<std::vector<std::vector<float>>> testField("floatVecVecTest");
   testField.TraverseVisitor(prepVisitor, 1);
   RPrintVisitor visitor(os, ' ', 30);
   visitor.SetDeepestLevel(prepVisitor.GetDeepestLevel());
   visitor.SetNumFields(prepVisitor.GetNumFields());
   testField.TraverseVisitor(visitor, 1);
   std::string expected{std::string("")
       + "                              \n"
       + "  Field 1         : floatV... \n"
       + "  |__Field 1.1    : std::v... \n"
       + "    |__Field 1... : float ... \n"};
   EXPECT_EQ(expected, os.str());
}

/* Currently the width can't be set by PrintInfo(). This test will be enabled when this feature is added.
TEST(RNTuplePrint, TooShort)
{
FileRaii fileGuard("test.root");
{
   auto model = RNTupleModel::Create();
   auto fieldPt = model->MakeField<float>("pt");
   auto ntuple = RNTupleWriter::Recreate(std::move(model), "Staff", "test.root");
}
auto ntuple2 = RNTupleReader::Open("Staff", "test.root");
std::ostringstream os;
ntuple2->PrintInfo(ROOT::Experimental::ENTupleInfo::kSummary, os, '+', 29);
std::string fString{"The width is too small! Should be at least 30.\n"};
EXPECT_EQ(fString, os.str());
}
*/

TEST(RNTupleShow, Empty)
{
   std::string rootFileName{"empty.root"};
   std::string ntupleName{"EmptyNTuple"};
   FileRaii fileGuard(rootFileName);
   {
      auto model = RNTupleModel::Create();
      auto ntuple = RNTupleWriter::Recreate(std::move(model), ntupleName, rootFileName);
      ntuple->Fill();
      ntuple->Fill();
   }
   auto model2 = RNTupleModel::Create();
   auto ntuple2 = RNTupleReader::Open(std::move(model2), ntupleName, rootFileName);
   
   std::ostringstream os;
   ntuple2->Show(0, ROOT::Experimental::ENTupleFormat::kJSON, os);
   std::string fString{"The NTuple is empty.\n"};
   EXPECT_EQ(fString, os.str());
   
   std::ostringstream os1;
   ntuple2->Show(1, ROOT::Experimental::ENTupleFormat::kJSON, os1);
   std::string fString1{"The NTuple is empty.\n"};
   EXPECT_EQ(fString1, os1.str());
}


TEST(RNTupleShow, BasicTypes)
{
   std::string rootFileName{"basictypes.root"};
   std::string ntupleName{"SimpleTypesNtuple"};
   FileRaii fileGuard(rootFileName);
   {
      auto model = RNTupleModel::Create();
      auto fieldPt = model->MakeField<float>("pt");
      auto fielddb = model->MakeField<double>("db");
      auto fieldint = model->MakeField<int>("int");
      auto fielduint = model->MakeField<unsigned>("uint");
      auto field64uint = model->MakeField<std::uint64_t>("uint64");
      auto fieldstring = model->MakeField<std::string>("string");
      auto fieldbool = model->MakeField<bool>("boolean");
      auto fieldchar = model->MakeField<uint8_t>("uint8");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), ntupleName, rootFileName);
      
      *fieldPt = 5.0f;
      *fielddb = 9.99;
      *fieldint = -4;
      *fielduint = 3;
      *field64uint = 44444444444ull;
      *fieldstring = "TestString";
      *fieldbool = true;
      *fieldchar = 97;
      ntuple->Fill();
      
      *fieldPt = 8.5f;
      *fielddb = 9.998;
      *fieldint = -94;
      *fielduint = -30;
      *field64uint = 2299994967294ull;
      *fieldstring = "TestString2";
      *fieldbool = false;
      *fieldchar = 98;
      ntuple->Fill();
   }
   auto model2 = RNTupleModel::Create();
   auto fieldPt2 = model2->MakeField<float>("pt");
   auto fielddb = model2->MakeField<double>("db");
   auto fieldint = model2->MakeField<int>("int");
   auto fielduint = model2->MakeField<unsigned>("uint");
   auto field64uint = model2->MakeField<std::uint64_t>("uint64");
   auto fieldstring = model2->MakeField<std::string>("string");
   auto fieldbool = model2->MakeField<bool>("boolean");
   auto fieldchar = model2->MakeField<uint8_t>("uint8");
   auto ntuple2 = RNTupleReader::Open(std::move(model2), ntupleName, rootFileName);
   
   std::ostringstream os;
   ntuple2->Show(0, ROOT::Experimental::ENTupleFormat::kJSON, os);
   std::string fString{ std::string("")
      + "{\n"
      + "  \"pt\": 5,\n"
      + "  \"db\": 9.99,\n"
      + "  \"int\": -4,\n"
      + "  \"uint\": 3,\n"
      + "  \"uint64\": 44444444444,\n"
      + "  \"string\": \"TestString\",\n"
      + "  \"boolean\": true,\n"
      + "  \"uint8\": 'a'\n"
      + "}\n" };
   EXPECT_EQ(fString, os.str());
   
   std::ostringstream os1;
   ntuple2->Show(1, ROOT::Experimental::ENTupleFormat::kJSON, os1);
   std::string fString1{ std::string("")
      + "{\n"
      + "  \"pt\": 8.5,\n"
      + "  \"db\": 9.998,\n"
      + "  \"int\": -94,\n"
      + "  \"uint\": 4294967266,\n"
      + "  \"uint64\": 2299994967294,\n"
      + "  \"string\": \"TestString2\",\n"
      + "  \"boolean\": false,\n"
      + "  \"uint8\": 'b'\n"
      + "}\n" };
   EXPECT_EQ(fString1, os1.str());
   
   std::ostringstream os2;
   ntuple2->Show(10, ROOT::Experimental::ENTupleFormat::kJSON, os2);
   std::string fString2{ "Index should be smaller than number of entries in ntuple.\n" };
   EXPECT_EQ(fString2, os2.str());
}

TEST(RNTupleShow, VectorFields)
{
   std::string rootFileName{"ShowVector.root"};
   std::string ntupleName{"VecNTuple"};
   FileRaii fileGuard(rootFileName);
   {
      auto model = RNTupleModel::Create();
      auto fieldIntVec = model->MakeField<std::vector<int>>("intVec");
      auto fieldFloatVecVec = model->MakeField<std::vector<std::vector<float>>>("floatVecVec");
      auto fieldBoolVecVec = model->MakeField<std::vector<std::vector<bool>>>("booleanVecVec");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), ntupleName, rootFileName);
      
      *fieldIntVec = std::vector<int>{4, 5, 6};
      *fieldFloatVecVec = std::vector<std::vector<float>>{std::vector<float>{0.1, 0.2}, std::vector<float>{1.1, 1.2}};
      *fieldBoolVecVec = std::vector<std::vector<bool>>{std::vector<bool>{true, false}, std::vector<bool>{false, false}};
      ntuple->Fill();
      
      fieldIntVec->emplace_back(7);
      fieldFloatVecVec->emplace_back(std::vector<float>{2.2, 2.3});
      fieldBoolVecVec->emplace_back(std::vector<bool>{false, true});
      ntuple->Fill();
   }
   auto model2 = RNTupleModel::Create();
   auto fieldIntVec = model2->MakeField<std::vector<int>>("intVec");
   auto fieldFloatVecVec = model2->MakeField<std::vector<std::vector<float>>>("floatVecVec");
   auto fieldBoolVecVec = model2->MakeField<std::vector<std::vector<bool>>>("booleanVecVec");
   auto ntuple2 = RNTupleReader::Open(std::move(model2), ntupleName, rootFileName);
   
   std::ostringstream os;
   ntuple2->Show(0, ROOT::Experimental::ENTupleFormat::kJSON, os);
   std::string fString{ std::string("")
      + "{\n"
      + "  \"intVec\": { 4, 5, 6 },\n"
      + "  \"floatVecVec\": { { 0.100000f, 0.200000f }, { 1.10000f, 1.20000f } },\n"
      + "  \"booleanVecVec\": { { true, false }, { false, false } }\n"
      + "}\n" };
   EXPECT_EQ(fString, os.str());
   
   std::ostringstream os1;
   ntuple2->Show(1, ROOT::Experimental::ENTupleFormat::kJSON, os1);
   std::string fString1{ std::string("")
      + "{\n"
      + "  \"intVec\": { 4, 5, 6, 7 },\n"
      + "  \"floatVecVec\": { { 0.100000f, 0.200000f }, { 1.10000f, 1.20000f }, { 2.20000f, 2.30000f } },\n"
      + "  \"booleanVecVec\": { { true, false }, { false, false }, { false, true } }\n"
      + "}\n" };
   EXPECT_EQ(fString1, os1.str());
}

TEST(RNTupleShow, ObjectFields)
{
   std::string rootFileName{"ShowObject.root"};
   std::string ntupleName{"ClassContainingNTuple"};
   FileRaii fileGuard(rootFileName);
   {
      auto model = RNTupleModel::Create();
      auto customStructfield = model->MakeField<CustomStruct>("CustomStruct");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), ntupleName, rootFileName);
      
      *customStructfield = CustomStruct(4.0f, std::vector<float>{0.1f, 0.2f, 0.3f}, std::vector<std::vector<float>>{{1.1f, 1.2f, 1.3f}, {2.1f, 2.2f, 2.3f}}, "ExampleString");
      ntuple->Fill();
      
      *customStructfield = CustomStruct(5.0f, std::vector<float>{3.1f, 3.2f, 3.3f}, std::vector<std::vector<float>>{{4.1f, 4.2f, 4.3f}, {5.1f, 5.2f, 5.3f}}, "AnotherString");
      ntuple->Fill();
   }
   auto model2 = RNTupleModel::Create();
   auto customStructfield = model2->MakeField<CustomStruct>("CustomStruct");
   auto ntuple2 = RNTupleReader::Open(std::move(model2), ntupleName, rootFileName);
   
   std::ostringstream os;
   ntuple2->Show(0, ROOT::Experimental::ENTupleFormat::kJSON, os);
   std::string fString{ std::string("")
      + "{\n"
      + "  \"CustomStruct\": \n"
      + "  {\n"
      + "    \"a\": 4,\n"
      + "    \"v1\": { 0.100000f, 0.200000f, 0.300000f },\n"
      + "    \"v2\": { { 1.10000f, 1.20000f, 1.30000f }, { 2.10000f, 2.20000f, 2.30000f } },\n"
      + "    \"s\": \"ExampleString\"\n"
      + "  }\n"
      + "}\n" };
   EXPECT_EQ(fString, os.str());
   
   std::ostringstream os1;
   ntuple2->Show(1, ROOT::Experimental::ENTupleFormat::kJSON, os1);
   std::string fString1{ std::string("")
      + "{\n"
      + "  \"CustomStruct\": \n"
      + "  {\n"
      + "    \"a\": 5,\n"
      + "    \"v1\": { 3.10000f, 3.20000f, 3.30000f },\n"
      + "    \"v2\": { { 4.10000f, 4.20000f, 4.30000f }, { 5.10000f, 5.20000f, 5.30000f } },\n"
      + "    \"s\": \"AnotherString\"\n"
      + "  }\n"
      + "}\n" };
   EXPECT_EQ(fString1, os1.str());
}

TEST(RNTupleShow, stdArrayAndClusterSize)
{
   std::string rootFileName{"arrayAndCluster.root"};
   std::string ntupleName{"ArrayAndClusterNTuple"};
   FileRaii fileGuard(rootFileName);
   {
      auto model = RNTupleModel::Create();
      auto Intarrayfield = model->MakeField<std::array<int, 2>>("IntArray");
      auto Floatarrayfield = model->MakeField<std::array<float, 3>>("FloatArray");
      auto Vecarrayfield = model->MakeField<std::array<std::vector<double>, 4>>("ArrayOfVec");
      auto StringArray = model->MakeField<std::array<std::string, 2>>("stringArray");
      auto ClusterSize = model->MakeField<ClusterSize_t>("ClusterSizeField");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), ntupleName, rootFileName);
      
      *Intarrayfield = {1, 3};
      *Floatarrayfield = {3.5f, 4.6f, 5.7f};
      *Vecarrayfield = {std::vector<double>{1, 2}, std::vector<double>{4, 5}, std::vector<double>{7, 8, 9}, std::vector<double>{11} };
      *StringArray = {"First", "Second"};
      *ClusterSize = ClusterSize_t(44);
      ntuple->Fill();
      
      *Intarrayfield = {2, 5};
      *Floatarrayfield = {2.3f, 5.7f, 11.13f};
      *Vecarrayfield = {std::vector<double>{17, 19}, std::vector<double>{23, 29}, std::vector<double>{31, 37, 41}, std::vector<double>{43} };
      *StringArray = {"Third", "Fourth"};
      *ClusterSize = ClusterSize_t(32);
      ntuple->Fill();
   }
   auto model2 = RNTupleModel::Create();
   auto Intarrayfield = model2->MakeField<std::array<int, 2>>("IntArray");
   auto Floatarrayfield = model2->MakeField<std::array<float, 3>>("FloatArray");
   auto Vecarrayfield = model2->MakeField<std::array<std::vector<double>, 4>>("ArrayOfVec");
   auto StringArray = model2->MakeField<std::array<std::string, 2>>("stringArray");
   auto ClusterSize = model2->MakeField<ClusterSize_t>("ClusterSizeField");
   auto ntuple2 = RNTupleReader::Open(std::move(model2), ntupleName, rootFileName);
   
   std::ostringstream os;
   ntuple2->Show(0, ROOT::Experimental::ENTupleFormat::kJSON, os);
   std::string fString{ std::string("")
      + "{\n"
      + "  \"IntArray\": [1, 3],\n"
      + "  \"FloatArray\": [3.50000f, 4.60000f, 5.70000f],\n"
      + "  \"ArrayOfVec\": [{ 1.0000000, 2.0000000 }, { 4.0000000, 5.0000000 }, { 7.0000000, 8.0000000, 9.0000000 }, { 11.000000 }],\n"
      + "  \"stringArray\": [\"First\", \"Second\"],\n"
      + "  \"ClusterSizeField\": 44\n"
      + "}\n"};
   EXPECT_EQ(fString, os.str());
   
   std::ostringstream os1;
   ntuple2->Show(1, ROOT::Experimental::ENTupleFormat::kJSON, os1);
   std::string fString1{ std::string("")
      + "{\n"
      + "  \"IntArray\": [2, 5],\n"
      + "  \"FloatArray\": [2.30000f, 5.70000f, 11.1300f],\n"
      + "  \"ArrayOfVec\": [{ 17.000000, 19.000000 }, { 23.000000, 29.000000 }, { 31.000000, 37.000000, 41.000000 }, { 43.000000 }],\n"
      + "  \"stringArray\": [\"Third\", \"Fourth\"],\n"
      + "  \"ClusterSizeField\": 32\n"
      + "}\n"};
   EXPECT_EQ(fString1, os1.str());
}

