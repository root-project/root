#include <ROOT/RFieldVisitor.hxx>
#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleModel.hxx>

#include <TFile.h>

#include "gtest/gtest.h"

#include <exception>
#include <iostream>
#include <sstream>
#include <vector>

#include "CustomStruct.hxx"

using RNTupleReader = ROOT::Experimental::RNTupleReader;
using RNTupleWriter = ROOT::Experimental::RNTupleWriter;
using RNTupleModel = ROOT::Experimental::RNTupleModel;
using RFieldBase = ROOT::Experimental::Detail::RFieldBase;
using RPrintSchemaVisitor = ROOT::Experimental::RPrintSchemaVisitor;
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

TEST(RNtuplePrint, Int)
{
   std::stringstream os;
   RPrintSchemaVisitor visitor(os);
   RField<int> testField("intTest");
   testField.AcceptVisitor(visitor);
   std::string expected{std::string("")
       + "* Field 1   : intTest (std::int32_t)                                           *\n"};
   EXPECT_EQ(expected, os.str());
}

TEST(RNtuplePrint, Float)
{
   std::stringstream os;
   RPrintSchemaVisitor visitor(os, 'a');
   RField<float> testField("floatTest");
   testField.AcceptVisitor(visitor);
   std::string expected{std::string("")
       + "a Field 1   : floatTest (float)                                                a\n"};
   EXPECT_EQ(expected, os.str());
}

TEST(RNtuplePrint, Vector)
{
   std::stringstream os;
   RPrepareVisitor prepVisitor;
   RField<std::vector<float>> testField("floatVecTest");
   testField.AcceptVisitor(prepVisitor);
   RPrintSchemaVisitor visitor(os, '$');
   visitor.SetDeepestLevel(prepVisitor.GetDeepestLevel());
   visitor.SetNumFields(prepVisitor.GetNumFields());
   testField.AcceptVisitor(visitor);
   std::string expected{std::string("")
       + "$ Field 1       : floatVecTest (std::vector<float>)                            $\n"
       + "$   Field 1.1   : float (float)                                                $\n"};
   EXPECT_EQ(expected, os.str());
}

TEST(RNtuplePrint, VectorNested)
{
   std::stringstream os;
   RPrepareVisitor prepVisitor;
   RField<std::vector<std::vector<float>>> testField("floatVecVecTest");
   testField.AcceptVisitor(prepVisitor);
   RPrintSchemaVisitor visitor(os, 'x');
   visitor.SetDeepestLevel(prepVisitor.GetDeepestLevel());
   visitor.SetNumFields(prepVisitor.GetNumFields());
   testField.AcceptVisitor(visitor);
   std::string expected{std::string("")
       + "x Field 1           : floatVecVecTest (std::vector<std::vector<float>>)        x\n"
       + "x   Field 1.1       : std::vector<float> (std::vector<float>)                  x\n"
       + "x     Field 1.1.1   : float (float)                                            x\n"};
   EXPECT_EQ(expected, os.str());
}

TEST(RNtuplePrint, NarrowManyEntriesVecVecTraverse)
{
   std::stringstream os;
   RPrepareVisitor prepVisitor;
   RField<std::vector<std::vector<float>>> testField("floatVecVecTest");
   testField.AcceptVisitor(prepVisitor);
   RPrintSchemaVisitor visitor(os, ' ', 25);
   visitor.SetDeepestLevel(prepVisitor.GetDeepestLevel());
   visitor.SetNumFields(prepVisitor.GetNumFields());
   testField.AcceptVisitor(visitor);
   std::string expected{std::string("")
       + "  Field 1    : floatV... \n"
       + "    Field... : std::v... \n"
       + "      Fie... : float ... \n"};
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
   std::string rootFileName{"test_ntuple_show_empty.root"};
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
   std::string fString{"{}\n"};
   EXPECT_EQ(fString, os.str());

   std::ostringstream os1;
   ntuple2->Show(1, ROOT::Experimental::ENTupleFormat::kJSON, os1);
   std::string fString1{"{}\n"};
   EXPECT_EQ(fString1, os1.str());
}


TEST(RNTupleShow, BasicTypes)
{
   std::string rootFileName{"test_ntuple_show_basictypes.root"};
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
      + "  \"uint8\": 97\n"
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
      + "  \"uint8\": 98\n"
      + "}\n" };
   EXPECT_EQ(fString1, os1.str());

   // TODO(jblomer): this should fail to an exception instead
   EXPECT_DEATH(ntuple2->Show(2), ".*");
}

TEST(RNTupleShow, Vectors)
{
   std::string rootFileName{"test_ntuple_show_vector.root"};
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
      *fieldBoolVecVec = std::vector<std::vector<bool>>{std::vector<bool>{false, true, false}, std::vector<bool>{false, true}, std::vector<bool>{true, false, false }};
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
      + "  \"intVec\": [4, 5, 6],\n"
      + "  \"floatVecVec\": [[0.1, 0.2], [1.1, 1.2]],\n"
      + "  \"booleanVecVec\": [[false, true, false], [false, true], [true, false, false]]\n"
      + "}\n" };
   EXPECT_EQ(fString, os.str());

   std::ostringstream os1;
   ntuple2->Show(1, ROOT::Experimental::ENTupleFormat::kJSON, os1);
   std::string fString1{ std::string("")
      + "{\n"
      + "  \"intVec\": [4, 5, 6, 7],\n"
      + "  \"floatVecVec\": [[0.1, 0.2], [1.1, 1.2], [2.2, 2.3]],\n"
      + "  \"booleanVecVec\": [[false, true, false], [false, true], [true, false, false], [false, true]]\n"
      + "}\n" };
   EXPECT_EQ(fString1, os1.str());
}

TEST(RNTupleShow, Arrays)
{
   std::string rootFileName{"test_ntuple_show_array.root"};
   std::string ntupleName{"Arrays"};
   FileRaii fileGuard(rootFileName);
   {
      auto model = RNTupleModel::Create();
      auto Intarrayfield = model->MakeField<std::array<int, 2>>("IntArray");
      auto Floatarrayfield = model->MakeField<std::array<float, 3>>("FloatArray");
      auto Vecarrayfield = model->MakeField<std::array<std::vector<double>, 4>>("ArrayOfVec");
      auto StringArray = model->MakeField<std::array<std::string, 2>>("stringArray");
      auto arrayOfArray = model->MakeField<std::array<std::array<bool, 2>, 3>>("ArrayOfArray");
      auto arrayVecfield = model->MakeField<std::vector<std::array<float, 2>>>("VecOfArray");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), ntupleName, rootFileName);

      *Intarrayfield = {1, 3};
      *Floatarrayfield = {3.5f, 4.6f, 5.7f};
      *Vecarrayfield = {std::vector<double>{1, 2}, std::vector<double>{4, 5}, std::vector<double>{7, 8, 9}, std::vector<double>{11} };
      *StringArray = {"First", "Second"};
      *arrayOfArray = { std::array<bool,2>{ true, false }, std::array<bool,2>{ false, true }, std::array<bool,2>{ false, false } };
      *arrayVecfield = { std::array<float, 2>{ 0, 1 }, std::array<float, 2>{ 2, 3 }, std::array<float, 2>{ 4, 5 } };
      ntuple->Fill();

      *Intarrayfield = {2, 5};
      *Floatarrayfield = {2.3f, 5.7f, 11.13f};
      *Vecarrayfield = {std::vector<double>{17, 19}, std::vector<double>{23, 29}, std::vector<double>{31, 37, 41}, std::vector<double>{43} };
      *StringArray = {"Third", "Fourth"};
      *arrayOfArray = { std::array<bool,2>{ true, true }, std::array<bool,2>{ false, true }, std::array<bool,2>{ true, true } };
      *arrayVecfield = { std::array<float, 2>{ 6, 7 }, std::array<float, 2>{ 8, 9 } };
      ntuple->Fill();
   }
   auto model2 = RNTupleModel::Create();
   auto Intarrayfield = model2->MakeField<std::array<int, 2>>("IntArray");
   auto Floatarrayfield = model2->MakeField<std::array<float, 3>>("FloatArray");
   auto Vecarrayfield = model2->MakeField<std::array<std::vector<double>, 4>>("ArrayOfVec");
   auto StringArray = model2->MakeField<std::array<std::string, 2>>("stringArray");
   auto arrayOfArray = model2->MakeField<std::array<std::array<bool, 2>, 3>>("ArrayOfArray");
   auto arrayVecfield = model2->MakeField<std::vector<std::array<float, 2>>>("VecOfArray");
   auto ntuple2 = RNTupleReader::Open(std::move(model2), ntupleName, rootFileName);

   std::ostringstream os;
   ntuple2->Show(0, ROOT::Experimental::ENTupleFormat::kJSON, os);
   std::string fString{ std::string("")
      + "{\n"
      + "  \"IntArray\": [1, 3],\n"
      + "  \"FloatArray\": [3.5, 4.6, 5.7],\n"
      + "  \"ArrayOfVec\": [[1, 2], [4, 5], [7, 8, 9], [11]],\n"
      + "  \"stringArray\": [\"First\", \"Second\"],\n"
      + "  \"ArrayOfArray\": [[true, false], [false, true], [false, false]],\n"
      + "  \"VecOfArray\": [[0, 1], [2, 3], [4, 5]]\n"
      + "}\n"};
   EXPECT_EQ(fString, os.str());

   std::ostringstream os1;
   ntuple2->Show(1, ROOT::Experimental::ENTupleFormat::kJSON, os1);
   std::string fString1{ std::string("")
      + "{\n"
      + "  \"IntArray\": [2, 5],\n"
      + "  \"FloatArray\": [2.3, 5.7, 11.13],\n"
      + "  \"ArrayOfVec\": [[17, 19], [23, 29], [31, 37, 41], [43]],\n"
      + "  \"stringArray\": [\"Third\", \"Fourth\"],\n"
      + "  \"ArrayOfArray\": [[true, true], [false, true], [true, true]],\n"
      + "  \"VecOfArray\": [[6, 7], [8, 9]]\n"
      + "}\n"};
   EXPECT_EQ(fString1, os1.str());
}

TEST(RNTupleShow, Objects)
{
   std::string rootFileName{"test_ntuple_show_object.root"};
   std::string ntupleName{"Objects"};
   FileRaii fileGuard(rootFileName);
   {
      auto model = RNTupleModel::Create();
      auto customStructfield = model->MakeField<CustomStruct>("CustomStruct");
      auto customStructVec = model->MakeField<std::vector<CustomStruct>>("CustomStructVec");
      auto customStructArray = model->MakeField<std::array<CustomStruct, 2>>("CustomStructArray");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), ntupleName, rootFileName);

      *customStructfield = CustomStruct{4.1f, std::vector<float>{0.1f, 0.2f, 0.3f}, std::vector<std::vector<float>>{{1.1f, 1.2f, 1.3f}, {2.1f, 2.2f, 2.3f}}, "Example1String"};
      *customStructVec = {
         CustomStruct{4.2f, std::vector<float>{0.1f, 0.2f, 0.3f}, std::vector<std::vector<float>>{{1.1f, 1.3f}, {2.1f, 2.2f, 2.3f}}, "Example2String"},
         CustomStruct{4.3f, std::vector<float>{0.1f, 0.2f, 0.3f}, std::vector<std::vector<float>>{{1.1f, 1.2f, 1.3f}, {2.1f, 2.3f}}, "Example3String"},
         CustomStruct{4.4f, std::vector<float>{0.1f, 0.3f}, std::vector<std::vector<float>>{{1.1f, 1.2f, 1.3f}, {2.1f, 2.2f, 2.3f}}, "Example4String"}
      };
      *customStructArray = {
      CustomStruct{4.5f, std::vector<float>{0.1f, 0.2f, 0.3f}, std::vector<std::vector<float>>{{1.1f, 1.3f}, {2.1f, 2.2f, 2.3f}}, "AnotherString1"},
      CustomStruct{4.6f, std::vector<float>{0.1f, 0.2f, 0.3f}, std::vector<std::vector<float>>{{1.1f, 1.2f, 1.3f}, {2.1f, 2.3f}}, "AnotherString2"}
      };
      ntuple->Fill();
   }
   auto model2 = RNTupleModel::Create();
   auto customStructfield = model2->MakeField<CustomStruct>("CustomStruct");
   auto customStructVec = model2->MakeField<std::vector<CustomStruct>>("CustomStructVec");
   auto customStructArray = model2->MakeField<std::array<CustomStruct, 2>>("CustomStructArray");
   auto ntuple2 = RNTupleReader::Open(std::move(model2), ntupleName, rootFileName);

   std::ostringstream os;
   ntuple2->Show(0, ROOT::Experimental::ENTupleFormat::kJSON, os);
   std::string fString{ std::string("")
      + "{\n"
      + "  \"CustomStruct\": {\n"
      + "    \"a\": 4.1,\n"
      + "    \"v1\": [0.1, 0.2, 0.3],\n"
      + "    \"v2\": [[1.1, 1.2, 1.3], [2.1, 2.2, 2.3]],\n"
      + "    \"s\": \"Example1String\"\n"
      + "  },\n"
      + "  \"CustomStructVec\": [{\"a\": 4.2, \"v1\": [0.1, 0.2, 0.3], \"v2\": [[1.1, 1.3], [2.1, 2.2, 2.3]], "
      +      "\"s\": \"Example2String\"}, {\"a\": 4.3, \"v1\": [0.1, 0.2, 0.3], "
      +      "\"v2\": [[1.1, 1.2, 1.3], [2.1, 2.3]], \"s\": \"Example3String\"}, "
      +      "{\"a\": 4.4, \"v1\": [0.1, 0.3], \"v2\": [[1.1, 1.2, 1.3], [2.1, 2.2, 2.3]], "
      +      "\"s\": \"Example4String\"}],\n"
      + "  \"CustomStructArray\": [{\"a\": 4.5, \"v1\": [0.1, 0.2, 0.3], \"v2\": [[1.1, 1.3], [2.1, 2.2, 2.3]], "
      +      "\"s\": \"AnotherString1\"}, {\"a\": 4.6, \"v1\": [0.1, 0.2, 0.3], "
      +      "\"v2\": [[1.1, 1.2, 1.3], [2.1, 2.3]], \"s\": \"AnotherString2\"}]\n"
      + "}\n" };
   EXPECT_EQ(fString, os.str());
}
