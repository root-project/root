#include <ROOT/RFieldVisitor.hxx>
#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <TFile.h>

#include "gtest/gtest.h"

#include <iostream>
#include <random>
#include <sstream>
#include <vector>

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

/* Currently the width can't be set by PrintInfo(). Will be enabled when this feature is added.
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
