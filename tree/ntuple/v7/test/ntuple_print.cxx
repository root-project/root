#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RFieldVisitor.hxx>

#include <TClass.h>
#include <TFile.h>

#include "gtest/gtest.h"

#include <iostream>
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
      TFile *file = TFile::Open("test.root", "RECREATE");
   public:
      FileRaii(const std::string &path) : fPath(path)
      {
         
      }
      FileRaii(const FileRaii&) = delete;
      FileRaii& operator=(const FileRaii&) = delete;
      ~FileRaii() {
         file->Close();
         std::remove(fPath.c_str());
      }
   };
    
} // anonymous namespace

TEST(RNtuplePrint, FullString)
{
   FileRaii fileGuard("test.root");
   {
   auto model = RNTupleModel::Create();
   auto fieldPt = model->MakeField<float>("pt");
   auto ntuple = RNTupleWriter::Recreate(std::move(model), "Staff", "test.root");
   *fieldPt = 5.0f;
   ntuple->Fill();
   }
   auto model2 = RNTupleModel::Create();
   auto fieldPt2 = model2->MakeField<float>("pt");
   auto ntuple2 = RNTupleReader::Open(std::move(model2), "Staff", "test.root");
   std::ostringstream os;
   ntuple2->Print(os);
   std::string fString{"************************************ NTUPLE ************************************\n* Ntuple  : Staff                                                              *\n* Entries : 1                                                                  *\n********************************************************************************\n* Field 1   : pt (float)                                                       *\n********************************************************************************\n"};
   EXPECT_EQ(fString, os.str());
}

TEST(RNtuplePrint, IntTest)
{
   std::stringstream os;
   RPrintVisitor visitor(os);
   visitor.SetAvailableSpaceForStrings();
   RField<int> testField("intTest");
   testField.AcceptVisitor(visitor, 1);
   std::string expected{"********************************************************************************\n* Field -1  : intTest (std::int32_t)                                           *\n"};
   EXPECT_EQ(expected, os.str());
}

TEST(RNtuplePrint, FloatTest)
{
   std::stringstream os;
   RPrintVisitor visitor(os);
   visitor.SetAvailableSpaceForStrings();
   RField<float> testField("floatTest");
   testField.AcceptVisitor(visitor, 1);
   std::string expected{"********************************************************************************\n* Field -1  : floatTest (float)                                                *\n"};
   EXPECT_EQ(expected, os.str());
}

TEST(RNtuplePrint, FloatTestTraverse)
{
   std::stringstream os;
   RPrintVisitor visitor(os, 'a');
   visitor.SetAvailableSpaceForStrings();
   RField<float> testField("floatTest");
   testField.TraverseVisitor(visitor, 1);
   std::string expected{"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\na Field -1  : floatTest (float)                                                a\n"};
   EXPECT_EQ(expected, os.str());
}

TEST(RNtuplePrint, VecTestAccept)
{
   std::stringstream os;
   RPrintVisitor visitor(os, 'a');
   visitor.SetAvailableSpaceForStrings();
   RField<std::vector<float>> testField("floatTest");
   testField.AcceptVisitor(visitor, 1);
   std::string expected{"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\na Field -1  : floatTest (std::vector<float>)                                   a\n"};
   EXPECT_EQ(expected, os.str());
}

TEST(RNtuplePrint, VecTestTraverse)
{
   std::stringstream os;
   RPrepareVisitor prepVisitor;
   RField<std::vector<float>> testField("floatVecTest");
   testField.TraverseVisitor(prepVisitor, 1);
   RPrintVisitor visitor(os, '$');
   visitor.SetDeepestLevel(prepVisitor.getDeepestLevel());
   visitor.SetNumFields(prepVisitor.getNumFields());
   visitor.SetAvailableSpaceForStrings();
   visitor.ResizeFlagVec();
   testField.TraverseVisitor(visitor, 1);
   std::string expected{"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n$ Field -1      : floatVecTest (std::vector<float>)                            $\n$ |__Field 1    : floatVecTest/floatVecTest (float)                            $\n"};
   EXPECT_EQ(expected, os.str());
}

TEST(RNtuplePrint, VecVecTestTraverse)
{
   std::stringstream os;
   RPrepareVisitor prepVisitor;
   RField<std::vector<std::vector<float>>> testField("floatVecVecTest");
   testField.TraverseVisitor(prepVisitor, 1);
   RPrintVisitor visitor(os, 'x');
   visitor.SetDeepestLevel(prepVisitor.getDeepestLevel());
   visitor.SetNumFields(prepVisitor.getNumFields());
   visitor.SetAvailableSpaceForStrings();
   visitor.ResizeFlagVec();
   testField.TraverseVisitor(visitor, 1);
   std::string expected{"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\nx Field -1          : floatVecVecTest (std::vector<std::vector<float>>)        x\nx |__Field 1        : floatVecVecTest/floatVecVecTest (std::vector<float>)     x\nx   |__Field 1.1    : floatVecVecTest/floatVecVecTest/floatVecVecTest (float)  x\n"};
   EXPECT_EQ(expected, os.str());
}

TEST(RNtuplePrint, NarrowManyEntriesVecVecTestTraverse)
{
   std::stringstream os;
   RPrepareVisitor prepVisitor;
   RField<std::vector<std::vector<float>>> testField("floatVecVecTest");
   testField.TraverseVisitor(prepVisitor, 1);
   RPrintVisitor visitor(os, ' ', 30);
   visitor.SetDeepestLevel(prepVisitor.getDeepestLevel());
   visitor.SetNumFields(prepVisitor.getNumFields());
   visitor.SetAvailableSpaceForStrings();
   visitor.ResizeFlagVec();
   testField.TraverseVisitor(visitor, 1);
   std::string expected{"                              \n  Field -1        : floatV... \n  |__Field 1      : floatV... \n    |__Field 1.1  : floatV... \n"};
   EXPECT_EQ(expected, os.str());
}
