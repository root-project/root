// This hides the TBrowser.h file for this unit test and allows to define an initializable custom TBrowser class.
#define ROOT_TBrowser

#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include <TDirectory.h>
#include <TFile.h>

#include <ROOT/RBrowseVisitor.hxx>
#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleBrowser.hxx>
#include <ROOT/RNTupleModel.hxx>

#include "gtest/gtest.h"


using RNTupleReader = ROOT::Experimental::RNTupleReader;
using RNTupleBrowser = ROOT::Experimental::RNTupleBrowser;
using RNTupleWriter = ROOT::Experimental::RNTupleWriter;
using RNTupleModel = ROOT::Experimental::RNTupleModel;
using RNonFolder = ROOT::Experimental::RNTupleFieldElement;
using RFolder = ROOT::Experimental::RNTupleFieldElementFolder;

/* This test cannot test the entire TBrwoser-support, it has no access to the TBrowser GUI.
 * It only test certain functions, things it cannot test are:
 * - The way the ntuple is displayed in TBrowser.
 * - If histogram is displayed correctly.
 * - It can count how often the TBrowser::Add() was called, but not what happens when it's called.
 *
 * To check if tests were succesful, it often uses the member variable RNTupleBrowser::fUnitTest
 * defined in RNTupleBrowser.hxx.
 */

namespace {
  /**
   * An RAII wrapper around an open temporary file on disk. It cleans up the guarded file when the wrapper object
   * goes out of scope.
   */
class FileRaii {
private:
   std::string fPath;
   TFile *file;
public:
   FileRaii(const std::string &path) :
      fPath(path),
      file(TFile::Open(fPath.c_str(), "RECREATE"))
      { }
   FileRaii(const FileRaii&) = delete;
   FileRaii& operator=(const FileRaii&) = delete;
   TFile* GetFile() const { return file; }
   ~FileRaii() {
      file->Close();
      std::remove(fPath.c_str());
   }
};
} // anonymous namespace

// Allows to create TBrowser object, which can be used by functions requiring it. However calling it's members is not possible, because members of TBrowser.h are called instead if called from a file other than this file. The first 3 digits tell how often the Browse::Add function was called (or in this case rather "supposed" to be called). The fourth digit tells how many test.root files should be created in the beginning.
class TBrowser{
};


TEST(RNTupleBrowse, Floattest)
{
   FileRaii fileGuard("test.root");
   TDirectory *directory = fileGuard.GetFile()->mkdir("Staff");
   {
      auto model = RNTupleModel::Create();
      auto fieldPt = model->MakeField<float>("FloatField");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "Staff", "test.root");
      *fieldPt = 5.0f;
      ntuple->Fill();
   } // flushes content to test.root
   
   RNTupleBrowser browseObject(directory, 1000);
   browseObject.SetDirectory(directory);
   TBrowser browser;
   browseObject.Browse(&browser);
   
   ASSERT_EQ(browseObject.fNTupleBrowsePtrVec.size(), static_cast<std::size_t>(1));
   ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(0))->GetType(), ROOT::Experimental::fieldDatatype_float);
   ASSERT_EQ(browseObject.GetfUnitTest(), 1001);
}



TEST(RNTupleBrowse, Stringtest)
{
   
   FileRaii fileGuard("test.root");
   TDirectory *directory = fileGuard.GetFile()->mkdir("Staff2");
   {
      auto model = RNTupleModel::Create();
      auto fieldPt = model->MakeField<std::string>("StringField");
      auto fieldPt2 = model->MakeField<std::string>("StringField2");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "Staff2", "test.root");
      *fieldPt = "TestString";
      ntuple->Fill();
   } // flushes content to test.root
   
   RNTupleBrowser browseObject(directory, 1000);
   browseObject.SetDirectory(directory);
   TBrowser browser;
   browseObject.Browse(&browser);
   
   ASSERT_EQ(browseObject.fNTupleBrowsePtrVec.size(), static_cast<std::size_t>(2));
   ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(0))->GetType(), ROOT::Experimental::fieldDatatype_noHist);
   ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(1))->GetType(), ROOT::Experimental::fieldDatatype_noHist);
   ASSERT_EQ(browseObject.GetfUnitTest(), 1002);
}



TEST(RNTupleBrowse, MixedFieldstest)
{
   FileRaii fileGuard("test.root");
   TDirectory *directory = fileGuard.GetFile()->mkdir("Staff3");
   {
      auto model = RNTupleModel::Create();
      auto fieldPt = model->MakeField<double>("DoubleField");
      auto fieldPt2 = model->MakeField<std::string>("StringField");
      auto fieldPt3 = model->MakeField<int>("IntField");
      auto fieldPt4 = model->MakeField<std::string>("StringField2");
      auto fieldPt5 = model->MakeField<float>("floatField");
      auto fieldPt6 = model->MakeField<std::uint32_t>("uInt32Field");
      auto fieldPt7 = model->MakeField<std::uint64_t>("uInt64Field");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "Staff3", "test.root");
      *fieldPt = 7.9;
      ntuple->Fill();
   } // flushes content to test.root
   
   RNTupleBrowser browseObject(directory, 1000);
   browseObject.SetDirectory(directory);
   TBrowser browser;
   browseObject.Browse(&browser);
   
   ASSERT_EQ(browseObject.fNTupleBrowsePtrVec.size(), static_cast<std::size_t>(7));
   ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(0))->GetType(), ROOT::Experimental::fieldDatatype_double);
   ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(1))->GetType(), ROOT::Experimental::fieldDatatype_noHist);
   ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(2))->GetType(), ROOT::Experimental::fieldDatatype_Int32);
   ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(3))->GetType(), ROOT::Experimental::fieldDatatype_noHist);
   ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(4))->GetType(), ROOT::Experimental::fieldDatatype_float);
   ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(5))->GetType(), ROOT::Experimental::fieldDatatype_UInt32);
   ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(6))->GetType(), ROOT::Experimental::fieldDatatype_UInt64);
   ASSERT_EQ(browseObject.GetfUnitTest(), 1007);
}



TEST(RNTupleBrowse, Browsetest)
{
   FileRaii fileGuard("test.root");
   TDirectory *directory = fileGuard.GetFile()->mkdir("Staff4");
   {
      auto model = RNTupleModel::Create();
      auto fieldPt = model->MakeField<double>("DoubleField");
      auto fieldPt2 = model->MakeField<std::string>("StringField");
      auto fieldPt3 = model->MakeField<int>("IntField");
      auto fieldPt4 = model->MakeField<std::string>("StringField2");
      auto fieldPt5 = model->MakeField<float>("floatField");
      auto fieldPt6 = model->MakeField<std::uint32_t>("uInt32Field");
      auto fieldPt7 = model->MakeField<std::uint64_t>("uInt64Field");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "Staff4", "test.root");
      *fieldPt = 7.9;
      ntuple->Fill();
   } // flushes content to test.root
   
   RNTupleBrowser browseObject(directory, 1000);
   browseObject.SetDirectory(directory);
   TBrowser browser;
   browseObject.Browse(&browser);
   
   ASSERT_EQ(browseObject.fNTupleBrowsePtrVec.size(), static_cast<std::size_t>(7));
   ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(0))->GetType(), ROOT::Experimental::fieldDatatype_double);
   ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(1))->GetType(), ROOT::Experimental::fieldDatatype_noHist);
   ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(2))->GetType(), ROOT::Experimental::fieldDatatype_Int32);
   ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(3))->GetType(), ROOT::Experimental::fieldDatatype_noHist);
   ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(4))->GetType(), ROOT::Experimental::fieldDatatype_float);
   ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(5))->GetType(), ROOT::Experimental::fieldDatatype_UInt32);
   ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(6))->GetType(), ROOT::Experimental::fieldDatatype_UInt64);
   ASSERT_EQ(browseObject.GetfUnitTest(), 1007);
   // situation up to here is exaclty the same as in the unit test above.
   
   // no ASSERT_EQ here, it only checks if calling the Browse-method leads to an error.
   for (int i = 0; i < 7; ++i)
      static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i))->Browse(&browser);
}



TEST(RNTupleBrowse, MultipleBrowsetest)
{
   FileRaii fileGuard("test.root");
   TDirectory *directory = fileGuard.GetFile()->mkdir("Staff5");
   {
      auto model = RNTupleModel::Create();
      auto fieldPt = model->MakeField<double>("DoubleField");
      auto fieldPt2 = model->MakeField<std::string>("StringField");
      auto fieldPt3 = model->MakeField<int>("IntField");
      auto fieldPt4 = model->MakeField<std::string>("StringField2");
      auto fieldPt5 = model->MakeField<float>("floatField");
      auto fieldPt6 = model->MakeField<std::uint32_t>("uInt32Field");
      auto fieldPt7 = model->MakeField<std::uint64_t>("uInt64Field");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "Staff5", "test.root");
      *fieldPt = 7.9;
      ntuple->Fill();
   } // flushes content to test.root
   
   RNTupleBrowser browseObject(directory, 1000);
   browseObject.SetDirectory(directory);
   TBrowser browser;
   browseObject.Browse(&browser);
   browseObject.Browse(&browser);
   
   ASSERT_EQ(browseObject.fNTupleBrowsePtrVec.size(), static_cast<std::size_t>(14));
   for(int i = 0; i < 8; i+=7) {
   ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i))->GetType(), ROOT::Experimental::fieldDatatype_double);
   ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+1))->GetType(), ROOT::Experimental::fieldDatatype_noHist);
   ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+2))->GetType(), ROOT::Experimental::fieldDatatype_Int32);
   ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+3))->GetType(), ROOT::Experimental::fieldDatatype_noHist);
   ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+4))->GetType(), ROOT::Experimental::fieldDatatype_float);
   ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+5))->GetType(), ROOT::Experimental::fieldDatatype_UInt32);
   ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+6))->GetType(), ROOT::Experimental::fieldDatatype_UInt64);
   ASSERT_EQ(browseObject.GetfUnitTest(), 1014);
   }
   
   // no ASSERT_EQ here, it only checks if calling the Browse-method leads to an error.
   for (int i = 0; i < 14; ++i)
      static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i))->Browse(&browser);
   for (int i = 0; i < 14; ++i)
      static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i))->Browse(&browser);
}



TEST(RNTupleBrowse, VecBrowsetest)
{
   FileRaii fileGuard("test.root");
   TDirectory *directory = fileGuard.GetFile()->mkdir("Staff6");

   {
      auto model = RNTupleModel::Create();
      auto fieldPt = model->MakeField<double>("DoubleField");
      auto fieldPt2 = model->MakeField<std::string>("StringField");
      auto fieldPt3 = model->MakeField<std::vector<int>>("VecIntField");
      auto fieldPt4 = model->MakeField<std::string>("StringField2");
      auto fieldPt5 = model->MakeField<float>("floatField");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "Staff6", "test.root");
   } // flushes content to test.root
   
   RNTupleBrowser browseObject(directory, 1000);
   browseObject.SetDirectory(directory);
   TBrowser browser;
   browseObject.Browse(&browser);
   browseObject.Browse(&browser);
   
   ASSERT_EQ(browseObject.fNTupleBrowsePtrVec.size(), static_cast<std::size_t>(10)); // instead of 12, because only fields directly attached to the RootField should be "browsed"
   for(int i = 0; i < 6; i+=5) {
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i))->GetType(), ROOT::Experimental::fieldDatatype_double);
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+1))->GetType(), ROOT::Experimental::fieldDatatype_noHist);
      EXPECT_EQ(static_cast<RFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+2))->GetFieldName(), "VecIntField");
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+3))->GetType(), ROOT::Experimental::fieldDatatype_noHist);
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+4))->GetType(), ROOT::Experimental::fieldDatatype_float);
   }
   
   for (int j = 0; j < 3; ++j) {
      for (int i = 0; i < 10; ++i) {
         if (i%5 == 2) {
            static_cast<RFolder*>(browseObject.fNTupleBrowsePtrVec.at(i))->Browse(&browser);
         } else {
            static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i))->Browse(&browser);
         }
      }
   }
   
   ASSERT_EQ(browseObject.fNTupleBrowsePtrVec.size(), static_cast<std::size_t>(16)); // create 2*3=6 instances of int-BrowseFields stored.
   for (int i = 10; i < 16; ++i) {
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i))->GetType(), ROOT::Experimental::fieldDatatype_parentIsVec);
   }
}



TEST(RNTupleBrowse, DoubleVecBrowsetest)
{
   FileRaii fileGuard("test.root");
   TDirectory *directory = fileGuard.GetFile()->mkdir("Staff7");

   {
      auto model = RNTupleModel::Create();
      auto fieldPt = model->MakeField<double>("DoubleField");
      auto fieldPt2 = model->MakeField<std::string>("StringField");
      auto fieldPt3 = model->MakeField<std::vector<int>>("VecIntField");
      auto fieldPt4 = model->MakeField<std::string>("StringField2");
      auto fieldPt5 = model->MakeField<std::vector<int>>("VecIntField2");
      auto fieldPt6 = model->MakeField<float>("floatField");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "Staff7", "test.root");
   } // flushes content to test.root
   
   RNTupleBrowser browseObject(directory, 1000);
   browseObject.SetDirectory(directory);
   TBrowser browser;
   browseObject.Browse(&browser);
   browseObject.Browse(&browser);
   
   ASSERT_EQ(browseObject.fNTupleBrowsePtrVec.size(), static_cast<std::size_t>(12)); // instead of 16, because only fields directly attached to the RootField should be "browsed"
   for (int i = 0; i < 7; i+=6) {
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i))->GetType(), ROOT::Experimental::fieldDatatype_double);
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+1))->GetType(), ROOT::Experimental::fieldDatatype_noHist);
      EXPECT_EQ(static_cast<RFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+2))->GetFieldName(), "VecIntField");
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+3))->GetType(), ROOT::Experimental::fieldDatatype_noHist);
      EXPECT_EQ(static_cast<RFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+4))->GetFieldName(), "VecIntField2");
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+5))->GetType(), ROOT::Experimental::fieldDatatype_float);
   }
   
   for (int j = 0; j < 3; ++j) {
      for (int i = 0; i < 12; ++i) {
         if (i%6 == 2 || i%6 == 4) {
            static_cast<RFolder*>(browseObject.fNTupleBrowsePtrVec.at(i))->Browse(&browser);
         } else {
            static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i))->Browse(&browser);
         }
      }
   }
   
   ASSERT_EQ(browseObject.fNTupleBrowsePtrVec.size(), static_cast<std::size_t>(24)); // create 2*2*3=12 instances of int-BrowseFields stored.
   for(int i = 12; i < 24; ++i)
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i))->GetType(), ROOT::Experimental::fieldDatatype_parentIsVec);
   ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(0))->GetType(), ROOT::Experimental::fieldDatatype_double);
   ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(1))->GetType(), ROOT::Experimental::fieldDatatype_noHist);
   EXPECT_EQ(static_cast<RFolder*>(browseObject.fNTupleBrowsePtrVec.at(2))->GetFieldName(), "VecIntField");
   ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(3))->GetType(), ROOT::Experimental::fieldDatatype_noHist);
   EXPECT_EQ(static_cast<RFolder*>(browseObject.fNTupleBrowsePtrVec.at(4))->GetFieldName(), "VecIntField2");
   ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(5))->GetType(), ROOT::Experimental::fieldDatatype_float);
}



TEST(RNTupleBrowse, MultipleRootFiletest)
{
   FileRaii fileGuard("test.root");
   TDirectory *directory = fileGuard.GetFile()->mkdir("Staff8");
   {
      auto model = RNTupleModel::Create();
      auto fieldPt = model->MakeField<double>("DoubleField");
      auto fieldPt2 = model->MakeField<std::string>("StringField");
      auto fieldPt3 = model->MakeField<std::vector<int>>("VecIntField");
      auto fieldPt4 = model->MakeField<std::string>("StringField2");
      auto fieldPt5 = model->MakeField<std::vector<int>>("VecIntField2");
      auto fieldPt6 = model->MakeField<float>("floatField");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "Staff8", "test.root");
   } // flushes content to test.root
   
   FileRaii fileGuard2("test2.root");
   TDirectory *directory2 = fileGuard2.GetFile()->mkdir("Staff9");
   {
      auto model = RNTupleModel::Create();
      auto fieldPt = model->MakeField<double>("Double2Field");
      //auto fieldPt2 = model->MakeField<std::string>("String2Field");
      auto fieldPt3 = model->MakeField<std::vector<int>>("VecInt2Field");
      auto fieldPt4 = model->MakeField<std::string>("String2Field2");
      auto fieldPt5 = model->MakeField<std::vector<int>>("VecInt2Field2");
      auto fieldPt6 = model->MakeField<float>("float2Field");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "Staff9", "test2.root");
   } // flushes content to test.root
   
   RNTupleBrowser browseObject(directory, 1000);
   browseObject.SetDirectory(directory);
   TBrowser browser;
   browseObject.Browse(&browser);
   browseObject.Browse(&browser);
   
   RNTupleBrowser browseObject2(directory2, 2000);
   browseObject2.SetDirectory(directory2);
   TBrowser browser2;
   browseObject2.Browse(&browser2);
   browseObject2.Browse(&browser2);
   
   ASSERT_EQ(browseObject.fNTupleBrowsePtrVec.size(), static_cast<std::size_t>(12)); // instead of 16, because only fields directly attached to the RootField should be "browsed"
   for (int i = 0; i < 7; i+=6) {
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i))->GetType(), ROOT::Experimental::fieldDatatype_double);
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+1))->GetType(), ROOT::Experimental::fieldDatatype_noHist);
      EXPECT_EQ(static_cast<RFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+2))->GetFieldName(), "VecIntField");
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+3))->GetType(), ROOT::Experimental::fieldDatatype_noHist);
      EXPECT_EQ(static_cast<RFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+4))->GetFieldName(), "VecIntField2");
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+5))->GetType(), ROOT::Experimental::fieldDatatype_float);
   }
   
   ASSERT_EQ(browseObject2.fNTupleBrowsePtrVec.size(), static_cast<std::size_t>(10)); // instead of 12, because only fields directly attached to the RootField should be "browsed"
   for (int i = 0; i < 7; i+=5) {
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject2.fNTupleBrowsePtrVec.at(i))->GetType(), ROOT::Experimental::fieldDatatype_double);
      EXPECT_EQ(static_cast<RFolder*>(browseObject2.fNTupleBrowsePtrVec.at(i+1))->GetFieldName(), "VecInt2Field");
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject2.fNTupleBrowsePtrVec.at(i+2))->GetType(), ROOT::Experimental::fieldDatatype_noHist);
      EXPECT_EQ(static_cast<RFolder*>(browseObject2.fNTupleBrowsePtrVec.at(i+3))->GetFieldName(), "VecInt2Field2");
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject2.fNTupleBrowsePtrVec.at(i+4))->GetType(), ROOT::Experimental::fieldDatatype_float);
   }
}



TEST(RNTupleBrowse, MultipleRootFileMultipleBrowsetest)
{
   FileRaii fileGuard("test.root");
   TDirectory *directory = fileGuard.GetFile()->mkdir("Staff10");
   {
      auto model = RNTupleModel::Create();
      auto fieldPt = model->MakeField<double>("DoubleField");
      auto fieldPt2 = model->MakeField<std::string>("StringField");
      auto fieldPt3 = model->MakeField<std::vector<int>>("VecIntField");
      auto fieldPt4 = model->MakeField<std::string>("StringField2");
      auto fieldPt5 = model->MakeField<std::vector<int>>("VecIntField2");
      auto fieldPt6 = model->MakeField<float>("floatField");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "Staff10", "test.root");
   } // flushes content to test.root
   
   FileRaii fileGuard2("test2.root");
   TDirectory *directory2 = fileGuard2.GetFile()->mkdir("Staff11");
   {
      auto model = RNTupleModel::Create();
      auto fieldPt = model->MakeField<double>("Double2Field");
      auto fieldPt2 = model->MakeField<std::string>("String2Field");
      auto fieldPt3 = model->MakeField<std::vector<int>>("VecInt2Field");
      auto fieldPt4 = model->MakeField<std::string>("String2Field2");
      // difference to the ntuple above: auto fieldPt5 = model->MakeField<std::vector<int>>("VecInt2Field2");
      auto fieldPt6 = model->MakeField<float>("float2Field");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "Staff11", "test2.root");
   } // flushes content to test.root
   
   RNTupleBrowser browseObject(directory, 1000);
   browseObject.SetDirectory(directory);
   TBrowser browser;
   browseObject.Browse(&browser);
   browseObject.Browse(&browser);
   
   RNTupleBrowser browseObject2(directory2, 2000);
   browseObject2.SetDirectory(directory2);
   TBrowser browser2;
   browseObject2.Browse(&browser2);
   browseObject2.Browse(&browser2);
   
   ASSERT_EQ(browseObject.fNTupleBrowsePtrVec.size(), static_cast<std::size_t>(12)); // instead of 16, because only fields directly attached to the RootField should be "browsed"
   for (int i = 0; i < 7; i+=6) {
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i))->GetType(), ROOT::Experimental::fieldDatatype_double);
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+1))->GetType(), ROOT::Experimental::fieldDatatype_noHist);
      EXPECT_EQ(static_cast<RFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+2))->GetFieldName(), "VecIntField");
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+3))->GetType(), ROOT::Experimental::fieldDatatype_noHist);
      EXPECT_EQ(static_cast<RFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+4))->GetFieldName(), "VecIntField2");
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+5))->GetType(), ROOT::Experimental::fieldDatatype_float);
   }
   
   ASSERT_EQ(browseObject2.fNTupleBrowsePtrVec.size(), static_cast<std::size_t>(10)); // instead of 12, because only fields directly attached to the RootField should be "browsed"
   for (int i = 0; i < 7; i+=5) {
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject2.fNTupleBrowsePtrVec.at(i))->GetType(), ROOT::Experimental::fieldDatatype_double);
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject2.fNTupleBrowsePtrVec.at(i+1))->GetType(), ROOT::Experimental::fieldDatatype_noHist);
      EXPECT_EQ(static_cast<RFolder*>(browseObject2.fNTupleBrowsePtrVec.at(i+2))->GetFieldName(), "VecInt2Field");
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject2.fNTupleBrowsePtrVec.at(i+3))->GetType(), ROOT::Experimental::fieldDatatype_noHist);
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject2.fNTupleBrowsePtrVec.at(i+4))->GetType(), ROOT::Experimental::fieldDatatype_float);
   }
   
   // browses a couple of times the different root files.
   for (int i = 0; i < 12; ++i) {
      if ( i%6 == 2 || i%6 == 4 ) {
         static_cast<RFolder*>(browseObject.fNTupleBrowsePtrVec.at(i))->Browse(&browser);
      } else {
         static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i))->Browse(&browser);
      }
   }
   for (int i = 0; i < 10; ++i) {
      if ( i%5 == 2 ) {
         static_cast<RFolder*>(browseObject2.fNTupleBrowsePtrVec.at(i))->Browse(&browser2);
      } else {
         static_cast<RNonFolder*>(browseObject2.fNTupleBrowsePtrVec.at(i))->Browse(&browser2);
      }
   }
   for (int i = 0; i < 12; ++i) {
      if ( i%6 == 2 || i%6 == 4 ) {
         static_cast<RFolder*>(browseObject.fNTupleBrowsePtrVec.at(i))->Browse(&browser);
      } else {
         static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i))->Browse(&browser);
      }
   }
   for (int i = 0; i < 10; ++i) {
      if ( i%5 == 2 ) {
         static_cast<RFolder*>(browseObject2.fNTupleBrowsePtrVec.at(i))->Browse(&browser2);
      } else {
         static_cast<RNonFolder*>(browseObject2.fNTupleBrowsePtrVec.at(i))->Browse(&browser2);
      }
   }
   
   
   ASSERT_EQ(browseObject.fNTupleBrowsePtrVec.size(), static_cast<std::size_t>(20));
   for (int i = 0; i < 7; i+=6) {
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i))->GetType(), ROOT::Experimental::fieldDatatype_double);
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+1))->GetType(), ROOT::Experimental::fieldDatatype_noHist);
      EXPECT_EQ(static_cast<RFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+2))->GetFieldName(), "VecIntField");
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+3))->GetType(), ROOT::Experimental::fieldDatatype_noHist);
      EXPECT_EQ(static_cast<RFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+4))->GetFieldName(), "VecIntField2");
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+5))->GetType(), ROOT::Experimental::fieldDatatype_float);
   }
   
   ASSERT_EQ(browseObject2.fNTupleBrowsePtrVec.size(), static_cast<std::size_t>(14));
   for (int i = 0; i < 7; i+=5) {
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject2.fNTupleBrowsePtrVec.at(i))->GetType(), ROOT::Experimental::fieldDatatype_double);
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject2.fNTupleBrowsePtrVec.at(i+1))->GetType(), ROOT::Experimental::fieldDatatype_noHist);
      EXPECT_EQ(static_cast<RFolder*>(browseObject2.fNTupleBrowsePtrVec.at(i+2))->GetFieldName(), "VecInt2Field");
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject2.fNTupleBrowsePtrVec.at(i+3))->GetType(), ROOT::Experimental::fieldDatatype_noHist);
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject2.fNTupleBrowsePtrVec.at(i+4))->GetType(), ROOT::Experimental::fieldDatatype_float);
   }
   
   for (int i = 12; i < 20; ++i) {
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i))->GetType(), ROOT::Experimental::fieldDatatype_parentIsVec);
   }
   for (int i = 10; i < 14; ++i) {
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject2.fNTupleBrowsePtrVec.at(i))->GetType(), ROOT::Experimental::fieldDatatype_parentIsVec);
   }
}

