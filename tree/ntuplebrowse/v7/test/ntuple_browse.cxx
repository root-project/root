#include <ROOT/RBrowseVisitor.hxx>
#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleBrowser.hxx>
#include <ROOT/RNTupleModel.hxx>

#include <TDirectory.h>
#include <TFile.h>

#include "gtest/gtest.h"

#include <memory>
#include <sstream>
#include <typeinfo>
#include <vector>

using RFolder = ROOT::Experimental::RNTupleBrowseFolder;
using RNonFolder = ROOT::Experimental::RNTupleBrowseLeaf;
using RNTupleBrowser = ROOT::Experimental::RNTupleBrowser;
using RNTupleModel = ROOT::Experimental::RNTupleModel;
using RNTupleReader = ROOT::Experimental::RNTupleReader;
using RNTupleWriter = ROOT::Experimental::RNTupleWriter;

/* This test cannot test the entire TBrwoser-support, it has no access to the TBrowser GUI.
 * It only test certain functions, things it cannot test are:
 * - The way the ntuple is displayed in TBrowser.
 * - If histogram is displayed correctly.
 * - It doesn't know what happens after TBrowser::Add() is called, and also doesn't know how often it was called.
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

// helper function for tests
std::string GetFileName(const TDirectory *directory) {
   std::string fullPath(directory->GetPath());
   return std::string(fullPath, 0, fullPath.find(".root") + 5);
}

TEST(RNTupleBrowse, Floattest)
{
   FileRaii fileGuard("test.root");
   TDirectory *directory = fileGuard.GetFile()->mkdir("Staff");
   {
      auto model = RNTupleModel::Create();
      auto fieldPt = model->MakeField<float>("FloatField");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), directory->GetName(), GetFileName(directory));
      *fieldPt = 5.0f;
      ntuple->Fill();
   } // flushes content to test.root
   
   RNTupleBrowser browseObject(directory);
   browseObject.SetDirectory(directory);
   browseObject.Browse(nullptr);
   
   ASSERT_EQ(browseObject.fNTupleBrowsePtrVec.size(), static_cast<std::size_t>(1));
   EXPECT_STREQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(0).get())->GetName(), "FloatField");
}



TEST(RNTupleBrowse, Stringtest)
{
   
   FileRaii fileGuard("test2.root");
   TDirectory *directory = fileGuard.GetFile()->mkdir("Staff2");
   {
      auto model = RNTupleModel::Create();
      auto fieldPt = model->MakeField<std::string>("StringField");
      auto fieldPt2 = model->MakeField<std::string>("StringField2");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), directory->GetName(), GetFileName(directory));
      *fieldPt = "TestString";
      ntuple->Fill();
   } // flushes content to test.root
   
   RNTupleBrowser browseObject(directory);
   browseObject.SetDirectory(directory);
   browseObject.Browse(nullptr);
   
   ASSERT_EQ(browseObject.fNTupleBrowsePtrVec.size(), static_cast<std::size_t>(2));
   EXPECT_STREQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(0).get())->GetName(), "StringField");
   EXPECT_STREQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(1).get())->GetName(), "StringField2");
}



TEST(RNTupleBrowse, MixedFieldstest)
{
   FileRaii fileGuard("test3.root");
   TDirectory *directory = fileGuard.GetFile()->mkdir("Staff3");
   {
      auto model = RNTupleModel::Create();
      auto fieldPt = model->MakeField<double>("DoubleField");
      auto fieldPt2 = model->MakeField<std::string>("StringField");
      auto fieldPt3 = model->MakeField<std::int32_t>("IntField");
      auto fieldPt4 = model->MakeField<std::string>("StringField2");
      auto fieldPt5 = model->MakeField<float>("floatField");
      auto fieldPt6 = model->MakeField<std::uint32_t>("uInt32Field");
      auto fieldPt7 = model->MakeField<std::uint64_t>("uInt64Field");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), directory->GetName(), GetFileName(directory));
      *fieldPt = 7.9;
      ntuple->Fill();
   } // flushes content to test.root
   
   RNTupleBrowser browseObject(directory);
   browseObject.SetDirectory(directory);
   browseObject.Browse(nullptr);
   
   ASSERT_EQ(browseObject.fNTupleBrowsePtrVec.size(), static_cast<std::size_t>(7));
   EXPECT_STREQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(0).get())->GetName(), "DoubleField");
   EXPECT_STREQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(1).get())->GetName(), "StringField");
   EXPECT_STREQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(2).get())->GetName(), "IntField");
   EXPECT_STREQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(3).get())->GetName(), "StringField2");
   EXPECT_STREQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(4).get())->GetName(), "floatField");
   EXPECT_STREQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(5).get())->GetName(), "uInt32Field");
   EXPECT_STREQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(6).get())->GetName(), "uInt64Field");
}



TEST(RNTupleBrowse, Browsetest)
{
   FileRaii fileGuard("test4.root");
   TDirectory *directory = fileGuard.GetFile()->mkdir("Staff4");
   {
      auto model = RNTupleModel::Create();
      auto fieldPt = model->MakeField<double>("DoubleField");
      auto fieldPt2 = model->MakeField<std::string>("StringField");
      auto fieldPt3 = model->MakeField<std::int32_t>("IntField");
      auto fieldPt4 = model->MakeField<std::string>("StringField2");
      auto fieldPt5 = model->MakeField<float>("floatField");
      auto fieldPt6 = model->MakeField<std::uint32_t>("uInt32Field");
      auto fieldPt7 = model->MakeField<std::uint64_t>("uInt64Field");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), directory->GetName(), GetFileName(directory));
      *fieldPt = 7.9;
      ntuple->Fill();
   } // flushes content to test.root
   
   RNTupleBrowser browseObject(directory);
   browseObject.SetDirectory(directory);
   browseObject.Browse(nullptr);
   
   ASSERT_EQ(browseObject.fNTupleBrowsePtrVec.size(), static_cast<std::size_t>(7));
   EXPECT_STREQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(0).get())->GetName(), "DoubleField");
   EXPECT_STREQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(1).get())->GetName(), "StringField");
   EXPECT_STREQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(2).get())->GetName(), "IntField");
   EXPECT_STREQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(3).get())->GetName(), "StringField2");
   EXPECT_STREQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(4).get())->GetName(), "floatField");
   EXPECT_STREQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(5).get())->GetName(), "uInt32Field");
   EXPECT_STREQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(6).get())->GetName(), "uInt64Field");
   // situation up to here is exaclty the same as in the unit test above.
   
   // no ASSERT_EQ here, it only checks if calling the Browse-method (drawing Histo) leads to an error.
   for (int i = 0; i < 7; ++i)
      static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i).get())->Browse(nullptr);
}



TEST(RNTupleBrowse, MultipleBrowsetest)
{
   FileRaii fileGuard("test5.root");
   TDirectory *directory = fileGuard.GetFile()->mkdir("Staff5");
   {
      auto model = RNTupleModel::Create();
      auto fieldPt = model->MakeField<double>("DoubleField");
      auto fieldPt2 = model->MakeField<std::string>("StringField");
      auto fieldPt3 = model->MakeField<std::int32_t>("IntField");
      auto fieldPt4 = model->MakeField<std::string>("StringField2");
      auto fieldPt5 = model->MakeField<float>("floatField");
      auto fieldPt6 = model->MakeField<std::uint32_t>("uInt32Field");
      auto fieldPt7 = model->MakeField<std::uint64_t>("uInt64Field");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), directory->GetName(), GetFileName(directory));
      *fieldPt = 7.9;
      ntuple->Fill();
   } // flushes content to test.root
   
   RNTupleBrowser browseObject(directory);
   browseObject.SetDirectory(directory);
   browseObject.Browse(nullptr);
   browseObject.Browse(nullptr);
   
   ASSERT_EQ(browseObject.fNTupleBrowsePtrVec.size(), static_cast<std::size_t>(14));
   for(int i = 0; i < 8; i+=7) {
      EXPECT_STREQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(0).get())->GetName(), "DoubleField");
      EXPECT_STREQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(1).get())->GetName(), "StringField");
      EXPECT_STREQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(2).get())->GetName(), "IntField");
      EXPECT_STREQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(3).get())->GetName(), "StringField2");
      EXPECT_STREQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(4).get())->GetName(), "floatField");
      EXPECT_STREQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(5).get())->GetName(), "uInt32Field");
      EXPECT_STREQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(6).get())->GetName(), "uInt64Field");
   }
   
   // Checks if calling the Browse-method (draw Histo) leads to an error.
   for (int i = 0; i < 14; ++i)
      static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i).get())->Browse(nullptr);
   for (int i = 0; i < 14; ++i)
      static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i).get())->Browse(nullptr);
}


TEST(RNTupleBrowse, VecBrowsetest)
{
   FileRaii fileGuard("test6.root");
   TDirectory *directory = fileGuard.GetFile()->mkdir("Staff6");

   {
      auto model = RNTupleModel::Create();
      auto fieldPt = model->MakeField<double>("DoubleField");
      auto fieldPt2 = model->MakeField<std::string>("StringField");
      auto fieldPt3 = model->MakeField<std::vector<std::int32_t>>("VecIntField");
      auto fieldPt4 = model->MakeField<std::string>("StringField2");
      auto fieldPt5 = model->MakeField<float>("floatField");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), directory->GetName(), GetFileName(directory));
   } // flushes content to test.root
   
   RNTupleBrowser browseObject(directory);
   browseObject.SetDirectory(directory);
   browseObject.Browse(nullptr);
   browseObject.Browse(nullptr);
   
   ASSERT_EQ(browseObject.fNTupleBrowsePtrVec.size(), static_cast<std::size_t>(10)); // instead of 12, because only fields directly attached to the RootField should be "browsed"
   for(int i = 0; i < 6; i+=5) {
      EXPECT_STREQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i).get())->GetName(), "DoubleField");
      EXPECT_STREQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+1).get())->GetName(), "StringField");
      EXPECT_STREQ(static_cast<RFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+2).get())->GetName(), "VecIntField");
      EXPECT_STREQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+3).get())->GetName(), "StringField2");
      EXPECT_STREQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+4).get())->GetName(), "floatField");
   }
   
   for (int j = 0; j < 3; ++j) {
      for (int i = 0; i < 10; ++i) {
         if (i%5 == 2) {
            static_cast<RFolder*>(browseObject.fNTupleBrowsePtrVec.at(i).get())->Browse(nullptr);
         } else {
            static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i).get())->Browse(nullptr);
         }
      }
   }
   
   ASSERT_EQ(browseObject.fNTupleBrowsePtrVec.size(), static_cast<std::size_t>(16)); // create 2*3=6 instances of int-BrowseFields stored.
   for (int i = 10; i < 16; ++i) {
      EXPECT_STREQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i).get())->GetName(), "std::int32_t");
   }
}



TEST(RNTupleBrowse, DoubleVecBrowsetest)
{
   FileRaii fileGuard("test7.root");
   TDirectory *directory = fileGuard.GetFile()->mkdir("Staff7");

   {
      auto model = RNTupleModel::Create();
      auto fieldPt = model->MakeField<double>("DoubleField");
      auto fieldPt2 = model->MakeField<std::string>("StringField");
      auto fieldPt3 = model->MakeField<std::vector<std::int32_t>>("VecIntField");
      auto fieldPt4 = model->MakeField<std::string>("StringField2");
      auto fieldPt5 = model->MakeField<std::vector<double>>("VecDoubleField");
      auto fieldPt6 = model->MakeField<float>("floatField");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), directory->GetName(), GetFileName(directory));
   } // flushes content to test.root
   
   RNTupleBrowser browseObject(directory);
   browseObject.SetDirectory(directory);
   browseObject.Browse(nullptr);
   browseObject.Browse(nullptr);
   
   ASSERT_EQ(browseObject.fNTupleBrowsePtrVec.size(), static_cast<std::size_t>(12)); // instead of 16, because only fields directly attached to the RootField should be "browsed"
   for (int i = 0; i < 7; i+=6) {
      EXPECT_STREQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i).get())->GetName(), "DoubleField");
      EXPECT_STREQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+1).get())->GetName(), "StringField");
      EXPECT_STREQ(static_cast<RFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+2).get())->GetName(), "VecIntField");
      EXPECT_STREQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+3).get())->GetName(), "StringField2");
      EXPECT_STREQ(static_cast<RFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+4).get())->GetName(), "VecDoubleField");
      EXPECT_STREQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+5).get())->GetName(), "floatField");
   }
   
   for (int j = 0; j < 3; ++j) {
      for (int i = 0; i < 12; ++i) {
         if (i%6 == 2 || i%6 == 4) {
            static_cast<RFolder*>(browseObject.fNTupleBrowsePtrVec.at(i).get())->Browse(nullptr);
         } else {
            static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i).get())->Browse(nullptr);
         }
      }
   }
   
   ASSERT_EQ(browseObject.fNTupleBrowsePtrVec.size(), static_cast<std::size_t>(24)); // create 2*2*3=12 instances of int-BrowseFields stored.
   for(int i = 12; i < 24; i+=2) {
      EXPECT_STREQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i).get())->GetName(), "std::int32_t");
      EXPECT_STREQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+1).get())->GetName(), "double");
   }
   EXPECT_STREQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(0).get())->GetName(), "DoubleField");
   EXPECT_STREQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(1).get())->GetName(), "StringField");
   EXPECT_STREQ(static_cast<RFolder*>(browseObject.fNTupleBrowsePtrVec.at(2).get())->GetName(), "VecIntField");
   EXPECT_STREQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(3).get())->GetName(), "StringField2");
   EXPECT_STREQ(static_cast<RFolder*>(browseObject.fNTupleBrowsePtrVec.at(4).get())->GetName(), "VecDoubleField");
   EXPECT_STREQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(5).get())->GetName(), "floatField");
}



TEST(RNTupleBrowse, MultipleRootFiletest)
{
   FileRaii fileGuard("test8.root");
   TDirectory *directory = fileGuard.GetFile()->mkdir("Staff8");
   {
      auto model = RNTupleModel::Create();
      auto fieldPt = model->MakeField<double>("DoubleField");
      auto fieldPt2 = model->MakeField<std::string>("StringField");
      auto fieldPt3 = model->MakeField<std::vector<double>>("VecDoubleField");
      auto fieldPt4 = model->MakeField<std::string>("StringField2");
      auto fieldPt5 = model->MakeField<std::vector<std::int32_t>>("VecIntField");
      auto fieldPt6 = model->MakeField<float>("floatField");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), directory->GetName(), GetFileName(directory));
   } // flushes content to test.root
   
   FileRaii fileGuard2("test9.root");
   TDirectory *directory2 = fileGuard2.GetFile()->mkdir("Staff9");
   {
      auto model = RNTupleModel::Create();
      auto fieldPt = model->MakeField<double>("Double2Field");
      //auto fieldPt2 = model->MakeField<std::string>("String2Field");
      auto fieldPt3 = model->MakeField<std::vector<std::int32_t>>("VecInt2Field");
      auto fieldPt4 = model->MakeField<std::string>("String2Field2");
      auto fieldPt5 = model->MakeField<std::vector<double>>("VecDouble2Field");
      auto fieldPt6 = model->MakeField<float>("float2Field");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), directory2->GetName(), GetFileName(directory2));
   } // flushes content to test.root
   
   RNTupleBrowser browseObject(directory);
   browseObject.SetDirectory(directory);
   browseObject.Browse(nullptr);
   browseObject.Browse(nullptr);
   
   RNTupleBrowser browseObject2(directory2);
   browseObject2.SetDirectory(directory2);
   browseObject2.Browse(nullptr);
   browseObject2.Browse(nullptr);
   
   ASSERT_EQ(browseObject.fNTupleBrowsePtrVec.size(), static_cast<std::size_t>(12)); // instead of 16, because only fields directly attached to the RootField should be "browsed"
   for (int i = 0; i < 7; i+=6) {
      EXPECT_STREQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i).get())->GetName(), "DoubleField");
      EXPECT_STREQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+1).get())->GetName(), "StringField");
      EXPECT_STREQ(static_cast<RFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+2).get())->GetName(), "VecDoubleField");
      EXPECT_STREQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+3).get())->GetName(), "StringField2");
      EXPECT_STREQ(static_cast<RFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+4).get())->GetName(), "VecIntField");
      EXPECT_STREQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+5).get())->GetName(), "floatField");
   }
   
   ASSERT_EQ(browseObject2.fNTupleBrowsePtrVec.size(), static_cast<std::size_t>(10)); // instead of 12, because only fields directly attached to the RootField should be "browsed"
   for (int i = 0; i < 7; i+=5) {
      EXPECT_STREQ(static_cast<RNonFolder*>(browseObject2.fNTupleBrowsePtrVec.at(i).get())->GetName(), "Double2Field");
      EXPECT_STREQ(static_cast<RFolder*>(browseObject2.fNTupleBrowsePtrVec.at(i+1).get())->GetName(), "VecInt2Field");
      EXPECT_STREQ(static_cast<RNonFolder*>(browseObject2.fNTupleBrowsePtrVec.at(i+2).get())->GetName(), "String2Field2");
      EXPECT_STREQ(static_cast<RFolder*>(browseObject2.fNTupleBrowsePtrVec.at(i+3).get())->GetName(), "VecDouble2Field");
      EXPECT_STREQ(static_cast<RNonFolder*>(browseObject2.fNTupleBrowsePtrVec.at(i+4).get())->GetName(), "float2Field");
   }
}



TEST(RNTupleBrowse, MultipleRootFileMultipleBrowsetest)
{
   FileRaii fileGuard("test10.root");
   TDirectory *directory = fileGuard.GetFile()->mkdir("Staff10");
   {
      auto model = RNTupleModel::Create();
      auto fieldPt = model->MakeField<double>("DoubleField");
      auto fieldPt2 = model->MakeField<std::string>("StringField");
      auto fieldPt3 = model->MakeField<std::vector<std::int32_t>>("VecIntField");
      auto fieldPt4 = model->MakeField<std::string>("StringField2");
      auto fieldPt5 = model->MakeField<std::vector<double>>("VecDoubleField");
      auto fieldPt6 = model->MakeField<float>("floatField");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), directory->GetName(), GetFileName(directory));
   } // flushes content to test.root
   
   FileRaii fileGuard2("test11.root");
   TDirectory *directory2 = fileGuard2.GetFile()->mkdir("Staff11");
   {
      auto model = RNTupleModel::Create();
      auto fieldPt = model->MakeField<double>("Double2Field");
      auto fieldPt2 = model->MakeField<std::string>("String2Field");
      auto fieldPt3 = model->MakeField<std::vector<std::int32_t>>("VecInt2Field");
      auto fieldPt4 = model->MakeField<std::string>("String2Field2");
      // difference to the ntuple above: auto fieldPt5 = model->MakeField<std::vector<double>>("VecDouble2Field");
      auto fieldPt6 = model->MakeField<float>("float2Field");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), directory2->GetName(), GetFileName(directory2));
   } // flushes content to test.root
   
   RNTupleBrowser browseObject(directory);
   browseObject.SetDirectory(directory);
   browseObject.Browse(nullptr);
   browseObject.Browse(nullptr);
   
   RNTupleBrowser browseObject2(directory2);
   browseObject2.SetDirectory(directory2);
   browseObject2.Browse(nullptr);
   browseObject2.Browse(nullptr);
   
   ASSERT_EQ(browseObject.fNTupleBrowsePtrVec.size(), static_cast<std::size_t>(12)); // instead of 16, because only fields directly attached to the RootField should be "browsed"
   for (int i = 0; i < 7; i+=6) {
      EXPECT_STREQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i).get())->GetName(), "DoubleField");
      EXPECT_STREQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+1).get())->GetName(), "StringField");
      EXPECT_STREQ(static_cast<RFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+2).get())->GetName(), "VecIntField");
      EXPECT_STREQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+3).get())->GetName(), "StringField2");
      EXPECT_STREQ(static_cast<RFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+4).get())->GetName(), "VecDoubleField");
      EXPECT_STREQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+5).get())->GetName(), "floatField");
   }
   
   ASSERT_EQ(browseObject2.fNTupleBrowsePtrVec.size(), static_cast<std::size_t>(10)); // instead of 12, because only fields directly attached to the RootField should be "browsed"
   for (int i = 0; i < 7; i+=5) {
      EXPECT_STREQ(static_cast<RNonFolder*>(browseObject2.fNTupleBrowsePtrVec.at(i).get())->GetName(), "Double2Field");
      EXPECT_STREQ(static_cast<RNonFolder*>(browseObject2.fNTupleBrowsePtrVec.at(i+1).get())->GetName(), "String2Field");
      EXPECT_STREQ(static_cast<RFolder*>(browseObject2.fNTupleBrowsePtrVec.at(i+2).get())->GetName(), "VecInt2Field");
      EXPECT_STREQ(static_cast<RNonFolder*>(browseObject2.fNTupleBrowsePtrVec.at(i+3).get())->GetName(), "String2Field2");
      EXPECT_STREQ(static_cast<RNonFolder*>(browseObject2.fNTupleBrowsePtrVec.at(i+4).get())->GetName(), "float2Field");
   }
   
   // browses a couple of times the different root files.
   for (int i = 0; i < 12; ++i) {
      if ( i%6 == 2 || i%6 == 4 ) {
         static_cast<RFolder*>(browseObject.fNTupleBrowsePtrVec.at(i).get())->Browse(nullptr);
      } else {
         static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i).get())->Browse(nullptr);
      }
   }
   for (int i = 0; i < 10; ++i) {
      if ( i%5 == 2 ) {
         static_cast<RFolder*>(browseObject2.fNTupleBrowsePtrVec.at(i).get())->Browse(nullptr);
      } else {
         static_cast<RNonFolder*>(browseObject2.fNTupleBrowsePtrVec.at(i).get())->Browse(nullptr);
      }
   }
   for (int i = 0; i < 12; ++i) {
      if ( i%6 == 2 || i%6 == 4 ) {
         static_cast<RFolder*>(browseObject.fNTupleBrowsePtrVec.at(i).get())->Browse(nullptr);
      } else {
         static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i).get())->Browse(nullptr);
      }
   }
   for (int i = 0; i < 10; ++i) {
      if ( i%5 == 2 ) {
         static_cast<RFolder*>(browseObject2.fNTupleBrowsePtrVec.at(i).get())->Browse(nullptr);
      } else {
         static_cast<RNonFolder*>(browseObject2.fNTupleBrowsePtrVec.at(i).get())->Browse(nullptr);
      }
   }
   
   
   ASSERT_EQ(browseObject.fNTupleBrowsePtrVec.size(), static_cast<std::size_t>(20));
   for (int i = 0; i < 7; i+=6) {
      EXPECT_STREQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i).get())->GetName(), "DoubleField");
      EXPECT_STREQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+1).get())->GetName(), "StringField");
      EXPECT_STREQ(static_cast<RFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+2).get())->GetName(), "VecIntField");
      EXPECT_STREQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+3).get())->GetName(), "StringField2");
      EXPECT_STREQ(static_cast<RFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+4).get())->GetName(), "VecDoubleField");
      EXPECT_STREQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+5).get())->GetName(), "floatField");
   }
   
   ASSERT_EQ(browseObject2.fNTupleBrowsePtrVec.size(), static_cast<std::size_t>(14));
   for (int i = 0; i < 7; i+=5) {
      EXPECT_STREQ(static_cast<RNonFolder*>(browseObject2.fNTupleBrowsePtrVec.at(i).get())->GetName(), "Double2Field");
      EXPECT_STREQ(static_cast<RNonFolder*>(browseObject2.fNTupleBrowsePtrVec.at(i+1).get())->GetName(), "String2Field");
      EXPECT_STREQ(static_cast<RFolder*>(browseObject2.fNTupleBrowsePtrVec.at(i+2).get())->GetName(), "VecInt2Field");
      EXPECT_STREQ(static_cast<RNonFolder*>(browseObject2.fNTupleBrowsePtrVec.at(i+3).get())->GetName(), "String2Field2");
      EXPECT_STREQ(static_cast<RNonFolder*>(browseObject2.fNTupleBrowsePtrVec.at(i+4).get())->GetName(), "float2Field");
   }
   
   for (int i = 12; i < 20; i+=2) {
      EXPECT_STREQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i).get())->GetName(), "std::int32_t");
      EXPECT_STREQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+1).get())->GetName(), "double");
   }
   for (int i = 10; i < 14; ++i) {
      EXPECT_STREQ(static_cast<RNonFolder*>(browseObject2.fNTupleBrowsePtrVec.at(i).get())->GetName(), "std::int32_t");
   }
}

