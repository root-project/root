// This hides the TBrowser.h file for this unit test and allows to define an initializable custom TBrowser class.
#define ROOT_TBrowser

#include <ROOT/RBrowseVisitor.hxx>
#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleBrowser.hxx>
#include <ROOT/RNTupleModel.hxx>

#include <TDirectory.h>
#include <TFile.h>

#include "gtest/gtest.h"

#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

// can't check how it's displayed in TBrowser
// can't compare histograms
// can count how often the Add function was called, but not how the result would be.

using RNTupleReader = ROOT::Experimental::RNTupleReader;
using RNTupleBrowser = ROOT::Experimental::RNTupleBrowser;
using RNTupleWriter = ROOT::Experimental::RNTupleWriter;
using RNTupleModel = ROOT::Experimental::RNTupleModel;
using RNonFolder = ROOT::Experimental::RNTupleFieldElement;
using RFolder = ROOT::Experimental::RNTupleFieldElementFolder;

namespace {
   
   /**
   * An RAII wrapper around an open temporary file on disk. It cleans up the guarded file when the wrapper object
   * goes out of scope.
   */
class FileRaii {
private:
   std::string fPath;
   TFile *file = TFile::Open(fPath.c_str(), "RECREATE");
public:
   FileRaii(const std::string &path) : fPath(path) { }
   FileRaii(const FileRaii&) = delete;
   FileRaii& operator=(const FileRaii&) = delete;
   ~FileRaii() {
      file->Close();
      std::remove(fPath.c_str());
   }
};
    
} // anonymous namespace


class TBrowser{
   // no add function here, because
};




TEST(RNTupleBrowse, Floattest)
{
   FileRaii fileGuard("test.root");
   {
      auto model = RNTupleModel::Create();
      auto fieldPt = model->MakeField<float>("FloatField");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "Staff", "test.root");
      *fieldPt = 5.0f;
      ntuple->Fill();
   } // flushes content to test.root
   
   TDirectory directory;
   directory.SetName("Staff");
   TDirectory* directoryPtr = &directory;
   RNTupleBrowser browseObject(directoryPtr, 1000);
   browseObject.SetDirectory(directoryPtr);
   TBrowser browsi;
   TBrowser* browsiPtr = &browsi;
   browseObject.Browse(browsiPtr);
      
   ASSERT_EQ(browseObject.fNTupleBrowsePtrVec.size(), static_cast<std::size_t>(1));
   ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(0))->fType, ROOT::Experimental::numericDatatype_float);
   ASSERT_EQ(browseObject.GetfUnitTest(), 1001);
}



TEST(RNTupleBrowse, Stringtest)
{
   FileRaii fileGuard("test.root");
   {
      auto model = RNTupleModel::Create();
      auto fieldPt = model->MakeField<std::string>("StringField");
      auto fieldPt2 = model->MakeField<std::string>("StringField2");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "Staff2", "test.root");
      *fieldPt = "TestString";
      ntuple->Fill();
   } // flushes content to test.root
   
   TDirectory directory;
   directory.SetName("Staff2");
   TDirectory* directoryPtr = &directory;
   RNTupleBrowser browseObject(directoryPtr, 1000);
   browseObject.SetDirectory(directoryPtr);
   TBrowser browsi;
   TBrowser* browsiPtr = &browsi;
   browseObject.Browse(browsiPtr);
   
   ASSERT_EQ(browseObject.fNTupleBrowsePtrVec.size(), static_cast<std::size_t>(2));
   ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(0))->fType, ROOT::Experimental::numericDatatype_noHist);
   ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(1))->fType, ROOT::Experimental::numericDatatype_noHist);
   ASSERT_EQ(browseObject.GetfUnitTest(), 1002);
}



TEST(RNTupleBrowse, MixedFieldstest)
{
   FileRaii fileGuard("test.root");
   {
      auto model = RNTupleModel::Create();
      auto fieldPt = model->MakeField<double>("DoubleField");
      auto fieldPt2 = model->MakeField<std::string>("StringField");
      auto fieldPt3 = model->MakeField<int>("IntField");
      auto fieldPt4 = model->MakeField<std::string>("StringField2");
      auto fieldPt5 = model->MakeField<float>("floatField");
      auto fieldPt6 = model->MakeField<std::uint32_t>("uInt32Field");
      auto fieldPt7 = model->MakeField<std::uint64_t>("uInt64Field");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "Staff2", "test.root");
      *fieldPt = 7.9;
      ntuple->Fill();
   } // flushes content to test.root
   
   TDirectory directory;
   directory.SetName("Staff2");
   TDirectory* directoryPtr = &directory;
   RNTupleBrowser browseObject(directoryPtr, 1000);
   browseObject.SetDirectory(directoryPtr);
   TBrowser browsi;
   TBrowser* browsiPtr = &browsi;
   browseObject.Browse(browsiPtr);
      
   ASSERT_EQ(browseObject.fNTupleBrowsePtrVec.size(), static_cast<std::size_t>(7));
   ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(0))->fType, ROOT::Experimental::numericDatatype_double);
   ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(1))->fType, ROOT::Experimental::numericDatatype_noHist);
   ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(2))->fType, ROOT::Experimental::numericDatatype_Int32);
   ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(3))->fType, ROOT::Experimental::numericDatatype_noHist);
   ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(4))->fType, ROOT::Experimental::numericDatatype_float);
   ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(5))->fType, ROOT::Experimental::numericDatatype_UInt32);
   ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(6))->fType, ROOT::Experimental::numericDatatype_UInt64);
   ASSERT_EQ(browseObject.GetfUnitTest(), 1007);
}

TEST(RNTupleBrowse, Browsetest)
{
   FileRaii fileGuard("test.root");
   {
      auto model = RNTupleModel::Create();
      auto fieldPt = model->MakeField<double>("DoubleField");
      auto fieldPt2 = model->MakeField<std::string>("StringField");
      auto fieldPt3 = model->MakeField<int>("IntField");
      auto fieldPt4 = model->MakeField<std::string>("StringField2");
      auto fieldPt5 = model->MakeField<float>("floatField");
      auto fieldPt6 = model->MakeField<std::uint32_t>("uInt32Field");
      auto fieldPt7 = model->MakeField<std::uint64_t>("uInt64Field");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "Staff2", "test.root");
      *fieldPt = 7.9;
      ntuple->Fill();
   } // flushes content to test.root
   
   TDirectory directory;
   directory.SetName("Staff2");
   TDirectory* directoryPtr = &directory;
   RNTupleBrowser browseObject(directoryPtr, 1000);
   browseObject.SetDirectory(directoryPtr);
   TBrowser browsi;
   TBrowser* browsiPtr = &browsi;
   browseObject.Browse(browsiPtr);
   
   ASSERT_EQ(browseObject.fNTupleBrowsePtrVec.size(), static_cast<std::size_t>(7));
   ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(0))->fType, ROOT::Experimental::numericDatatype_double);
   ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(1))->fType, ROOT::Experimental::numericDatatype_noHist);
   ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(2))->fType, ROOT::Experimental::numericDatatype_Int32);
   ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(3))->fType, ROOT::Experimental::numericDatatype_noHist);
   ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(4))->fType, ROOT::Experimental::numericDatatype_float);
   ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(5))->fType, ROOT::Experimental::numericDatatype_UInt32);
   ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(6))->fType, ROOT::Experimental::numericDatatype_UInt64);
   ASSERT_EQ(browseObject.GetfUnitTest(), 1007);
   // situation up to here is exaclty the same as in the unit test above.
   
   
   // no ASSERT_EQ here, it only checks if calling the Browse-method leads to an error.
   for(int i = 0; i < 7; ++i)
   static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i))->Browse(browsiPtr);
}

TEST(RNTupleBrowse, MultipleBrowsetest)
{
   FileRaii fileGuard("test.root");
   {
      auto model = RNTupleModel::Create();
      auto fieldPt = model->MakeField<double>("DoubleField");
      auto fieldPt2 = model->MakeField<std::string>("StringField");
      auto fieldPt3 = model->MakeField<int>("IntField");
      auto fieldPt4 = model->MakeField<std::string>("StringField2");
      auto fieldPt5 = model->MakeField<float>("floatField");
      auto fieldPt6 = model->MakeField<std::uint32_t>("uInt32Field");
      auto fieldPt7 = model->MakeField<std::uint64_t>("uInt64Field");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "Staff2", "test.root");
      *fieldPt = 7.9;
      ntuple->Fill();
   } // flushes content to test.root
   
   TDirectory directory;
   directory.SetName("Staff2");
   TDirectory* directoryPtr = &directory;
   RNTupleBrowser browseObject(directoryPtr, 1000);
   browseObject.SetDirectory(directoryPtr);
   TBrowser browsi;
   TBrowser* browsiPtr = &browsi;
   browseObject.Browse(browsiPtr);
   browseObject.Browse(browsiPtr);
   
   ASSERT_EQ(browseObject.fNTupleBrowsePtrVec.size(), static_cast<std::size_t>(14));
   for(int i = 0; i < 8; i+=7) {
   ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i))->fType, ROOT::Experimental::numericDatatype_double);
   ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+1))->fType, ROOT::Experimental::numericDatatype_noHist);
   ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+2))->fType, ROOT::Experimental::numericDatatype_Int32);
   ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+3))->fType, ROOT::Experimental::numericDatatype_noHist);
   ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+4))->fType, ROOT::Experimental::numericDatatype_float);
   ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+5))->fType, ROOT::Experimental::numericDatatype_UInt32);
   ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+6))->fType, ROOT::Experimental::numericDatatype_UInt64);
   ASSERT_EQ(browseObject.GetfUnitTest(), 1014);
   }
   
   // no ASSERT_EQ here, it only checks if calling the Browse-method leads to an error.
   for(int i = 0; i < 14; ++i)
      static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i))->Browse(browsiPtr);
   for(int i = 0; i < 14; ++i)
      static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i))->Browse(browsiPtr);
}

TEST(RNTupleBrowse, VecBrowsetest)
{
   FileRaii fileGuard("test.root");
   {
      auto model = RNTupleModel::Create();
      auto fieldPt = model->MakeField<double>("DoubleField");
      auto fieldPt2 = model->MakeField<std::string>("StringField");
      auto fieldPt3 = model->MakeField<std::vector<int>>("VecIntField");
      auto fieldPt4 = model->MakeField<std::string>("StringField2");
      auto fieldPt5 = model->MakeField<float>("floatField");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "Staff2", "test.root");
   } // flushes content to test.root
   
   TDirectory directory;
   directory.SetName("Staff2");
   TDirectory* directoryPtr = &directory;
   RNTupleBrowser browseObject(directoryPtr, 1000);
   browseObject.SetDirectory(directoryPtr);
   TBrowser browsi;
   TBrowser* browsiPtr = &browsi;
   browseObject.Browse(browsiPtr);
   browseObject.Browse(browsiPtr);
   
   ASSERT_EQ(browseObject.fNTupleBrowsePtrVec.size(), static_cast<std::size_t>(10)); // instead of 12, because only fields directly attached to the RootField should be "browsed"
   for(int i = 0; i < 6; i+=5) {
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i))->fType, ROOT::Experimental::numericDatatype_double);
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+1))->fType, ROOT::Experimental::numericDatatype_noHist);
      EXPECT_EQ(static_cast<RFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+2))->GetFieldName(), "VecIntField");
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+3))->fType, ROOT::Experimental::numericDatatype_noHist);
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+4))->fType, ROOT::Experimental::numericDatatype_float);
   }
   for(int j = 0; j < 3; ++j) {
   for(int i = 0; i < 10; ++i)
      if(i%5 == 2) static_cast<RFolder*>(browseObject.fNTupleBrowsePtrVec.at(i))->Browse(browsiPtr);
      else
      static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i))->Browse(browsiPtr);
   }
   ASSERT_EQ(browseObject.fNTupleBrowsePtrVec.size(), static_cast<std::size_t>(16)); // create 2*3=6 instances of int-BrowseFields stored.
   for(int i = 10; i < 16; ++i) ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i))->fType, ROOT::Experimental::numericDatatype_parentIsVec);
}

TEST(RNTupleBrowse, DoubleVecBrowsetest)
{
   FileRaii fileGuard("test.root");
   {
      auto model = RNTupleModel::Create();
      auto fieldPt = model->MakeField<double>("DoubleField");
      auto fieldPt2 = model->MakeField<std::string>("StringField");
      auto fieldPt3 = model->MakeField<std::vector<int>>("VecIntField");
      auto fieldPt4 = model->MakeField<std::string>("StringField2");
      auto fieldPt5 = model->MakeField<std::vector<int>>("VecIntField2");
      auto fieldPt6 = model->MakeField<float>("floatField");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "Staff2", "test.root");
   } // flushes content to test.root
   
   TDirectory directory;
   directory.SetName("Staff2");
   TDirectory* directoryPtr = &directory;
   RNTupleBrowser browseObject(directoryPtr, 1000);
   browseObject.SetDirectory(directoryPtr);
   TBrowser browsi;
   TBrowser* browsiPtr = &browsi;
   browseObject.Browse(browsiPtr);
   browseObject.Browse(browsiPtr);
   
   ASSERT_EQ(browseObject.fNTupleBrowsePtrVec.size(), static_cast<std::size_t>(12)); // instead of 16, because only fields directly attached to the RootField should be "browsed"
   for(int i = 0; i < 7; i+=6) {
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i))->fType, ROOT::Experimental::numericDatatype_double);
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+1))->fType, ROOT::Experimental::numericDatatype_noHist);
      EXPECT_EQ(static_cast<RFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+2))->GetFieldName(), "VecIntField");
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+3))->fType, ROOT::Experimental::numericDatatype_noHist);
      EXPECT_EQ(static_cast<RFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+4))->GetFieldName(), "VecIntField2");
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+5))->fType, ROOT::Experimental::numericDatatype_float);
   }
   for(int j = 0; j < 3; ++j) {
      for(int i = 0; i < 12; ++i)
         if(i%6 == 2 || i%6 == 4) static_cast<RFolder*>(browseObject.fNTupleBrowsePtrVec.at(i))->Browse(browsiPtr);
         else
            static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i))->Browse(browsiPtr);
   }
   ASSERT_EQ(browseObject.fNTupleBrowsePtrVec.size(), static_cast<std::size_t>(24)); // create 2*2*3=12 instances of int-BrowseFields stored.
   for(int i = 12; i < 24; ++i)
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i))->fType, ROOT::Experimental::numericDatatype_parentIsVec);
   ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(0))->fType, ROOT::Experimental::numericDatatype_double);
   ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(1))->fType, ROOT::Experimental::numericDatatype_noHist);
   EXPECT_EQ(static_cast<RFolder*>(browseObject.fNTupleBrowsePtrVec.at(2))->GetFieldName(), "VecIntField");
   ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(3))->fType, ROOT::Experimental::numericDatatype_noHist);
   EXPECT_EQ(static_cast<RFolder*>(browseObject.fNTupleBrowsePtrVec.at(4))->GetFieldName(), "VecIntField2");
   ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(5))->fType, ROOT::Experimental::numericDatatype_float);
}


TEST(RNTupleBrowse, MultipleRootFiletest)
{
   FileRaii fileGuard("test.root");
   {
      auto model = RNTupleModel::Create();
      auto fieldPt = model->MakeField<double>("DoubleField");
      auto fieldPt2 = model->MakeField<std::string>("StringField");
      auto fieldPt3 = model->MakeField<std::vector<int>>("VecIntField");
      auto fieldPt4 = model->MakeField<std::string>("StringField2");
      auto fieldPt5 = model->MakeField<std::vector<int>>("VecIntField2");
      auto fieldPt6 = model->MakeField<float>("floatField");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "Staff2", "test.root");
   } // flushes content to test.root
   
   FileRaii fileGuard2("test2.root");
   {
      auto model = RNTupleModel::Create();
      auto fieldPt = model->MakeField<double>("Double2Field");
      auto fieldPt2 = model->MakeField<std::string>("String2Field");
      auto fieldPt3 = model->MakeField<std::vector<int>>("VecInt2Field");
      auto fieldPt4 = model->MakeField<std::string>("String2Field2");
      auto fieldPt5 = model->MakeField<std::vector<int>>("VecInt2Field2");
      auto fieldPt6 = model->MakeField<float>("float2Field");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "Staff22", "test2.root");
   } // flushes content to test.root
   
   TDirectory directory;
   directory.SetName("Staff2");
   TDirectory* directoryPtr = &directory;
   RNTupleBrowser browseObject(directoryPtr, 1000);
   browseObject.SetDirectory(directoryPtr);
   TBrowser browsi;
   TBrowser* browsiPtr = &browsi;
   browseObject.Browse(browsiPtr);
   browseObject.Browse(browsiPtr);
   
   TDirectory directory2;
   directory2.SetName("Staff22");
   TDirectory* directoryPtr2 = &directory2;
   RNTupleBrowser browseObject2(directoryPtr2, 2000);
   browseObject2.SetDirectory(directoryPtr2);
   TBrowser browsi2;
   TBrowser* browsiPtr2 = &browsi2;
   browseObject2.Browse(browsiPtr2);
   browseObject2.Browse(browsiPtr2);
   
   ASSERT_EQ(browseObject.fNTupleBrowsePtrVec.size(), static_cast<std::size_t>(12)); // instead of 16, because only fields directly attached to the RootField should be "browsed"
   for(int i = 0; i < 7; i+=6) {
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i))->fType, ROOT::Experimental::numericDatatype_double);
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+1))->fType, ROOT::Experimental::numericDatatype_noHist);
      EXPECT_EQ(static_cast<RFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+2))->GetFieldName(), "VecIntField");
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+3))->fType, ROOT::Experimental::numericDatatype_noHist);
      EXPECT_EQ(static_cast<RFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+4))->GetFieldName(), "VecIntField2");
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+5))->fType, ROOT::Experimental::numericDatatype_float);
   }
   ASSERT_EQ(browseObject.fNTupleBrowsePtrVec.size(), static_cast<std::size_t>(12)); // instead of 16, because only fields directly attached to the RootField should be "browsed"
   for(int i = 0; i < 7; i+=6) {
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject2.fNTupleBrowsePtrVec.at(i))->fType, ROOT::Experimental::numericDatatype_double);
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject2.fNTupleBrowsePtrVec.at(i+1))->fType, ROOT::Experimental::numericDatatype_noHist);
      EXPECT_EQ(static_cast<RFolder*>(browseObject2.fNTupleBrowsePtrVec.at(i+2))->GetFieldName(), "VecInt2Field");
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject2.fNTupleBrowsePtrVec.at(i+3))->fType, ROOT::Experimental::numericDatatype_noHist);
      EXPECT_EQ(static_cast<RFolder*>(browseObject2.fNTupleBrowsePtrVec.at(i+4))->GetFieldName(), "VecInt2Field2");
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject2.fNTupleBrowsePtrVec.at(i+5))->fType, ROOT::Experimental::numericDatatype_float);
   }
}

TEST(RNTupleBrowse, MultipleRootFileMultipleBrowsetest)
{
   FileRaii fileGuard("test.root");
   {
      auto model = RNTupleModel::Create();
      auto fieldPt = model->MakeField<double>("DoubleField");
      auto fieldPt2 = model->MakeField<std::string>("StringField");
      auto fieldPt3 = model->MakeField<std::vector<int>>("VecIntField");
      auto fieldPt4 = model->MakeField<std::string>("StringField2");
      auto fieldPt5 = model->MakeField<std::vector<int>>("VecIntField2");
      auto fieldPt6 = model->MakeField<float>("floatField");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "Staff2", "test.root");
   } // flushes content to test.root
   
   FileRaii fileGuard2("test2.root");
   {
      auto model = RNTupleModel::Create();
      auto fieldPt = model->MakeField<double>("Double2Field");
      auto fieldPt2 = model->MakeField<std::string>("String2Field");
      auto fieldPt3 = model->MakeField<std::vector<int>>("VecInt2Field");
      auto fieldPt4 = model->MakeField<std::string>("String2Field2");
      auto fieldPt5 = model->MakeField<std::vector<int>>("VecInt2Field2");
      auto fieldPt6 = model->MakeField<float>("float2Field");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "Staff22", "test2.root");
   } // flushes content to test.root
   
   TDirectory directory;
   directory.SetName("Staff2");
   TDirectory* directoryPtr = &directory;
   RNTupleBrowser browseObject(directoryPtr, 1000);
   browseObject.SetDirectory(directoryPtr);
   TBrowser browsi;
   TBrowser* browsiPtr = &browsi;
   browseObject.Browse(browsiPtr);
   browseObject.Browse(browsiPtr);
   
   TDirectory directory2;
   directory2.SetName("Staff22");
   TDirectory* directoryPtr2 = &directory2;
   RNTupleBrowser browseObject2(directoryPtr2, 2000);
   browseObject2.SetDirectory(directoryPtr2);
   TBrowser browsi2;
   TBrowser* browsiPtr2 = &browsi2;
   browseObject2.Browse(browsiPtr2);
   browseObject2.Browse(browsiPtr2);
   
   ASSERT_EQ(browseObject.fNTupleBrowsePtrVec.size(), static_cast<std::size_t>(12)); // instead of 16, because only fields directly attached to the RootField should be "browsed"
   for(int i = 0; i < 7; i+=6) {
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i))->fType, ROOT::Experimental::numericDatatype_double);
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+1))->fType, ROOT::Experimental::numericDatatype_noHist);
      EXPECT_EQ(static_cast<RFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+2))->GetFieldName(), "VecIntField");
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+3))->fType, ROOT::Experimental::numericDatatype_noHist);
      EXPECT_EQ(static_cast<RFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+4))->GetFieldName(), "VecIntField2");
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+5))->fType, ROOT::Experimental::numericDatatype_float);
   }
   ASSERT_EQ(browseObject.fNTupleBrowsePtrVec.size(), static_cast<std::size_t>(12)); // instead of 16, because only fields directly attached to the RootField should be "browsed"
   for(int i = 0; i < 7; i+=6) {
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject2.fNTupleBrowsePtrVec.at(i))->fType, ROOT::Experimental::numericDatatype_double);
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject2.fNTupleBrowsePtrVec.at(i+1))->fType, ROOT::Experimental::numericDatatype_noHist);
      EXPECT_EQ(static_cast<RFolder*>(browseObject2.fNTupleBrowsePtrVec.at(i+2))->GetFieldName(), "VecInt2Field");
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject2.fNTupleBrowsePtrVec.at(i+3))->fType, ROOT::Experimental::numericDatatype_noHist);
      EXPECT_EQ(static_cast<RFolder*>(browseObject2.fNTupleBrowsePtrVec.at(i+4))->GetFieldName(), "VecInt2Field2");
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject2.fNTupleBrowsePtrVec.at(i+5))->fType, ROOT::Experimental::numericDatatype_float);
   }
   for(int i = 0; i < 12; ++i)
      if(i%6 == 2 || i%6 == 4) static_cast<RFolder*>(browseObject.fNTupleBrowsePtrVec.at(i))->Browse(browsiPtr);
      else
         static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i))->Browse(browsiPtr);
   for(int i = 0; i < 12; ++i)
      if(i%6 == 2 || i%6 == 4) static_cast<RFolder*>(browseObject2.fNTupleBrowsePtrVec.at(i))->Browse(browsiPtr2);
      else
         static_cast<RNonFolder*>(browseObject2.fNTupleBrowsePtrVec.at(i))->Browse(browsiPtr2);
   for(int i = 0; i < 12; ++i)
      if(i%6 == 2 || i%6 == 4) static_cast<RFolder*>(browseObject.fNTupleBrowsePtrVec.at(i))->Browse(browsiPtr);
      else
         static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i))->Browse(browsiPtr);
   for(int i = 0; i < 12; ++i)
      if(i%6 == 2 || i%6 == 4) static_cast<RFolder*>(browseObject2.fNTupleBrowsePtrVec.at(i))->Browse(browsiPtr2);
      else
         static_cast<RNonFolder*>(browseObject2.fNTupleBrowsePtrVec.at(i))->Browse(browsiPtr2);
   ASSERT_EQ(browseObject.fNTupleBrowsePtrVec.size(), static_cast<std::size_t>(20)); // instead of 16, because only fields directly attached to the RootField should be "browsed"
   for(int i = 0; i < 7; i+=6) {
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i))->fType, ROOT::Experimental::numericDatatype_double);
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+1))->fType, ROOT::Experimental::numericDatatype_noHist);
      EXPECT_EQ(static_cast<RFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+2))->GetFieldName(), "VecIntField");
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+3))->fType, ROOT::Experimental::numericDatatype_noHist);
      EXPECT_EQ(static_cast<RFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+4))->GetFieldName(), "VecIntField2");
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i+5))->fType, ROOT::Experimental::numericDatatype_float);
   }
   ASSERT_EQ(browseObject.fNTupleBrowsePtrVec.size(), static_cast<std::size_t>(20)); // instead of 16, because only fields directly attached to the RootField should be "browsed"
   for(int i = 0; i < 7; i+=6) {
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject2.fNTupleBrowsePtrVec.at(i))->fType, ROOT::Experimental::numericDatatype_double);
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject2.fNTupleBrowsePtrVec.at(i+1))->fType, ROOT::Experimental::numericDatatype_noHist);
      EXPECT_EQ(static_cast<RFolder*>(browseObject2.fNTupleBrowsePtrVec.at(i+2))->GetFieldName(), "VecInt2Field");
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject2.fNTupleBrowsePtrVec.at(i+3))->fType, ROOT::Experimental::numericDatatype_noHist);
      EXPECT_EQ(static_cast<RFolder*>(browseObject2.fNTupleBrowsePtrVec.at(i+4))->GetFieldName(), "VecInt2Field2");
      ASSERT_EQ(static_cast<RNonFolder*>(browseObject2.fNTupleBrowsePtrVec.at(i+5))->fType, ROOT::Experimental::numericDatatype_float);
   }
   for(int i = 12; i < 20; ++i) ASSERT_EQ(static_cast<RNonFolder*>(browseObject.fNTupleBrowsePtrVec.at(i))->fType, ROOT::Experimental::numericDatatype_parentIsVec);
   for(int i = 12; i < 20; ++i) ASSERT_EQ(static_cast<RNonFolder*>(browseObject2.fNTupleBrowsePtrVec.at(i))->fType, ROOT::Experimental::numericDatatype_parentIsVec);
}




   
