#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RFieldVisitor.hxx>

#include <TClass.h>
#include <TFile.h>

#include "gtest/gtest.h"

#include <sstream>
#include <iostream>

using RNTupleReader = ROOT::Experimental::RNTupleReader;
using RNTupleWriter = ROOT::Experimental::RNTupleWriter;
using RNTupleModel = ROOT::Experimental::RNTupleModel;
using RFieldBase = ROOT::Experimental::Detail::RFieldBase;
using RPrintVisitor = ROOT::Experimental::RPrintVisitor;
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
        FileRaii(const std::string &path) : fPath(path)
        {
        }
        FileRaii(const FileRaii&) = delete;
        FileRaii& operator=(const FileRaii&) = delete;
        ~FileRaii() {
            std::remove(fPath.c_str());
        }
    };
    
} // anonymous namespace

TEST(RNtuplePrint, FullString)
{
    TFile *file = TFile::Open("test.root", "RECREATE");
    FileRaii fileGuard("test.root");
    {
    auto model = RNTupleModel::Create();
    auto fieldPt = model->MakeField<float>("pt");
    auto ntuple = RNTupleWriter::Recreate(std::move(model), "Staff", "test.root");
    //*fieldPt = 5.0f;
    //ntuple->Fill();
    }
    auto model2 = RNTupleModel::Create();
    auto fieldPt2 = model2->MakeField<float>("pt");
    auto ntuple2 = RNTupleReader::Open(std::move(model2), "Staff", "test.root");
    std::ostringstream os;
    ntuple2->Print(os);
    std::string fString{"******************************* NTUPLE ******************************\n* Ntuple  : Staff                                                   *\n* Entries : 0                                                       *\n*********************************************************************\n* Field    0 : pt (float)                                           *\n*********************************************************************\n"};
    EXPECT_EQ(fString, os.str());
    file->Close();
}

TEST(RNtuplePrint, IntTest)
{
    std::stringstream os;
    RPrintVisitor fPrintVisitor(os);
    RField<int> fTestname("fIntTest");
    fTestname.AcceptVisitor(fPrintVisitor);
    std::string expected{"* Field   -1 : fIntTest (std::int32_t)                              *\n"};
    EXPECT_EQ(expected, os.str());
}

TEST(RNtuplePrint, FloatTest)
{
    std::stringstream os;
    RPrintVisitor fPrintVisitor(os);
    RField<float> fTestname("fFloatTest");
    fTestname.AcceptVisitor(fPrintVisitor);
    std::string expected{"* Field   -1 : fFloatTest (float)                                   *\n"};
    EXPECT_EQ(expected, os.str());
}

/*
TEST(RNtuplePrint, ShortTest)
{
    std::stringstream os;
    RPrintVisitor fPrintVisitor(os);
    RField<short> fTestname("fShortTest");
    fTestname.AcceptVisitor(fPrintVisitor);
    std::string expected{"* Field   -1 : fShortTest (short)                                   *\n"};
    EXPECT_EQ(expected, os.str());
}*/
