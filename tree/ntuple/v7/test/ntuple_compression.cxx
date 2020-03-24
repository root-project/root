#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleDescriptor.hxx>
#include <ROOT/RNTupleDS.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleOptions.hxx>
#include <ROOT/RPageStorage.hxx>
#include <ROOT/RVec.hxx>

#include <TClass.h>
#include <TFile.h>
#include <TRandom3.h>

#include "gtest/gtest.h"

#include <cstdio>
#include <exception>
#include <memory>
#include <string>
#include <utility>
#if __cplusplus >= 201703L
#include <variant>
#endif

using RNTupleWriter = ROOT::Experimental::RNTupleWriter;
using RNTupleWriteOptions = ROOT::Experimental::RNTupleWriteOptions;
using RNTupleModel = ROOT::Experimental::RNTupleModel;

//File Wrapper
namespace {

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

}

TEST(RNTuple, nTupleCompressionTest){
    

    FileRaii fileGuard("test_ntuple_compression.root");
    auto file = TFile::Open(fileGuard.GetPath().c_str(), "RECREATE");

    auto model = RNTupleModel::Create();
    
    const int NEvents = 1000;

    auto& randData = *model->MakeField<std::vector<double>>("data");
   
    auto ntuple = RNTupleWriter::Recreate(std::move(model), "CompTest", fileGuard.GetPath().c_str(), RNTupleWriteOptions());
    
    TRandom3 rnd(42);
    
    for (auto i=0; i<NEvents; ++i)
    {
        randData = rnd.Rndm();
        ntuple->Fill();    
    }
    
    auto compress = (*file)->GetCompressionSettings();
        
    ASSERT_EQ(compress, 404);
    
    file->Close();
    delete file;
}