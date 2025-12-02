#include <ROOT/RFieldBase.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleReader.hxx>
#include <ROOT/RNTupleWriter.hxx>

#include <cstdio>
#include <string>
#include <utility>

#include "StreamerBeyond.hxx"
#include "gtest/gtest.h"

namespace {

class FileRaii {
private:
   std::string fPath;
   bool fPreserveFile = false;

public:
   explicit FileRaii(const std::string &path) : fPath(path) {}
   FileRaii(FileRaii &&) = default;
   FileRaii(const FileRaii &) = delete;
   FileRaii &operator=(FileRaii &&) = default;
   FileRaii &operator=(const FileRaii &) = delete;
   ~FileRaii()
   {
      if (!fPreserveFile)
         std::remove(fPath.c_str());
   }
   std::string GetPath() const { return fPath; }

   // Useful if you want to keep a test file after the test has finished running
   // for debugging purposes. Should only be used locally and never pushed.
   void PreserveFile() { fPreserveFile = true; }
};

} // anonymous namespace

TEST(RField, StreamerBeyond)
{
   FileRaii fileGuard("test_ntuple_rfield_streamer_beyond.root");

   {
      auto model = ROOT::RNTupleModel::Create();
      auto f = ROOT::RFieldBase::Create("f", "StreamerBeyond").Unwrap();
      EXPECT_TRUE(dynamic_cast<ROOT::RStreamerField *>(f.get()));
      model->AddField(std::move(f));
      auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());

      auto ptr = writer->GetModel().GetDefaultEntry().GetPtr<StreamerBeyond>("f");
      ptr->fOne = std::vector<std::int64_t>(100000000, -1);
      ptr->fTwo = std::vector<std::int64_t>(100000000, -2);

      writer->Fill();
   }

   auto reader = ROOT::RNTupleReader::Open("ntpl", fileGuard.GetPath());
   ASSERT_EQ(1u, reader->GetNEntries());
   StreamerBeyond sb;
   auto view = reader->GetView("f", &sb, "StreamerBeyond");

   view(0);

   auto ptr = view.GetValue().GetPtr<StreamerBeyond>();
   EXPECT_EQ(100000000u, ptr->fOne.size());
   EXPECT_EQ(-1, ptr->fOne.at(1000));
   EXPECT_EQ(100000000u, ptr->fTwo.size());
   EXPECT_EQ(-2, ptr->fTwo.at(2000));
}
