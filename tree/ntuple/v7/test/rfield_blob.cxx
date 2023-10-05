#include "ntuple_test.hxx"

#include <TMemFile.h>

TEST(RField, Blob)
{
   FileRaii fileGuard("test_ntuple_rfield_blob.root");

   {
      auto model = RNTupleModel::Create();
      auto ptrBlob = model->MakeField<ROOT::Experimental::RNTupleBLOB>("blob");
      auto writer = RNTupleWriter::Recreate(std::move(model), "f", fileGuard.GetPath());

      TMemFile f("buffer", "CREATE");
      std::string str = "x";
      f.WriteObject(&str, "string");
      f.Close();
      auto data = std::shared_ptr<unsigned char[]>(new unsigned char[f.GetSize()]);
      f.CopyTo(data.get(), f.GetSize());
      ptrBlob->Set(data, f.GetSize());

      writer->Fill();
   }

   auto reader = RNTupleReader::Open("f", fileGuard.GetPath());
   ASSERT_EQ(1U, reader->GetNEntries());
   reader->LoadEntry(0);
   auto ptrBlob = reader->GetModel()->GetDefaultEntry()->Get<ROOT::Experimental::RNTupleBLOB>("blob");
   TMemFile f("buffer",
              TMemFile::ZeroCopyView_t(reinterpret_cast<char *>(ptrBlob->GetData().get()), ptrBlob->GetSize()));
   auto str = f.Get<std::string>("string");
   EXPECT_EQ("x", *str);
}
