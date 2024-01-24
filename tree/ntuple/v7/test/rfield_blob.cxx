#include "ntuple_test.hxx"

#include <TMemFile.h>

TEST(RField, Blob)
{
   FileRaii fileGuard("test_ntuple_rfield_blob.root");

   {
      auto model = RNTupleModel::Create();
      auto fldBlob = model->MakeField<std::vector<std::byte>>("blob");
      auto writer = RNTupleWriter::Recreate(std::move(model), "f", fileGuard.GetPath());

      TMemFile f("buffer", "CREATE");
      std::string str = "x";
      f.WriteObject(&str, "string");
      f.Close();
      fldBlob->resize(f.GetSize());
      f.CopyTo(fldBlob->data(), f.GetSize());
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("f", fileGuard.GetPath());
   ASSERT_EQ(1U, reader->GetNEntries());
   reader->LoadEntry(0);
   auto fldBlob = reader->GetModel()->GetDefaultEntry()->GetPtr<std::vector<std::byte>>("blob");
   TMemFile f("buffer", TMemFile::ZeroCopyView_t(reinterpret_cast<char *>(fldBlob->data()), fldBlob->size()));
   auto str = f.Get<std::string>("string");
   EXPECT_EQ("x", *str);
}
