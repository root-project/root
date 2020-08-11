#include "ntuple_test.hxx"

TEST(RPageStorageFriends, Basic)
{
   FileRaii fileGuard1("test_ntuple_friends1.root");
   FileRaii fileGuard2("test_ntuple_friends2.root");

   auto model1 = RNTupleModel::Create();
   auto fieldPt = model1->MakeField<float>("pt", 42.0);

   auto model2 = RNTupleModel::Create();
   auto fieldEta = model1->MakeField<float>("eta", 24.0);

   {
      RPageSinkFile sink("ntpl", fileGuard1.GetPath(), RNTupleWriteOptions());
      sink.Create(*model1.get());
      sink.CommitDataset();
      model1 = nullptr;
   }
   {
      RPageSinkFile sink("ntpl", fileGuard2.GetPath(), RNTupleWriteOptions());
      sink.Create(*model2.get());
      sink.CommitDataset();
      model2 = nullptr;
   }

   std::vector<std::unique_ptr<RPageSource>> realSources;
   realSources.emplace_back(std::make_unique<RPageSourceFile>("ntpl", fileGuard1.GetPath(), RNTupleReadOptions()));
   realSources.emplace_back(std::make_unique<RPageSourceFile>("ntpl", fileGuard2.GetPath(), RNTupleReadOptions()));
   RPageSourceFriends friendSource("myNTuple", realSources);
   friendSource.Attach();
}
