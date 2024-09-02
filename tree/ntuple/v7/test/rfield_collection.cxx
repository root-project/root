#include "ntuple_test.hxx"

TEST(RNTuple, CollectionFieldEntries)
{
   {
      auto collectionModel = RNTupleModel::CreateBare();
      collectionModel->MakeField<float>("E");
      auto mainModel = RNTupleModel::CreateBare();
      mainModel->MakeCollection("jets", std::move(collectionModel));
      mainModel->Freeze();

      auto entry = mainModel->CreateEntry();
      EXPECT_FALSE(entry->GetPtr<RNTupleCollectionWriter>("jets")->GetEntry().GetPtr<float>("E"));
   }

   {
      auto collectionModel = RNTupleModel::Create();
      auto mainModel = RNTupleModel::CreateBare();
      EXPECT_THROW(mainModel->MakeCollection("jets", std::move(collectionModel)), RException);
   }

   {
      auto collectionModel = RNTupleModel::CreateBare();
      collectionModel->MakeField<float>("E");
      auto mainModel = RNTupleModel::Create();
      mainModel->MakeCollection("jets", std::move(collectionModel));
      mainModel->Freeze();
      EXPECT_EQ(mainModel->GetModelId(), mainModel->GetDefaultEntry().GetModelId());
      EXPECT_NE(mainModel->GetModelId(),
                mainModel->GetDefaultEntry().GetPtr<RNTupleCollectionWriter>("jets")->GetEntry().GetModelId());
      EXPECT_EQ(mainModel->GetDefaultEntry().GetPtr<RNTupleCollectionWriter>("jets")->GetEntry().GetModelId(),
                dynamic_cast<const RCollectionField &>(mainModel->GetField("jets")).GetModelId());
   }

   {
      auto collectionModel = RNTupleModel::Create();
      auto ptrE = collectionModel->MakeField<float>("E");
      auto mainModel = RNTupleModel::Create();
      auto ptrCollectionWriter = mainModel->MakeCollection("jets", std::move(collectionModel));
      mainModel->Freeze();

      // We preserve the pointers from the collection model default entry...
      EXPECT_EQ(mainModel->GetDefaultEntry().GetPtr<RNTupleCollectionWriter>("jets"), ptrCollectionWriter);
      EXPECT_EQ(mainModel->GetDefaultEntry().GetPtr<RNTupleCollectionWriter>("jets")->GetEntry().GetPtr<float>("E"),
                ptrE);

      // But new main entries need to (re-)bind the collection writer entries
      auto entry = mainModel->CreateEntry();
      EXPECT_FALSE(entry->GetPtr<RNTupleCollectionWriter>("jets")->GetEntry().GetPtr<float>("E"));
   }
}

TEST(RNTuple, CollectionFieldSafety)
{
   FileRaii fileGuard("test_ntuple_collectionfield_safety.root");

   auto jetsModel = RNTupleModel::CreateBare();
   jetsModel->MakeField<float>("E");
   auto tracksModel = RNTupleModel::CreateBare();
   tracksModel->MakeField<float>("E");

   auto mainModel = RNTupleModel::CreateBare();
   mainModel->MakeCollection("jets", std::move(jetsModel));
   mainModel->MakeCollection("tracks", std::move(tracksModel));
   mainModel->Freeze();

   auto entry = mainModel->CreateEntry();
   auto jetsWriter = entry->GetPtr<RNTupleCollectionWriter>("jets");
   auto tracksWriter = entry->GetPtr<RNTupleCollectionWriter>("tracks");

   // Ensure that we catch mixed-up collection writers
   auto writer = RNTupleWriter::Recreate(std::move(mainModel), "ntpl", fileGuard.GetPath());
   writer->Fill(*entry);
   entry->BindValue<RNTupleCollectionWriter>("jets", tracksWriter);
   entry->BindValue<RNTupleCollectionWriter>("tracks", jetsWriter);
   EXPECT_THROW(writer->Fill(*entry), RException);

   // Ensure that we cannot mix up tokens of collection writers
   auto tokenJetsE = jetsWriter->GetEntry().GetToken("E");
   auto tokenTracksE = tracksWriter->GetEntry().GetToken("E");

   EXPECT_NO_THROW(jetsWriter->GetEntry().EmplaceNewValue(tokenJetsE));
   EXPECT_THROW(jetsWriter->GetEntry().EmplaceNewValue(tokenTracksE), RException);
}

TEST(RNTuple, CollectionFieldMultiEntry)
{
   FileRaii fileGuard("test_ntuple_collectionfield_multi_entry.root");

   auto particleModel = RNTupleModel::CreateBare();
   particleModel->MakeField<float>("energy");
   particleModel->Freeze();

   auto model = RNTupleModel::CreateBare();
   auto particles = model->MakeCollection("particles", std::move(particleModel));

   auto writer = RNTupleWriter::Recreate(std::move(model), "events", fileGuard.GetPath());
   auto entry1 = writer->CreateEntry();
   auto entry2 = writer->CreateEntry();

   auto particleWriter1 = entry1->GetPtr<RNTupleCollectionWriter>("particles");
   particleWriter1->GetEntry().EmplaceNewValue("energy");
   auto energy1 = particleWriter1->GetEntry().GetPtr<float>("energy");

   auto particleWriter2 = entry2->GetPtr<RNTupleCollectionWriter>("particles");
   particleWriter2->GetEntry().EmplaceNewValue("energy");
   auto energy2 = particleWriter2->GetEntry().GetPtr<float>("energy");

   *energy1 = 1.0;
   particleWriter1->Fill();
   writer->Fill(*entry1);

   *energy2 = 2.0;
   particleWriter2->Fill();
   writer->Fill(*entry2);

   writer.reset();

   auto reader = RNTupleReader::Open("events", fileGuard.GetPath());
   EXPECT_EQ(2u, reader->GetNEntries());
   auto viewE = reader->GetView<float>("particles.energy");
   EXPECT_FLOAT_EQ(1.0, viewE(0));
   EXPECT_FLOAT_EQ(2.0, viewE(1));
}
