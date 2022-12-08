#include "ntuple_test.hxx"

TEST(RNTupleProjection, Basics)
{
   auto model = RNTupleModel::Create();
   model->MakeField<float>("met", 0.0);
   //
   //   auto hitModel = RNTupleModel::Create();
   //   hitModel->MakeField<float>("x", 0.0);
   //   hitModel->MakeField<float>("y", 0.0);
   //
   //   auto trackModel = RNTupleModel::Create();
   //   trackModel->MakeField<float>("energy", 0.0);
   //
   //   trackModel->MakeCollection("hits", std::move(hitModel));
   //   model->MakeCollection("tracks", std::move(trackModel));
   //
   auto f1 = RFieldBase::Create("missingE", "std::vector<float>").Unwrap();
   model->AddProjectedField(std::move(f1), [](const RFieldBase &) { return ""; });
   //   // No such parent field
   //   EXPECT_THROW(model->AddProjectedField(std::move(f1), "na", [](const RFieldBase &){return "";}),
   //                RException);
   //
   //   // Top level field clash
   //   auto f2 = RFieldBase::Create("met", "float").Unwrap();
   //   EXPECT_THROW(model->AddProjectedField(std::move(f2), "", [](const RFieldBase &){return "";}),
   //                RException);
   //
   //   // Sub field name clash
   //   auto f3 = RFieldBase::Create("x", "float").Unwrap();
   //   EXPECT_THROW(model->AddProjectedField(std::move(f3), "tracks.hits", [](const RFieldBase &){return "";}),
   //                RException);
   //
   //   auto f4 = RFieldBase::Create("missingE", "float").Unwrap();
   //   model->AddProjectedField(std::move(f4), "", [](const RFieldBase &){return "";});
   //
   //   // Projection name clash
   //   auto f5 = RFieldBase::Create("missingE", "float").Unwrap();
   //   EXPECT_THROW(model->AddProjectedField(std::move(f5), "", [](const RFieldBase &){return "";}),
   //                RException);
   //
   //   auto f6 = RFieldBase::Create("X_coord", "float").Unwrap();
   //   model->AddProjectedField(std::move(f6), "tracks.hits", [](const RFieldBase &){return "";});
   //
   //   // Projection name clash
   //   auto f7 = RFieldBase::Create("X_coord", "float").Unwrap();
   //   EXPECT_THROW(model->AddProjectedField(std::move(f7), "tracks.hits", [](const RFieldBase &){return "";}),
   //                RException);
}
