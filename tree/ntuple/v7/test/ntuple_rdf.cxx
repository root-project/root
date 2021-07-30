#include "ntuple_test.hxx"

TEST(RNTuple, RDF)
{
   FileRaii fileGuard("test_ntuple_rdf.root");

   auto modelWrite = RNTupleModel::Create();
   auto wrPt = modelWrite->MakeField<float>("pt", 42.0);
   auto wrEnergy = modelWrite->MakeField<float>("energy", 7.0);
   auto wrTag = modelWrite->MakeField<std::string>("tag", "xyz");
   auto wrJets = modelWrite->MakeField<std::vector<float>>("jets");
   wrJets->push_back(1.0);
   wrJets->push_back(2.0);
   auto wrNnlo = modelWrite->MakeField<std::vector<std::vector<float>>>("nnlo");
   wrNnlo->push_back(std::vector<float>());
   wrNnlo->push_back(std::vector<float>{1.0});
   wrNnlo->push_back(std::vector<float>{1.0, 2.0, 4.0, 8.0});
   auto wrKlass = modelWrite->MakeField<CustomStruct>("klass");
   wrKlass->s = "abc";
   wrKlass->v1.push_back(100.);
   wrKlass->v1.push_back(200.);
   wrKlass->v1.push_back(300.);

   {
      auto ntuple = RNTupleWriter::Recreate(std::move(modelWrite), "myNTuple", fileGuard.GetPath());
      ntuple->Fill();
   }

   ROOT::EnableImplicitMT();
   auto rdf = ROOT::Experimental::MakeNTupleDataFrame("myNTuple", fileGuard.GetPath());
   EXPECT_EQ(42.0, *rdf.Min("pt"));
   auto s = rdf.Take<std::string>("klass.s");
   EXPECT_EQ(1ull, s.GetValue().size());
   EXPECT_EQ(std::string("abc"), s.GetValue()[0]);
   EXPECT_EQ(2U, *rdf.Min("R_rdf_sizeof_jets"));
   EXPECT_EQ(3U, *rdf.Min("R_rdf_sizeof_klass.v1"));
}
