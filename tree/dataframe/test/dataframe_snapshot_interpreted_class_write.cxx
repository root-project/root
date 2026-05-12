#include <ROOT/RDataFrame.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RFieldBase.hxx>
#include <ROOT/RNTupleWriter.hxx>
#include <ROOT/RNTupleReader.hxx>

#include <TInterpreter.h>
#include <TFile.h>
#include <TTree.h>

#include <memory>

#include <gtest/gtest.h>

// The files are intentionally not deleted
class RDFSnapshotInterpretedClassWrite : public ::testing::Test {
protected:
   inline constexpr static auto fgRNTupleFile{"RDFSnapshotInterpretedClassWriteRNTuple.root"};
   inline constexpr static auto fgRNTupleName{"RDFSnapshotInterpretedClassWriteRNTuple"};
   inline constexpr static auto fgTTreeFile{"RDFSnapshotInterpretedClassWriteTTree.root"};
   inline constexpr static auto fgTTreeName{"RDFSnapshotInterpretedClassWriteTTree"};

   static void WriteTTree(const char *datasetname, const char *filename)
   {
      auto f = std::make_unique<TFile>(filename, "recreate");
      auto t = std::make_unique<TTree>(datasetname, datasetname);

      gInterpreter->ProcessLine("RDFSnapshotInterpretedClassWriteClass myEvt{};");
      std::unique_ptr<TInterpreterValue> v = gInterpreter->MakeInterpreterValue();
      gInterpreter->Evaluate("myEvt", *v);
      void *voidPtr = v->GetAsPointer();
      t->Branch("evt", "RDFSnapshotInterpretedClassWriteClass", &voidPtr);
      t->Fill();
      f->Write();
   }

   static void WriteRNTuple(const char *datasetname, const char *filename)
   {
      auto model = ROOT::RNTupleModel::Create();
      model->AddField(ROOT::RFieldBase::Create("evt", "RDFSnapshotInterpretedClassWriteClass").Unwrap());
      auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), datasetname, filename);
      writer->Fill();
   }

   static void SetUpTestSuite()
   {
      gInterpreter->Declare(R"(
        struct RDFSnapshotInterpretedClassWriteClass{
            int evtId{42};
            std::vector<int> charge{1,-1,1};
            std::vector<float> pt{5,10,15};
            std::vector<float> eta{0, 1, 2};
        };
        )");

      WriteRNTuple(fgRNTupleName, fgRNTupleFile);

      WriteTTree(fgTTreeName, fgTTreeFile);
   }
};

TEST_F(RDFSnapshotInterpretedClassWrite, ReadTTree)
{
   ROOT::RDataFrame df{fgTTreeName, fgTTreeFile};

   auto s_evtId = df.Sum("evt.evtId");
   auto s_charge = df.Sum("evt.charge");
   auto s_pt = df.Sum("evt.pt");
   auto s_eta = df.Sum("evt.eta");

   EXPECT_EQ(*s_evtId, 42);
   EXPECT_EQ(*s_charge, 1);
   EXPECT_FLOAT_EQ(*s_pt, 30);
   EXPECT_FLOAT_EQ(*s_eta, 3);
}

TEST_F(RDFSnapshotInterpretedClassWrite, ReadRNTuple)
{
   ROOT::RDataFrame df{fgRNTupleName, fgRNTupleFile};
   auto s_evtId = df.Sum("evt.evtId");
   auto s_charge = df.Sum("evt.charge");
   auto s_pt = df.Sum("evt.pt");
   auto s_eta = df.Sum("evt.eta");

   EXPECT_EQ(*s_evtId, 42);
   EXPECT_EQ(*s_charge, 1);
   EXPECT_FLOAT_EQ(*s_pt, 30);
   EXPECT_FLOAT_EQ(*s_eta, 3);
}
