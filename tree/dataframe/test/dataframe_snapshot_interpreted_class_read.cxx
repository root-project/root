#include <ROOT/RDataFrame.hxx>
#include <ROOT/TestSupport.hxx>

#include <gtest/gtest.h>

class RDFSnapshotInterpretedClassRead : public ::testing::Test {
protected:
   inline constexpr static auto fgRNTupleFile{"RDFSnapshotInterpretedClassWriteRNTuple.root"};
   inline constexpr static auto fgRNTupleName{"RDFSnapshotInterpretedClassWriteRNTuple"};
   inline constexpr static auto fgTTreeFile{"RDFSnapshotInterpretedClassWriteTTree.root"};
   inline constexpr static auto fgTTreeName{"RDFSnapshotInterpretedClassWriteTTree"};

   static void TearDownTestSuite()
   {
      std::remove(fgRNTupleFile);
      std::remove(fgTTreeFile);
   }
};

TEST_F(RDFSnapshotInterpretedClassRead, ReadTTree)
{
   ROOT::TestSupport::CheckDiagsRAII diags;
   diags.requiredDiag(kWarning, "TClass::Init",
                      "no dictionary for class RDFSnapshotInterpretedClassWriteClass is available");
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

TEST_F(RDFSnapshotInterpretedClassRead, ReadRNTuple)
{
   // We would normally also get the TClass::Init warning for the RNTuple case, but since we read the TTree first,
   // the TClass is already initialized and we don't get the warning again.
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
