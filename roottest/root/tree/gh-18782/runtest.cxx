#include <TFile.h>
#include <TTree.h>

#include <vector>

#include "MyParticle.hxx"

#include <gtest/gtest.h>

struct MyParticleReproducer : public ::testing::Test {
   constexpr static auto fFileName{"particles.root"};
   constexpr static auto fTreeName{"Particles"};
   static void SetUpTestSuite()
   {
      auto file = std::make_unique<TFile>(fFileName, "RECREATE");
      auto tree = std::make_unique<TTree>(fTreeName, fTreeName);

      std::vector<Derived> objects(1);

      // Default splitlevel=99
      tree->Branch("objects", &objects);
      tree->Fill();

      file->Write();
   }
   static void TearDownTestSuite() { std::remove(fFileName); }
};

TEST_F(MyParticleReproducer, ClassHierarchy)
{
   auto file = std::make_unique<TFile>(fFileName);
   auto *tree = file->Get<TTree>(fTreeName);
   auto objectsOwner = std::make_unique<std::vector<Derived>>();
   auto *objectsPtr = objectsOwner.get();
   tree->SetBranchAddress("objects", &objectsPtr);

   tree->GetEntry(0);

   const auto &objects = *objectsPtr;
   ASSERT_EQ(objects.size(), 1);

   const auto &object = objects[0];
   EXPECT_EQ(object.fSplittableBaseInt, 101);
   EXPECT_FLOAT_EQ(object.fSplittableBaseFloat, 102);
   EXPECT_FLOAT_EQ(object.fUnsplittableBaseFloat, 201);
   EXPECT_EQ(object.fDerivedInt, 301);
}

int main(int argc, char **argv)
{
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}
