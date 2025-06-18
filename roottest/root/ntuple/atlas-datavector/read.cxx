#include <string>

#include <ROOT/RField.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleReader.hxx>

#include <gtest/gtest.h>

#include "AtlasLikeDataVector.hxx"

TEST(RNTupleAtlasDataVector, Read)
{
   const auto typeNameBefore = ROOT::RField<AtlasLikeDataVector<CustomStruct>>::TypeName();
   const std::string expectedBefore{"AtlasLikeDataVector<CustomStruct,DataModel_detail::NoBase>"};
   // Make sure no autoloading happened yet
   ASSERT_EQ(typeNameBefore, expectedBefore);

   auto reader = ROOT::RNTupleReader::Open("ntpl", "test_ntuple_datavector.root");
   const auto &entry = reader->GetModel().GetDefaultEntry();
   // The following call should not throw an exception
   ASSERT_NO_THROW(entry.GetPtr<AtlasLikeDataVector<CustomStruct>>("my_field"));

   const auto typeNameAfter = ROOT::RField<AtlasLikeDataVector<CustomStruct>>::TypeName();
   const std::string expectedAfter{"AtlasLikeDataVector<CustomStruct>"};
   // Make sure autoloading happened and the rule to suppress the second template argument kicked in
   ASSERT_EQ(typeNameAfter, expectedAfter);
}

int main(int argc, char **argv)
{
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}
