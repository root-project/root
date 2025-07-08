#include <string>

#include <ROOT/RField.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleReader.hxx>

#include <TClassEdit.h>

#include <gtest/gtest.h>

#include "AtlasLikeDataVector.hxx"

TEST(RNTupleAtlasDataVector, Read)
{
   const std::string fullTypeName{"AtlasLikeDataVector<CustomStruct,DataModel_detail::NoBase>"};
   const std::string shortTypeName{"AtlasLikeDataVector<CustomStruct>"};
   // Make sure that TypeName() expands the optional template argument
   EXPECT_EQ(fullTypeName, ROOT::RField<AtlasLikeDataVector<CustomStruct>>::TypeName());

   // Make sure autoloading did not happen yet, so ROOT Meta also expands the optional template argument
   EXPECT_EQ(fullTypeName, ROOT::Internal::GetDemangledTypeName(typeid(AtlasLikeDataVector<CustomStruct>)));

   // By creating the RField, we autoload the dictionary. Subsequently, ROOT Meta normalizes the type name
   // taking into account that AtlasLikeDataVector inherits from KeepFirstTemplateArguments<1>
   EXPECT_EQ(shortTypeName, ROOT::RField<AtlasLikeDataVector<CustomStruct>>("f").GetTypeName());
   EXPECT_EQ(shortTypeName, ROOT::Internal::GetDemangledTypeName(typeid(AtlasLikeDataVector<CustomStruct>)));
   // Ensure that RField<T>::TypeName() is not changing depending on the loaded dictionaries
   EXPECT_EQ(fullTypeName, ROOT::RField<AtlasLikeDataVector<CustomStruct>>::TypeName());

   // Ensure that we can access the field by typeid, short name, and long name and the
   // type name checks will be fine with it
   auto reader = ROOT::RNTupleReader::Open("ntpl", "test_ntuple_datavector.root");
   AtlasLikeDataVector<CustomStruct> dummy;
   EXPECT_NO_THROW(reader->GetView("my_field", &dummy));
   EXPECT_NO_THROW(reader->GetView("my_field", &dummy, fullTypeName));
   EXPECT_NO_THROW(reader->GetView("my_field", &dummy, shortTypeName));
   EXPECT_NO_THROW(reader->GetView<AtlasLikeDataVector<CustomStruct>>("my_field"));
   EXPECT_NO_THROW(reader->GetModel().GetDefaultEntry().GetPtr<AtlasLikeDataVector<CustomStruct>>("my_field"));
}

int main(int argc, char **argv)
{
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}
