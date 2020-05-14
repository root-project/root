#include <ROOT/RField.hxx>

#include "gtest/gtest.h"

#include "CustomStruct.hxx"

using RFieldBase = ROOT::Experimental::Detail::RFieldBase;

TEST(RNTuple, TypeName)
{
   EXPECT_STREQ("float", ROOT::Experimental::RField<float>::TypeName().c_str());
   EXPECT_STREQ("std::vector<std::string>", ROOT::Experimental::RField<std::vector<std::string>>::TypeName().c_str());
   EXPECT_STREQ("CustomStruct", ROOT::Experimental::RField<CustomStruct>::TypeName().c_str());
}


TEST(RNTuple, CreateField)
{
   auto field = RFieldBase::Create("test", "vector<unsigned int>");
   EXPECT_STREQ("std::vector<std::uint32_t>", field->GetType().c_str());
}
