#include <ROOT/RField.hxx>
#include <ROOT/RFieldValue.hxx>

#include "gtest/gtest.h"

#include <memory>

#include "CustomStruct.hxx"

using RFieldBase = ROOT::Experimental::Detail::RFieldBase;
using RFieldValue = ROOT::Experimental::Detail::RFieldValue;

TEST(RNTuple, TypeName)
{
   EXPECT_STREQ("float", ROOT::Experimental::RField<float>::TypeName().c_str());
   EXPECT_STREQ("std::vector<std::string>", ROOT::Experimental::RField<std::vector<std::string>>::TypeName().c_str());
#if defined(_MSC_VER) && !defined(R__ENABLE_BROKEN_WIN_TESTS)
   EXPECT_STREQ("struct CustomStruct", ROOT::Experimental::RField<CustomStruct>::TypeName().c_str());
#else
   EXPECT_STREQ("CustomStruct", ROOT::Experimental::RField<CustomStruct>::TypeName().c_str());
#endif
}


TEST(RNTuple, CreateField)
{
   auto field = std::unique_ptr<RFieldBase>(RFieldBase::Create("test", "vector<unsigned int>"));
   EXPECT_STREQ("std::vector<std::uint32_t>", field->GetType().c_str());
   auto value = field->GenerateValue();
   field->DestroyValue(value);
}
