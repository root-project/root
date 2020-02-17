// @(#)root/graf2d:$Id$
// Author: Sergey Linev <s.linev@gsi.de>, 2020-02-17


#include "gtest/gtest.h"

#include <ROOT/RStyle.hxx>

TEST(RStyleTest, ParseCss)
{
   auto style = std::make_shared<ROOT::Experimental::RStyle>();

   bool res = style->ParseString(
         "/* Basic example of CSS file \n"
         "   shows how different blocks can be defined */\n"
         "csstype {\n"
         "  attr1: value1;\n"
         "  attr2: value2;\n"
         "}\n"
         "\n"
         "// this is rule for css class\n"
         ".cssclass {\n"
         "   attr3: value3;\n"
         "   attr4: value4;\n"
         "}\n"
         "\n"
         "// this is rule for some specific identifier\n"
         "#cssid {\n"
         "   attr5: value5;\n"
         "   attr6: value6;\n"
         "}\n"
         );

   ASSERT_EQ(res, true);

   auto field1 = style->Eval("attr1", "csstype");

   ASSERT_NE(field1, nullptr);
   EXPECT_EQ(field1->GetString(), "value1");

   auto field4 = style->Eval("attr4", ".cssclass");

   ASSERT_NE(field4, nullptr);
   EXPECT_EQ(field4->GetString(), "value4");

   auto field5 = style->Eval("attr5", "#cssid");

   ASSERT_NE(field5, nullptr);
   EXPECT_EQ(field5->GetString(), "value5");
}


