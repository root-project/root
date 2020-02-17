// @(#)root/graf2d:$Id$
// Author: Sergey Linev <s.linev@gsi.de>, 2020-02-17


#include "gtest/gtest.h"

#include <ROOT/RStyle.hxx>

TEST(RStyleTest, ParseCss)
{
   auto style = std::make_shared<ROOT::Experimental::RStyle>();

   bool res1 = style->ParseString("csstype { attr1:value1; attr2:value2; }");
   EXPECT_EQ(res1, true);

   bool res2 = style->ParseString(".cssclass { attr3:value3; attr4:value4; }");
   EXPECT_EQ(res2, true);

   bool res3 = style->ParseString("#cssid { attr5:value5; attr6:value6; }");
   EXPECT_EQ(res3, true);

   if (res1 && res2 && res3) {

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
}


