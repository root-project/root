#include "gtest/gtest.h"

#include "ROOT/TText.hxx"

// Predef
TEST(TextTest, Predef) {

   using namespace ROOT;

   auto text = std::make_shared<Experimental::TText>(.5,.8, "Hello World");
   text->GetOptions().SetTextSize(40);

   EXPECT_FLOAT_EQ(text->GetX(), .5);
   EXPECT_FLOAT_EQ(text->GetY(), .8);
   EXPECT_EQ(40, text->GetSize());
}
