#include "gtest/gtest.h"

#include "ROOT/RDrawingAttr.hxx"

class Attr: public ROOT::Experimental::RDrawingAttrNodeBase {

};

class Opts: public ROOT::Experimental::RDrawingAttrTopEdgeBase {
public:
    Attr one{"one", *this};
    Attr two{"two", *this};
};

// Predef
TEST(OptsTest, ToString) {
   using namespace ROOT::Experimental;
   {
       Opts opts("top");

   }
   {
      RColor col{RColor::kBlue};
      col.SetAlpha(RColor::kTransparent);
      EXPECT_FLOAT_EQ(col.GetRed(), 0.);
      EXPECT_FLOAT_EQ(col.GetGreen(), 0.);
      EXPECT_FLOAT_EQ(col.GetBlue(), 1.);
      EXPECT_FLOAT_EQ(col.GetAlpha(), 0.);
   }
}
