#include "gtest/gtest.h"

#include "ROOT/RAttrText.hxx"
#include "ROOT/RAttrBox.hxx"
#include "ROOT/RDrawingOptsBase.hxx"


using namespace ROOT::Experimental;

class Opts: public RDrawingOptsBase, public RAttrBox {
public:
   Opts(): RDrawingOptsBase(), RAttrBox(FromOption, "box", *this) {}

   RAttrText Text() { return {FromOption, "text", *this}; }
};

TEST(OptsTest, AttribVsHolder) {
   RAttrText differentLifetimeText;
   RAttrBox differentLifetimeBox;

   // The weak holder pointer must be invalid.
   EXPECT_FALSE(differentLifetimeText.GetHolderPtr().lock());
   EXPECT_FALSE(differentLifetimeBox.GetHolderPtr().lock());

   {
      Opts opts;

      // The attribute's weak holder pointer must point to opts' Holder.
      EXPECT_EQ(opts.GetHolder(), opts.GetHolderPtr().lock());
      EXPECT_EQ(opts.GetHolder(), opts.Text().GetHolderPtr().lock());

      differentLifetimeText = opts.Text();
      differentLifetimeBox = opts;

      // The weak holder pointer must point to opts.
      EXPECT_TRUE(differentLifetimeText.GetHolderPtr().lock());
      EXPECT_TRUE(differentLifetimeBox.GetHolderPtr().lock());

   }
   // The weak holder pointer must point to opts.
   EXPECT_FALSE(differentLifetimeText.GetHolderPtr().lock());
   EXPECT_FALSE(differentLifetimeBox.GetHolderPtr().lock());

}

TEST(OptsTest, AttribStrings) {
   Opts opts;

   opts.Bottom().SetWidth(42.);
   opts.Text().SetSize(1.7);

   ASSERT_TRUE(opts.GetHolder());
   auto holder = opts.GetHolder();
   using Path = RDrawingAttrBase::Path;

   EXPECT_FALSE(holder->AtIf(Path{"DOES_NOT_EXIST"}));

   {
      ASSERT_TRUE(holder->AtIf(Path{"box.bottom.width"}));
      auto pVal = holder->AtIf(Path{"box.bottom.width"});
      float val = std::stof(*pVal);
      ASSERT_FLOAT_EQ(val, 42.f);
   }

   {
      ASSERT_TRUE(holder->AtIf(Path{"text.size"}));
      auto pVal = holder->AtIf(Path{"text.size"});
      float val = std::stof(*pVal);
      ASSERT_FLOAT_EQ(val, 1.7f);
   }
}

TEST(OptsTest, AttribVals) {
   Opts opts;

   opts.Text().SetColor(RColor::kBlue);
   opts.Bottom().SetWidth(42.);

   ASSERT_TRUE(opts.GetHolder());
   auto holder = opts.GetHolder();

   {
      // Value was set on this attr, not coming from style:
      ASSERT_FALSE(opts.Bottom().IsFromStyle("width"));
      ASSERT_FLOAT_EQ(opts.Bottom().GetWidth(), 42.f);
   }

   {
      // Value was set on this attr, not coming from style:
      ASSERT_FALSE(opts.Text().IsFromStyle("color"));
      ASSERT_EQ(opts.Text().GetColor(), RColor::kBlue);
   }

}
