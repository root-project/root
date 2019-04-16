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

/* TODO: assignment to stand-alone attrs
      differentLifetimeText = opts.Text();
      differentLifetimeBox = opts;

      // The weak holder pointer must point to opts.
      EXPECT_TRUE(differentLifetimeText.GetHolderPtr().lock());
      EXPECT_TRUE(differentLifetimeBox.GetHolderPtr().lock());
*/

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
      EXPECT_TRUE(holder->AtIf(Path{"box.bottom.width"}));
      auto pVal = holder->AtIf(Path{"box.bottom.width"});
      float val = std::stof(*pVal);
      EXPECT_FLOAT_EQ(val, 42.f);
   }

   {
      ASSERT_TRUE(holder->AtIf(Path{"text.size"}));
      auto pVal = holder->AtIf(Path{"text.size"});
      float val = std::stof(*pVal);
      EXPECT_FLOAT_EQ(val, 1.7f);
   }
}

TEST(OptsTest, AttribVals) {
   Opts opts;

   opts.Text().SetColor(RColor::kBlue);
   auto bottom = opts.Bottom();
   opts.Bottom().SetWidth(42.);

   ASSERT_TRUE(opts.GetHolder());
   auto holder = opts.GetHolder();

   {
      // Value was set on this attr, not coming from style:
      EXPECT_FALSE(opts.Bottom().IsFromStyle("width"));
      EXPECT_FLOAT_EQ(opts.Bottom().GetWidth(), 42.f);
      EXPECT_FLOAT_EQ(bottom.GetWidth(), 42.f);
   }

   {
      // Value was set on this attr, not coming from style:
      EXPECT_FALSE(opts.Text().IsFromStyle("color"));
      EXPECT_EQ(opts.Text().GetColor(), RColor::kBlue);
   }

}

TEST(OptsTest, NullAttribCompare) {
   RAttrLine al1;
   RAttrLine al2;
   EXPECT_EQ(al1, al2);
   EXPECT_EQ(al2, al1);
}

TEST(OptsTest, AttribEqual) {
   Opts opts;
   auto al1 = opts.Left();
   auto al2 = opts.Left();
   EXPECT_EQ(al1, al2);
   EXPECT_EQ(al2, al1);

   al1.SetColor(RColor::kRed);
   EXPECT_EQ(al1, al2);
   EXPECT_EQ(al2, al1);
}

TEST(OptsTest, AttribDiffer) {
   Opts opts1;
   Opts opts2;
   Opts opts3;
   auto al1 = opts1.Left();
   auto al2 = opts2.Left();
   auto al3 = opts3.Left();

   al1.SetWidth(7.);
   EXPECT_NE(al1, al2);
   EXPECT_NE(al2, al1);
   EXPECT_NE(al1, al3);
   EXPECT_EQ(al2, al3);
   EXPECT_EQ(al3, al2);

   al2.SetColor(RColor::kRed);
   EXPECT_NE(al1, al2);
   EXPECT_NE(al2, al1);
   EXPECT_NE(al1, al3);
   EXPECT_NE(al2, al3);
   EXPECT_NE(al3, al2);
}


TEST(OptsTest, AttribAssign) {
   Opts opts1;
   Opts opts2;

   auto attrBox1 = opts1.Border();
   auto attrBox2 = opts2.Border();

   EXPECT_EQ(attrBox2, attrBox1);
   EXPECT_EQ(attrBox1, attrBox2);

   attrBox1.SetWidth(42.);
   EXPECT_NE(attrBox2, attrBox1);

   attrBox2 = attrBox1;
   EXPECT_EQ(attrBox2, attrBox1);
   EXPECT_EQ(attrBox1, attrBox2);

   // But make sure that the attributes still use their options!
   EXPECT_EQ(opts1.Border(), attrBox1);
   EXPECT_EQ(opts2.Border(), attrBox2);

   EXPECT_FLOAT_EQ(attrBox1.GetWidth(), 42.);
   EXPECT_FLOAT_EQ(attrBox2.GetWidth(), 42.);
   EXPECT_FLOAT_EQ(opts1.Border().GetWidth(), 42.);
   EXPECT_FLOAT_EQ(opts2.Border().GetWidth(), 42.);

   // Are the two attributes disconnected?
   attrBox2.SetWidth(3.);
   EXPECT_NE(opts1.Border(), opts2.Border());
   EXPECT_FLOAT_EQ(attrBox1.GetWidth(), 42.);
   EXPECT_FLOAT_EQ(attrBox2.GetWidth(), 3.);
   EXPECT_FLOAT_EQ(opts1.Border().GetWidth(), 42.);
   EXPECT_FLOAT_EQ(opts2.Border().GetWidth(), 3.);
}
