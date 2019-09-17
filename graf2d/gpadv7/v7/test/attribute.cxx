#include "gtest/gtest.h"

#include "ROOT/RAttrText.hxx"
#include "ROOT/RAttrBox.hxx"


using namespace ROOT::Experimental;

class CustomAttrs : public RAttrBase {
   RAttrBox fAttrBox{this, "box_"};
   RAttrText fAttrText{this, "text_"};

protected:
   const RAttrValues::Map_t &GetDefaults() const override
   {
      static auto dflts = RAttrValues::Map_t().AddDefaults(fAttrBox).AddDefaults(fAttrText);
      return dflts;
   }

public:

   using RAttrBase::RAttrBase;

   CustomAttrs(const CustomAttrs &src) : CustomAttrs() { src.CopyTo(*this); }
   CustomAttrs &operator=(const CustomAttrs &src) { Clear(); src.CopyTo(*this); return *this; }

   RAttrBox &AttrBox() { return fAttrBox; }
   const RAttrBox &AttrBox() const { return fAttrBox; }

   RAttrText &AttrText() { return fAttrText; }
   const RAttrText &AttrText() const { return fAttrText; }
};


TEST(OptsTest, AttribStrings) {
   CustomAttrs attrs;

   attrs.AttrBox().Border().SetWidth(42.);
   attrs.AttrText().SetSize(1.7);

   {
      double val = attrs.GetValue<double>("box_border_width");
      EXPECT_FLOAT_EQ(val, 42.f);
   }

   {
      float val = attrs.GetValue<double>("text_size");
      EXPECT_FLOAT_EQ(val, 1.7f);
   }
}

TEST(OptsTest, AttribVals) {
   CustomAttrs attrs;

   attrs.AttrText().SetColor(RColor::kBlue);
   auto &border = attrs.AttrBox().Border();
   border.SetWidth(42.);

   {
      // Value was set on this attr, not coming from style:
      EXPECT_FLOAT_EQ(attrs.AttrBox().Border().GetWidth(), 42.f);
      EXPECT_FLOAT_EQ(border.GetWidth(), 42.f);
   }

   {
      // Value was set on this attr, not coming from style:
      EXPECT_EQ(attrs.AttrText().Color(), RColor::kBlue);
   }

}

TEST(OptsTest, NullAttribCompare) {
   RAttrLine al1;
   RAttrLine al2;
   EXPECT_EQ(al1, al2);
   EXPECT_EQ(al2, al1);
}

TEST(OptsTest, AttribEqual) {
   CustomAttrs attrs;

   auto &al1 = attrs.AttrBox().Border();
   auto &al2 = attrs.AttrBox().Border();
   EXPECT_EQ(al1, al2);
   EXPECT_EQ(al2, al1);

   al1.SetColor(RColor::kRed);

   EXPECT_EQ(al1, al2);
   EXPECT_EQ(al2, al1);
}

TEST(OptsTest, AttribDiffer) {
   CustomAttrs attrs1;
   CustomAttrs attrs2;
   CustomAttrs attrs3;

   auto &al1 = attrs1.AttrBox().Border();
   auto &al2 = attrs2.AttrBox().Border();
   auto &al3 = attrs3.AttrBox().Border();

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
   CustomAttrs attrs1;
   CustomAttrs attrs2;

   // deep copy - independent from origin
   auto attrBox1 = attrs1.AttrBox();
   auto attrBox2 = attrs2.AttrBox();

   EXPECT_EQ(attrBox2, attrBox1);
   EXPECT_EQ(attrBox1, attrBox2);

   attrBox1.Border().SetWidth(42.);
   EXPECT_NE(attrBox2, attrBox1);

   attrBox2 = attrBox1;
   EXPECT_EQ(attrBox2, attrBox1);
   EXPECT_EQ(attrBox1, attrBox2);

   // But original attributes now differ
   EXPECT_NE(attrs1.AttrBox(), attrBox1);
   EXPECT_NE(attrs2.AttrBox(), attrBox2);

   EXPECT_FLOAT_EQ(attrBox1.Border().GetWidth(), 42.);
   EXPECT_FLOAT_EQ(attrBox2.Border().GetWidth(), 42.);
   // default width return 1
   EXPECT_FLOAT_EQ(attrs1.AttrBox().Border().GetWidth(), 1.);
   EXPECT_FLOAT_EQ(attrs2.AttrBox().Border().GetWidth(), 1.);

   // Are the two attributes disconnected?
   attrBox2.Border().SetWidth(3.);
   EXPECT_EQ(attrs1.AttrBox().Border(), attrs2.AttrBox().Border());
   EXPECT_FLOAT_EQ(attrBox1.Border().GetWidth(), 42.);
   EXPECT_FLOAT_EQ(attrBox2.Border().GetWidth(), 3.);
   EXPECT_FLOAT_EQ(attrs1.AttrBox().Border().GetWidth(), 1.);
   EXPECT_FLOAT_EQ(attrs2.AttrBox().Border().GetWidth(), 1.);
}
