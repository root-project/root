#include "gtest/gtest.h"


#include "ROOT/RAttrLine.hxx"

#include "ROOT/RAttrText.hxx"


using namespace ROOT::Experimental;

class CustomAttrs : public RAttrBase {
   RAttrLine fAttrLine{"line", this};
   RAttrText fAttrText{"text", this};

protected:

   // provide method here while dictionary is not generated
   RAttrMap CollectDefaults() const override { return RAttrMap().AddDefaults(fAttrLine).AddDefaults(fAttrText); }

   R__ATTR_CLASS(CustomAttrs, "custom");

   const RAttrLine &AttrLine() const { return fAttrLine; }
   RAttrLine &AttrLine() { return fAttrLine; }

   const RAttrText &AttrText() const { return fAttrText; }
   RAttrText &AttrText() { return fAttrText; }

   double GetDirect(const std::string &name) const { return GetValue<double>(name); }
};


TEST(OptsTest, AttribStrings) {
   CustomAttrs attrs;

   attrs.AttrLine().SetLineWidth(42.);
   attrs.AttrText().SetTextSize(1.7);

   {
      auto val = attrs.GetDirect("line_width");
      EXPECT_FLOAT_EQ(val, 42.f);
   }

   {
      auto val = attrs.GetDirect("text_size");
      EXPECT_FLOAT_EQ(val, 1.7f);
   }
}

TEST(OptsTest, AttribVals) {
   CustomAttrs attrs;

   attrs.AttrText().SetTextColor(RColor::kBlue);
   auto &line = attrs.AttrLine();
   line.SetLineWidth(42.);

   {
      // Value was set on this attr, not coming from style:
      EXPECT_FLOAT_EQ(attrs.AttrLine().GetLineWidth(), 42.f);
      EXPECT_FLOAT_EQ(line.GetLineWidth(), 42.f);
   }

   {
      // Value was set on this attr, not coming from style:
      EXPECT_EQ(attrs.AttrText().GetTextColor(), RColor::kBlue);
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

   auto &al1 = attrs.AttrLine();
   auto &al2 = attrs.AttrLine();
   EXPECT_EQ(al1, al2);
   EXPECT_EQ(al2, al1);

   al1.SetLineColor(RColor::kRed);

   EXPECT_EQ(al1, al2);
   EXPECT_EQ(al2, al1);
}

TEST(OptsTest, AttribDiffer) {
   CustomAttrs attrs1;
   CustomAttrs attrs2;
   CustomAttrs attrs3;

   attrs1.AttrLine().SetLineWidth(7.);
   EXPECT_NE(attrs1, attrs2);
   EXPECT_NE(attrs2, attrs1);
   EXPECT_EQ(attrs2, attrs3);
   EXPECT_EQ(attrs3, attrs2);

   attrs2.AttrLine().SetLineColor(RColor::kRed);
   EXPECT_NE(attrs1, attrs2);
   EXPECT_NE(attrs2, attrs1);
   EXPECT_NE(attrs1, attrs3);
   EXPECT_NE(attrs3, attrs1);
   EXPECT_NE(attrs2, attrs3);
   EXPECT_NE(attrs3, attrs2);
}


TEST(OptsTest, AttribAssign) {
   CustomAttrs attrs1;
   CustomAttrs attrs2;

   // deep copy - independent from origin
   auto attrLine1 = attrs1.AttrLine();
   auto attrLine2 = attrs2.AttrLine();

   EXPECT_EQ(attrLine2, attrLine1);
   EXPECT_EQ(attrLine1, attrLine2);

   attrLine1.SetLineWidth(42.);
   EXPECT_NE(attrLine2, attrLine1);

   attrLine2 = attrLine1;
   EXPECT_EQ(attrLine2, attrLine1);
   EXPECT_EQ(attrLine1, attrLine2);

   // But original attributes now differ
   EXPECT_NE(attrs1.AttrLine(), attrLine1);
   EXPECT_NE(attrs2.AttrLine(), attrLine2);

   EXPECT_FLOAT_EQ(attrLine1.GetLineWidth(), 42.);
   EXPECT_FLOAT_EQ(attrLine2.GetLineWidth(), 42.);
   // default width return 1
   EXPECT_FLOAT_EQ(attrs1.AttrLine().GetLineWidth(), 1.);
   EXPECT_FLOAT_EQ(attrs2.AttrLine().GetLineWidth(), 1.);

   // Are the two attributes disconnected?
   attrLine2.SetLineWidth(3.);
   EXPECT_EQ(attrs1.AttrLine(), attrs2.AttrLine());
   EXPECT_FLOAT_EQ(attrLine1.GetLineWidth(), 42.);
   EXPECT_FLOAT_EQ(attrLine2.GetLineWidth(), 3.);
   EXPECT_FLOAT_EQ(attrs1.AttrLine().GetLineWidth(), 1.);
   EXPECT_FLOAT_EQ(attrs2.AttrLine().GetLineWidth(), 1.);
}

TEST(OptsTest, AttrLineCopy) {
   CustomAttrs attrs1;
   CustomAttrs attrs2;

   // deep copy - independent from origin
   attrs1.AttrLine().SetLineWidth(10).SetLineStyle(5).SetLineColor(RColor::kRed);

   EXPECT_NE(attrs1.AttrLine(), attrs2.AttrLine());

   attrs2.AttrLine() = attrs1.AttrLine();

   EXPECT_EQ(attrs1.AttrLine(), attrs2.AttrLine());

   EXPECT_FLOAT_EQ(attrs2.AttrLine().GetLineWidth(), 10);
   EXPECT_EQ(attrs2.AttrLine().GetLineStyle(), 5);
   EXPECT_EQ(attrs2.AttrLine().GetLineColor(), RColor::kRed);
}
