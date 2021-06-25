#include "gtest/gtest.h"

#include "ROOT/RLogger.hxx"
#include "ROOT/RAttrAggregation.hxx"
#include "ROOT/RAttrText.hxx"
#include "ROOT/RAttrFill.hxx"
#include "ROOT/RAttrLine.hxx"

using namespace ROOT::Experimental;

class CustomAttrs : public RAttrAggregation {
   RAttrLine    fAttrLine{this, "line"};    ///<! line attributes
   RAttrFill    fAttrFill{this, "fill"};    ///<! fill attributes
   RAttrText    fAttrText{this, "text"};    ///<! text attributes

protected:
   // required here while dictionary for CustomAttrs not created
   RAttrMap CollectDefaults() const override { return RAttrMap().AddDefaults(fAttrLine).AddDefaults(fAttrFill).AddDefaults(fAttrText); }

   R__ATTR_CLASS(CustomAttrs, "custom");

   const RAttrLine &AttrLine() const { return fAttrLine; }
   RAttrLine &AttrLine() { return fAttrLine; }

   const RAttrFill &AttrFill() const { return fAttrFill; }
   RAttrFill &AttrFill() { return fAttrFill; }

   const RAttrText &AttrText() const { return fAttrText; }
   RAttrText &AttrText() { return fAttrText; }

   double GetDirect(const std::string &name)
   {
      // CAUTION: name is not duplicated in RAttrValue
      RAttrValue<double> direct{this, name.c_str(), 0.};
      return direct.Get();
   }
};


TEST(RAttrTest, AttribStrings) {
   CustomAttrs attrs;

   attrs.AttrLine().SetWidth(42.);
   attrs.AttrText().SetSize(1.7);

   {
      auto val = attrs.GetDirect("line_width");
      EXPECT_FLOAT_EQ(val, 42.f);
   }

   {
      auto val = attrs.GetDirect("text_size");
      EXPECT_FLOAT_EQ(val, 1.7f);
   }
}

TEST(RAttrTest, AttribVals) {
   CustomAttrs attrs;

   attrs.AttrText().SetColor(RColor::kBlue);
   auto &line = attrs.AttrLine();
   line.SetWidth(42.);

   {
      // Value was set on this attr, not coming from style:
      EXPECT_FLOAT_EQ(attrs.AttrLine().GetWidth(), 42.f);
      EXPECT_FLOAT_EQ(line.GetWidth(), 42.f);
   }

   {
      // Value was set on this attr, not coming from style:
      EXPECT_EQ(attrs.AttrText().GetColor(), RColor::kBlue);
   }

}

TEST(RAttrTest, NullAttribCompare) {
   RAttrLine al1;
   RAttrLine al2;
   EXPECT_EQ(al1, al2);
   EXPECT_EQ(al2, al1);
}

TEST(RAttrTest, AttribEqual) {
   CustomAttrs attrs;

   auto &al1 = attrs.AttrLine();
   auto &al2 = attrs.AttrLine();
   EXPECT_EQ(al1, al2);
   EXPECT_EQ(al2, al1);

   al1.SetColor(RColor::kRed);

   EXPECT_EQ(al1, al2);
   EXPECT_EQ(al2, al1);
}

TEST(RAttrTest, AttribDiffer) {
   CustomAttrs attrs1;
   CustomAttrs attrs2;
   CustomAttrs attrs3;

   // RLogScopedVerbosity debugThis(GPadLog(), ELogLevel::kDebug);

   attrs1.AttrLine().SetWidth(7.);
   EXPECT_NE(attrs1, attrs2);
   EXPECT_NE(attrs2, attrs1);
   EXPECT_EQ(attrs2, attrs3);
   EXPECT_EQ(attrs3, attrs2);

   attrs2.AttrLine().SetColor(RColor::kRed);
   EXPECT_NE(attrs1, attrs2);
   EXPECT_NE(attrs2, attrs1);
   EXPECT_NE(attrs1, attrs3);
   EXPECT_NE(attrs3, attrs1);
   EXPECT_NE(attrs2, attrs3);
   EXPECT_NE(attrs3, attrs2);
}


TEST(RAttrTest, AttribAssign) {
   CustomAttrs attrs1;
   CustomAttrs attrs2;

   // deep copy - independent from origin
   auto attrLine1 = attrs1.AttrLine();
   auto attrLine2 = attrs2.AttrLine();

   EXPECT_EQ(attrLine2, attrLine1);
   EXPECT_EQ(attrLine1, attrLine2);

   attrLine1.SetWidth(42.);
   EXPECT_NE(attrLine2, attrLine1);

   attrLine2 = attrLine1;
   EXPECT_EQ(attrLine2, attrLine1);
   EXPECT_EQ(attrLine1, attrLine2);

   // But original attributes now differ
   EXPECT_NE(attrs1.AttrLine(), attrLine1);
   EXPECT_NE(attrs2.AttrLine(), attrLine2);

   EXPECT_FLOAT_EQ(attrLine1.GetWidth(), 42.);
   EXPECT_FLOAT_EQ(attrLine2.GetWidth(), 42.);
   // default width return 1
   EXPECT_FLOAT_EQ(attrs1.AttrLine().GetWidth(), 1.);
   EXPECT_FLOAT_EQ(attrs2.AttrLine().GetWidth(), 1.);

   // Are the two attributes disconnected?
   attrLine2.SetWidth(3.);
   EXPECT_EQ(attrs1.AttrLine(), attrs2.AttrLine());
   EXPECT_FLOAT_EQ(attrLine1.GetWidth(), 42.);
   EXPECT_FLOAT_EQ(attrLine2.GetWidth(), 3.);
   EXPECT_FLOAT_EQ(attrs1.AttrLine().GetWidth(), 1.);
   EXPECT_FLOAT_EQ(attrs2.AttrLine().GetWidth(), 1.);
}

TEST(RAttrTest, AttribValue) {

   RAttrValue<int> value1;

   EXPECT_EQ(value1.GetDefault(), 0);
   EXPECT_EQ(value1.Get(), 0);

   value1.Set(5);
   EXPECT_EQ(value1.Get(), 5);

   RAttrValue<int> value2;
   EXPECT_NE(value1, value2);
   EXPECT_NE(value2, value1);

   value2 = value1;
   EXPECT_EQ(value1, value2);
   EXPECT_EQ(value2, value1);
   EXPECT_EQ(value2.Get(), 5);

}

TEST(RAttrTest, EnumValue) {

   enum Style { kNone, kFirst, kSecond, kThird, kForth };

   RAttrValue<Style> value0{kFirst};
   EXPECT_EQ(value0.Get(), kFirst);
   EXPECT_EQ(value0.GetDefault(), kFirst);

   // value0.Set(3); // this is invalid syntax
   value0.Set(kThird);
   EXPECT_EQ(value0.Get(), kThird);
   EXPECT_EQ(value0.GetDefault(), kFirst);

   RAttrValue<Style> value1{kSecond};
   EXPECT_NE(value1, value0);

   value1 = value0;
   EXPECT_EQ(value1, value0);
   EXPECT_EQ(value1.GetDefault(), kSecond);
   value1.Clear();
   EXPECT_NE(value1, value0);
   EXPECT_EQ(value1.Get(), kSecond);

   RAttrValue<Style> value2 = value0;
   EXPECT_EQ(value2, value0);
   EXPECT_EQ(value2.Get(), kThird);
   EXPECT_EQ(value2.GetDefault(), kFirst);
}
