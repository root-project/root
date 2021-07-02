#include "gtest/gtest.h"

#include "ROOT/RLogger.hxx"
#include "ROOT/RAttrAggregation.hxx"
#include "ROOT/RAttrText.hxx"
#include "ROOT/RAttrFill.hxx"
#include "ROOT/RAttrLine.hxx"

using namespace ROOT::Experimental;

class CustomAttrs : public RAttrAggregation {

protected:
   // required here while dictionary for CustomAttrs not created
   RAttrMap CollectDefaults() const override { return RAttrMap().AddDefaults(line).AddDefaults(fill).AddDefaults(text); }

   R__ATTR_CLASS(CustomAttrs, "custom");

public:

   RAttrLine    line{this, "line"};    ///<! line attributes
   RAttrFill    fill{this, "fill"};    ///<! fill attributes
   RAttrText    text{this, "text"};    ///<! text attributes

   double GetDirect(const std::string &name)
   {
      // CAUTION: name is not duplicated in RAttrValue
      RAttrValue<double> direct{this, name.c_str(), 0.};
      return direct.Get();
   }
};


TEST(RAttrTest, AttribDirect) {
   CustomAttrs attrs;

   attrs.line.width = 42.;
   attrs.text.size = 1.7;

   {
      auto val = attrs.GetDirect("line_width");
      EXPECT_DOUBLE_EQ(val, 42.);
   }

   {
      auto val = attrs.GetDirect("text_size");
      EXPECT_DOUBLE_EQ(val, 1.7);
   }
}

TEST(RAttrTest, AttribVals) {
   CustomAttrs attrs;

   attrs.text.color = RColor::kBlue;
   attrs.line.width = 42.f;

   EXPECT_DOUBLE_EQ(attrs.line.width, 42.f);
   EXPECT_EQ(attrs.text.color, RColor::kBlue);

}

TEST(RAttrTest, NullAttribCompare) {
   RAttrLine al1;
   RAttrLine al2;
   EXPECT_EQ(al1, al2);
   EXPECT_EQ(al2, al1);
}

TEST(RAttrTest, AttribEqual) {
   CustomAttrs attrs;

   auto &al1 = attrs.line;
   auto &al2 = attrs.line;
   EXPECT_EQ(al1, al2);
   EXPECT_EQ(al2, al1);

   al1.color = RColor::kRed;

   EXPECT_EQ(al1, al2);
   EXPECT_EQ(al2, al1);
}

TEST(RAttrTest, AttribDiffer) {
   CustomAttrs attrs1;
   CustomAttrs attrs2;
   CustomAttrs attrs3;

   // RLogScopedVerbosity debugThis(GPadLog(), ELogLevel::kDebug);

   attrs1.line.width = 7.f;
   EXPECT_NE(attrs1, attrs2);
   EXPECT_NE(attrs2, attrs1);
   EXPECT_EQ(attrs2, attrs3);
   EXPECT_EQ(attrs3, attrs2);

   attrs2.line.color = RColor::kRed;
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
   auto attrLine1 = attrs1.line;
   auto attrLine2 = attrs2.line;

   EXPECT_EQ(attrLine2, attrLine1);
   EXPECT_EQ(attrLine1, attrLine2);

   attrLine1.width = 42.;
   EXPECT_NE(attrLine2, attrLine1);

   attrLine2 = attrLine1;
   EXPECT_EQ(attrLine2, attrLine1);
   EXPECT_EQ(attrLine1, attrLine2);

   // But original attributes now differ
   EXPECT_NE(attrs1.line, attrLine1);
   EXPECT_NE(attrs2.line, attrLine2);

   EXPECT_DOUBLE_EQ(attrLine1.width, 42.);
   EXPECT_DOUBLE_EQ(attrLine2.width, 42.);
   // default width return 1
   EXPECT_DOUBLE_EQ(attrs1.line.width, 1.);
   EXPECT_DOUBLE_EQ(attrs2.line.width, 1.);

   // Are the two attributes disconnected?
   attrLine2.width = 3.;
   EXPECT_EQ(attrs1.line, attrs2.line);
   EXPECT_DOUBLE_EQ(attrLine1.width, 42.);
   EXPECT_DOUBLE_EQ(attrLine2.width, 3.);
   EXPECT_DOUBLE_EQ(attrs1.line.width, 1.);
   EXPECT_DOUBLE_EQ(attrs2.line.width, 1.);
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
