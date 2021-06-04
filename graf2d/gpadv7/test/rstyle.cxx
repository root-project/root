// @(#)root/graf2d:$Id$
// Author: Sergey Linev <s.linev@gsi.de>, 2019-10-04


#include "gtest/gtest.h"

#include "ROOT/RStyle.hxx"

#include "ROOT/RDrawable.hxx"
#include "ROOT/RAttrText.hxx"
#include "ROOT/RAttrLine.hxx"
#include "ROOT/RAttrFill.hxx"
#include "ROOT/RAttrMargins.hxx"


using namespace ROOT::Experimental;

class CustomDrawable : public RDrawable {
   RAttrLine  fAttrLine{this, "line"};        ///<! line attributes
   RAttrFill  fAttrFill{this, "fill"};        ///<! fill attributes
   RAttrText  fAttrText{this, "text"};        ///<! text attributes
   RAttrMargins fAttrMargins{this, "margin"}; ///<! margin attributes

public:
   CustomDrawable() : RDrawable("custom") {}

   const RAttrLine &GetAttrLine() const { return fAttrLine; }
   CustomDrawable &SetAttrLine(const RAttrLine &attr) { fAttrLine = attr; return *this; }
   RAttrLine &AttrLine() { return fAttrLine; }

   const RAttrFill &GetAttrFill() const { return fAttrFill; }
   CustomDrawable &SetAttrFill(const RAttrFill &fill) { fAttrFill = fill; return *this; }
   RAttrFill &AttrFill() { return fAttrFill; }

   const RAttrText &GetAttrText() const { return fAttrText; }
   CustomDrawable &SetAttrText(const RAttrText &attr) { fAttrText = attr; return *this; }
   RAttrText &AttrText() { return fAttrText; }

   const RAttrMargins &GetMargins() const { return fAttrMargins; }
   CustomDrawable &SetMargins(const RAttrMargins &margins) { fAttrMargins = margins; return *this; }
   RAttrMargins &Margins() { return fAttrMargins; }

};


TEST(RStyleTest, CreateStyle)
{
   auto style = std::make_shared<RStyle>();

   style->AddBlock("custom").AddDouble("line_width", 2.);

   style->AddBlock("#customid").AddInt("fill_style", 5);

   style->AddBlock(".custom_class").AddDouble("text_size", 3.);

   CustomDrawable drawable;
   drawable.SetId("customid");
   drawable.SetCssClass("custom_class");

   drawable.UseStyle(style);

   EXPECT_DOUBLE_EQ(drawable.GetAttrLine().GetWidth(), 2.);

   EXPECT_EQ(drawable.GetAttrFill().GetStyle(), 5);

   EXPECT_DOUBLE_EQ(drawable.GetAttrText().GetSize(), 3.);
}


TEST(RStyleTest, CreateCss)
{
   auto style = RStyle::Parse(" custom { line_width: 2; line_color: red; }"
                              " #customid { fill_style: 5; }"
                              " .custom_class { text_size: 3; }");

   ASSERT_NE(style, nullptr);

   CustomDrawable drawable;
   drawable.SetId("customid");
   drawable.SetCssClass("custom_class");

   drawable.UseStyle(style);

   EXPECT_DOUBLE_EQ(drawable.GetAttrLine().GetWidth(), 2.);

   EXPECT_EQ(drawable.GetAttrLine().GetColor(), RColor::kRed);

   EXPECT_EQ(drawable.GetAttrFill().GetStyle(), 5);

   EXPECT_DOUBLE_EQ(drawable.GetAttrText().GetSize(), 3.);
}


TEST(RStyleTest, TestMargins)
{
   auto style = RStyle::Parse(" custom { margin_all: 0.3; margin_left: 0.2; margin_right: 0.4; }");

   ASSERT_NE(style, nullptr);

   CustomDrawable drawable;
   drawable.UseStyle(style);

   auto &margins = drawable.GetMargins();

   EXPECT_EQ(margins.GetLeft(), 0.2);
   EXPECT_EQ(margins.GetRight(), 0.4_normal);
   EXPECT_EQ(margins.GetAll(), 0.3_normal);
}


TEST(RStyleTest, LostStyle)
{
   CustomDrawable drawable;

   {
      auto style = std::make_shared<RStyle>();

      style->AddBlock("custom").AddDouble("line_width", 2.);

      // here weak_ptr will be set, therefore after style is deleted drawable will loose it
      drawable.UseStyle(style);

      EXPECT_DOUBLE_EQ(drawable.GetAttrLine().GetWidth(), 2.);
   }

   // here style no longer exists
   EXPECT_DOUBLE_EQ(drawable.GetAttrLine().GetWidth(), 1.);
}

