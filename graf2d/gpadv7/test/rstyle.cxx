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
   RAttrMargins fAttrMargins{this, "margins"}; ///<! margin attributes

public:
   RAttrLine  line{this, "line"};         ///<! line attributes
   RAttrFill  fill{this, "fill"};         ///<! fill attributes
   RAttrText  text{this, "text"};         ///<! text attributes

   CustomDrawable() : RDrawable("custom") {}

   const RAttrMargins &AttrMargins() const { return fAttrMargins; }
   RAttrMargins &AttrMargins() { return fAttrMargins; }
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

   EXPECT_DOUBLE_EQ(drawable.line.width, 2.f);

   EXPECT_EQ(drawable.fill.style, 5);

   EXPECT_DOUBLE_EQ(drawable.text.size, 3.);
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

   EXPECT_DOUBLE_EQ(drawable.line.width, 2.f);

   EXPECT_EQ(drawable.line.color, RColor::kRed);

   EXPECT_EQ(drawable.fill.style, 5);

   EXPECT_DOUBLE_EQ(drawable.text.size, 3.);
}


TEST(RStyleTest, TestMargins)
{
   auto style = RStyle::Parse(" custom { margins_all: 0.3; margins_left: 0.2; margins_right: 0.4; }");

   ASSERT_NE(style, nullptr);

   CustomDrawable drawable;
   drawable.UseStyle(style);

   auto &margins = drawable.AttrMargins();

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

      EXPECT_DOUBLE_EQ(drawable.line.width, 2.f);
   }

   // here style no longer exists
   EXPECT_DOUBLE_EQ(drawable.line.width, 1.f);
}

