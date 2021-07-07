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

public:
   RAttrLine  line{this, "line"};         ///<! line attributes
   RAttrFill  fill{this, "fill"};         ///<! fill attributes
   RAttrText  text{this, "text"};         ///<! text attributes
   RAttrMargins margins{this, "margins"}; ///<! margins attributes

   CustomDrawable() : RDrawable("custom") {}
};


TEST(RStyleTest, CreateStyle)
{
   auto style = std::make_shared<RStyle>();

   style->AddBlock("custom").AddDouble("line_width", 2.);

   style->AddBlock("#customid").AddInt("fill_style", RAttrFill::k3005);

   style->AddBlock(".custom_class").AddDouble("text_size", 3.);

   CustomDrawable drawable;
   drawable.SetId("customid");
   drawable.SetCssClass("custom_class");

   drawable.UseStyle(style);

   EXPECT_DOUBLE_EQ(drawable.line.width, 2.f);

   EXPECT_EQ(drawable.fill.style, RAttrFill::k3005);

   EXPECT_DOUBLE_EQ(drawable.text.size, 3.);
}


TEST(RStyleTest, CreateCss)
{
   auto style = RStyle::Parse(" custom { line_width: 2; line_color: red; }"
                              " #customid { fill_style: 3005; }"
                              " .custom_class { text_size: 3; }");

   ASSERT_NE(style, nullptr);

   CustomDrawable drawable;
   drawable.SetId("customid");
   drawable.SetCssClass("custom_class");

   drawable.UseStyle(style);

   EXPECT_DOUBLE_EQ(drawable.line.width, 2.f);

   EXPECT_EQ(drawable.line.color, RColor::kRed);

   EXPECT_EQ(drawable.fill.style, RAttrFill::k3005);

   EXPECT_DOUBLE_EQ(drawable.text.size, 3.);
}

TEST(RStyleTest, CaseInsensitive)
{
   auto style = RStyle::Parse(" custom { line_Width: 2; Line_coloR: red; }"
                              " #customID { fill_style: 3005; }"
                              " .custom_Cclass { text_size: 3; }");

   ASSERT_NE(style, nullptr);

   CustomDrawable drawable;
   drawable.SetId("customid");
   drawable.SetCssClass("custom_class");

   drawable.UseStyle(style);

   // attribute names should be case insensetive
   EXPECT_DOUBLE_EQ(drawable.line.width, 2.f);

   EXPECT_EQ(drawable.line.color, RColor::kRed);

   // but id should have exact match
   EXPECT_NE(drawable.fill.style, RAttrFill::k3005);

   // and class name should have exact match
   EXPECT_NE(drawable.text.size, 3.);
}


TEST(RStyleTest, TestMargins)
{
   auto style = RStyle::Parse(" custom { margins_top: 0.3; margins_left: 0.2; margins_right: 0.4; }");

   ASSERT_NE(style, nullptr);

   CustomDrawable drawable;
   drawable.UseStyle(style);

   EXPECT_EQ(drawable.margins.left, 0.2);
   EXPECT_EQ(drawable.margins.right, 0.4_normal);
   EXPECT_EQ(drawable.margins.top, 0.3_normal);
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

