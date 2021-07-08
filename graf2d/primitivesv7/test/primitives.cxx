#include "gtest/gtest.h"

#include "ROOT/RBox.hxx"
#include "ROOT/RLine.hxx"
#include "ROOT/RMarker.hxx"
#include "ROOT/RText.hxx"
#include "ROOT/RPaveText.hxx"
#include "ROOT/RLegend.hxx"
#include "ROOT/RCanvas.hxx"

using namespace ROOT::Experimental;

// Test RBox API
TEST(Primitives, RBox)
{
   RCanvas canv;
   auto box = canv.Draw<RBox>(RPadPos(0.1_normal, 0.3_normal), RPadPos(0.3_normal,0.6_normal));

   box->border.color = RColor::kRed;
   box->border.width = 5.;
   box->border.style = RAttrLine::kDotted;
   box->fill.color = RColor::kBlue;
   box->fill.style = RAttrFill::k3006;

   EXPECT_EQ(canv.NumPrimitives(), 1u);

   EXPECT_EQ(box->border.color, RColor::kRed);
   EXPECT_DOUBLE_EQ(box->border.width, 5.);
   EXPECT_EQ(box->border.style, RAttrLine::kDotted);

   EXPECT_EQ(box->fill.color, RColor::kBlue);
   EXPECT_EQ(box->fill.style, RAttrFill::k3006);
}

// Test RLine API
TEST(Primitives, RLine)
{
   RCanvas canv;
   auto line = canv.Draw<RLine>(RPadPos(0.1_normal, 0.1_normal), RPadPos(0.9_normal,0.9_normal));

   line->line.color = RColor::kRed;
   line->line.width = 5.;
   line->line.style = RAttrLine::kDashDotted;
   line->onFrame = true;
   line->clipping = false;

   EXPECT_EQ(canv.NumPrimitives(), 1u);

   EXPECT_EQ(line->line.color, RColor::kRed);
   EXPECT_DOUBLE_EQ(line->line.width, 5.);
   EXPECT_EQ(line->line.style, RAttrLine::kDashDotted);
   EXPECT_EQ(line->onFrame, true);
   EXPECT_EQ(line->clipping, false);
}

// Test RMarker API
TEST(Primitives, RMarker)
{
   RCanvas canv;
   auto marker = canv.Draw<RMarker>(RPadPos(0.5_normal, 0.5_normal));

   marker->marker = RAttrMarker(RColor::kGreen, 2.5, RAttrMarker::kStar);

   EXPECT_EQ(canv.NumPrimitives(), 1u);

   EXPECT_EQ(marker->marker.color, RColor::kGreen);
   EXPECT_DOUBLE_EQ(marker->marker.size, 2.5);
   EXPECT_EQ(marker->marker.style, RAttrMarker::kStar);
}

// Test RText API
TEST(Primitives, RText)
{
   RCanvas canv;

   auto text = canv.Draw<RText>(RPadPos(0.5_normal, 0.5_normal), "Hello World");

   text->text.color = RColor::kBlack;
   text->text.size = 12.5;
   text->text.angle = 90.;
   text->text.align = RAttrText::kLeftTop;
   text->text.font.family = "Arial";

   EXPECT_EQ(canv.NumPrimitives(), 1u);

   EXPECT_EQ(text->GetText(), "Hello World");
   EXPECT_EQ(text->text.color, RColor::kBlack);
   EXPECT_DOUBLE_EQ(text->text.size, 12.5);
   EXPECT_DOUBLE_EQ(text->text.angle, 90.);
   EXPECT_EQ(text->text.align, RAttrText::kLeftTop);
   EXPECT_EQ(text->text.font.family, "Arial");
}

// Test RLegend API
TEST(Primitives, RLegend)
{
   RCanvas canv;
   auto line1 = canv.Draw<RLine>(RPadPos(0.1_normal, 0.1_normal), RPadPos(0.9_normal,0.9_normal));
   auto line2 = canv.Draw<RLine>(RPadPos(0.1_normal, 0.9_normal), RPadPos(0.9_normal,0.1_normal));
   auto line3 = canv.Draw<RLine>(RPadPos(0.9_normal, 0.1_normal), RPadPos(0.1_normal,0.9_normal));

   line1->line.color = RColor::kRed;
   line2->line.color = RColor::kGreen;
   line3->line.color = RColor::kBlue;

   auto legend = canv.Draw<RLegend>("Legend title");
   legend->fill.style = RAttrFill::k3005;
   legend->fill.color = RColor::kWhite;
   legend->border.width = 2.;
   legend->border.color = RColor::kRed;
   legend->AddEntry(line1, "RLine 1", "l");
   legend->AddEntry(line2, "RLine 2", "l");
   legend->AddEntry(line3, "RLine 3", "l");

   auto custom = legend->AddEntry("test", "lfm");
   custom->line.color = RColor::kGreen;
   custom->line.width = 5.;
   custom->line.style = RAttrLine::kDashed;
   custom->fill.color = RColor::kBlue;
   custom->fill.style = RAttrFill::k3004;
   custom->marker.color = RColor::kRed;
   custom->marker.size = 3.;
   custom->marker.style = RAttrMarker::kOpenCross;

   EXPECT_EQ(canv.NumPrimitives(), 4u);

   EXPECT_EQ(legend->NumEntries(), 4u);
   EXPECT_EQ(legend->GetTitle(), "Legend title");
   EXPECT_EQ(legend->fill.style, RAttrFill::k3005);
   EXPECT_EQ(legend->fill.color, RColor::kWhite);
   EXPECT_EQ(legend->border.width, 2.);
   EXPECT_EQ(legend->border.color, RColor::kRed);

   EXPECT_EQ(legend->GetEntry(0).GetLine(), true);
   EXPECT_EQ(legend->GetEntry(1).GetFill(), false);
   EXPECT_EQ(legend->GetEntry(2).GetMarker(), false);

   EXPECT_EQ(legend->GetEntry(3).GetLine(), true);
   EXPECT_EQ(legend->GetEntry(3).GetFill(), true);
   EXPECT_EQ(legend->GetEntry(3).GetMarker(), true);
   auto custom2 = std::dynamic_pointer_cast<RLegend::RCustomDrawable>(legend->GetEntry(3).GetDrawable());

   EXPECT_NE(custom2, nullptr);
   EXPECT_EQ(custom2->line.color, RColor::kGreen);
   EXPECT_EQ(custom2->line.style, RAttrLine::kDashed);
   EXPECT_EQ(custom2->fill.color, RColor::kBlue);
   EXPECT_EQ(custom2->fill.style, RAttrFill::k3004);
   EXPECT_EQ(custom2->marker.color, RColor::kRed);
}

// Test RPaveText API
TEST(Primitives, RPaveText)
{
   RCanvas canv;

   auto text = canv.Add<RPaveText>();

   text->text.color = RColor::kBlack;
   text->text.size = 12;
   text->text.align = RAttrText::kLeftTop;
   text->text.font.family = "Times New Roman";
   text->border.color = RColor::kRed;
   text->border.width = 3.;
   text->fill.color = RColor::kBlue;
   text->fill.style = RAttrFill::k3003;

   text->AddLine("First line");
   text->AddLine("Second line");
   text->AddLine("Third line");

   EXPECT_EQ(canv.NumPrimitives(), 1u);

   EXPECT_EQ(text->NumLines(), 3u);
   EXPECT_EQ(text->GetLine(0), "First line");
   EXPECT_EQ(text->GetLine(1), "Second line");
   EXPECT_EQ(text->GetLine(2), "Third line");

   EXPECT_EQ(text->text.color, RColor::kBlack);
   EXPECT_DOUBLE_EQ(text->text.size, 12);
   EXPECT_EQ(text->text.align, RAttrText::kLeftTop);
   EXPECT_EQ(text->text.font.family, "Times New Roman");

   EXPECT_EQ(text->border.color, RColor::kRed);
   EXPECT_DOUBLE_EQ(text->border.width, 3.);

   EXPECT_EQ(text->fill.color, RColor::kBlue);
   EXPECT_EQ(text->fill.style, RAttrFill::k3003);
}
