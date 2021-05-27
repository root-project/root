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

   box->SetLineColor(RColor::kRed).SetLineWidth(5.).SetLineStyle(7);
   box->SetFillColor(RColor::kBlue).SetFillStyle(6);

   EXPECT_EQ(canv.NumPrimitives(), 1u);

   EXPECT_EQ(box->GetLineColor(), RColor::kRed);
   EXPECT_DOUBLE_EQ(box->GetLineWidth(), 5.);
   EXPECT_EQ(box->GetLineStyle(), 7);

   EXPECT_EQ(box->GetFillColor(), RColor::kBlue);
   EXPECT_EQ(box->GetFillStyle(), 6);
}

// Test RLine API
TEST(Primitives, RLine)
{
   RCanvas canv;
   auto line = canv.Draw<RLine>(RPadPos(0.1_normal, 0.1_normal), RPadPos(0.9_normal,0.9_normal));

   line->SetLineColor(RColor::kRed).SetLineWidth(5.).SetLineStyle(7);

   EXPECT_EQ(canv.NumPrimitives(), 1u);

   EXPECT_EQ(line->GetLineColor(), RColor::kRed);
   EXPECT_DOUBLE_EQ(line->GetLineWidth(), 5.);
   EXPECT_EQ(line->GetLineStyle(), 7);
}

// Test RMarker API
TEST(Primitives, RMarker)
{
   RCanvas canv;
   auto marker = canv.Draw<RMarker>(RPadPos(0.5_normal, 0.5_normal));

   marker->SetMarkerStyle(7).SetMarkerSize(2.5).SetMarkerColor(RColor::kGreen);

   EXPECT_EQ(canv.NumPrimitives(), 1u);

   EXPECT_EQ(marker->GetMarkerColor(), RColor::kGreen);
   EXPECT_DOUBLE_EQ(marker->GetMarkerSize(), 2.5);
   EXPECT_EQ(marker->GetMarkerStyle(), 7);
}

// Test RText API
TEST(Primitives, RText)
{
   RCanvas canv;

   auto text = canv.Draw<RText>(RPadPos(0.5_normal, 0.5_normal), "Hello World");

   text->SetTextColor(RColor::kBlack).SetTextSize(12.5).SetTextAngle(90.).SetTextAlign(13).SetFontFamily("Arial");

   EXPECT_EQ(canv.NumPrimitives(), 1u);

   EXPECT_EQ(text->GetText(), "Hello World");
   EXPECT_EQ(text->GetTextColor(), RColor::kBlack);
   EXPECT_DOUBLE_EQ(text->GetTextSize(), 12.5);
   EXPECT_DOUBLE_EQ(text->GetTextAngle(), 90.);
   EXPECT_EQ(text->GetTextAlign(), 13);
   EXPECT_EQ(text->GetFontFamily(), "Arial");
}

// Test RLegend API
TEST(Primitives, RLegend)
{
   RCanvas canv;
   auto line1 = canv.Draw<RLine>(RPadPos(0.1_normal, 0.1_normal), RPadPos(0.9_normal,0.9_normal));
   auto line2 = canv.Draw<RLine>(RPadPos(0.1_normal, 0.9_normal), RPadPos(0.9_normal,0.1_normal));
   auto line3 = canv.Draw<RLine>(RPadPos(0.9_normal, 0.1_normal), RPadPos(0.1_normal,0.9_normal));

   line1->SetLineColor(RColor::kRed);
   line2->SetLineColor(RColor::kGreen);
   line3->SetLineColor(RColor::kBlue);

   auto legend = canv.Draw<RLegend>("Legend title");
   legend->SetFillStyle(5).SetFillColor(RColor::kWhite);
   legend->SetLineWidth(2).SetLineColor(RColor::kRed);
   legend->AddEntry(line1, "RLine 1");
   legend->AddEntry(line2, "RLine 2");
   legend->AddEntry(line3, "RLine 3");

   EXPECT_EQ(canv.NumPrimitives(), 4u);

   EXPECT_EQ(legend->NumEntries(), 3u);
   EXPECT_EQ(legend->GetTitle(), "Legend title");
   EXPECT_EQ(legend->GetFillColor(), RColor::kWhite);
}

// Test RPaveText API
TEST(Primitives, RPaveText)
{
   RCanvas canv;

   auto text = canv.Draw<RPaveText>();

   text->SetTextColor(RColor::kBlack).SetTextSize(12).SetTextAlign(13).SetFontFamily("Times New Roman");
   text->SetLineColor(RColor::kRed).SetLineWidth(3);
   text->SetFillColor(RColor::kBlue).SetFillStyle(3003);

   text->AddLine("First line");
   text->AddLine("Second line");
   text->AddLine("Third line");

   EXPECT_EQ(canv.NumPrimitives(), 1u);

   EXPECT_EQ(text->NumLines(), 3u);
   EXPECT_EQ(text->GetLine(0), "First line");
   EXPECT_EQ(text->GetLine(1), "Second line");
   EXPECT_EQ(text->GetLine(2), "Third line");

   EXPECT_EQ(text->GetTextColor(), RColor::kBlack);
   EXPECT_DOUBLE_EQ(text->GetTextSize(), 12);
   EXPECT_EQ(text->GetTextAlign(), 13);
   EXPECT_EQ(text->GetFontFamily(), "Times New Roman");

   EXPECT_EQ(text->GetLineColor(), RColor::kRed);
   EXPECT_EQ(text->GetLineWidth(), 3);

   EXPECT_EQ(text->GetFillColor(), RColor::kBlue);
   EXPECT_EQ(text->GetFillStyle(), 3003);
}

