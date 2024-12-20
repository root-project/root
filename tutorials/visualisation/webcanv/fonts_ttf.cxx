/// \file
/// \ingroup tutorial_webcanv
/// \notebook -js
/// Usage of TTF fonts in web canvas.
///
/// One can load TTF font file and specify it for usage in the web canvas
/// Produced drawing also can be saved in PDF files.
///
/// Functionality available only in web-based graphics
///
/// \macro_image (tcanvas_js)
/// \macro_code
///
/// \author Sergey Linev

void fonts_ttf()
{
   TString fontdir = TROOT::GetDataDir() + "/fonts/";

   auto fontid = TWebCanvas::AddFont("UserBold", fontdir + "trebuc.ttf");
   auto fontid_bold = TWebCanvas::AddFont("UserBold", fontdir + "trebucbd.ttf");
   auto fontid_bold_italic = TWebCanvas::AddFont("UserBoldItalic", fontdir + "trebucbi.ttf");
   auto fontid_italic = TWebCanvas::AddFont("UserItalic", fontdir + "trebucit.ttf");

   if ((fontid < 0) || (fontid_bold < 0) || (fontid_bold_italic < 0) || (fontid_italic < 0))
      ::Error("fonts_ttf.cxx", "fail to load ttf fonts from %s", fontdir.Data());

   auto c1 = new TCanvas("c1", "c1", 1000, 600);

   if (!gROOT->IsBatch() && !c1->IsWeb())
      ::Warning("fonts_ttf.cxx", "macro will not work without enabling web-based canvas");

   auto l1 = new TLatex(0.5, 0.8, "Custom font from trebuc.ttf");
   l1->SetTextFont(fontid);
   l1->SetTextAlign(22);
   l1->SetTextSize(0.1);
   c1->Add(l1);

   auto l2 = new TLatex(0.5, 0.6, "Custom bold font from trebucbd.ttf");
   l2->SetTextFont(fontid_bold);
   l2->SetTextAlign(22);
   l2->SetTextSize(0.1);
   c1->Add(l2);

   auto l3 = new TLatex(0.5, 0.4, "Custom bold italic font from trebucbi.ttf");
   l3->SetTextFont(fontid_bold_italic);
   l3->SetTextAlign(22);
   l3->SetTextSize(0.1);
   c1->Add(l3);

   auto l4 = new TLatex(0.5, 0.2, "Custom italic font from trebucit.ttf");
   l4->SetTextFont(fontid_italic);
   l4->SetTextAlign(22);
   l4->SetTextSize(0.1);
   c1->Add(l4);
}