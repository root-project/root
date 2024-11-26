/// \file
/// \ingroup tutorial_webcanv
/// \notebook -js
/// Use of interactive URL links inside TLatex.
///
/// JSROOT now supports '#url[link]{label}' syntax
/// It can be combined with any other latex commands like color ot font.
/// While TLatex used in many places, one can add external links to histogram title,
/// axis title, legend entry and so on.
///
/// Functionality available only in web-based graphics
///
/// \macro_image (tcanvas_js)
/// \macro_code
///
/// \author Sergey Linev

void latex_url()
{
   auto c1 = new TCanvas("c1", "Use of #url in TLatex", 1200, 800);

   if (!gROOT->IsBatch() && !c1->IsWeb())
      ::Warning("latex_url.cxx", "macro may not work without enabling web-based canvas");

   auto latex = new TLatex(0.5, 0.5, "Link on #color[4]{#url[https://root.cern]{root.cern}} web site");
   latex->SetTextSize(0.1);
   latex->SetTextAlign(22);
   c1->Add(latex);
}
