/// \file
/// \ingroup tutorial_webcanv
/// \notebook -js
/// Demonstrates how to use interactive URL links in TLatex.
///
/// JSROOT and standard SVG output support the syntax '#url[link]{label}'.
/// This can be combined with other LaTeX commands, such as color or font settings.
///
/// Since TLatex is used in many contexts, external links can be added to
/// histogram titles, axis titles, legend entries, and more.
///
/// This functionality is available only in web-based graphics and
/// standard SVG output.
///
/// \macro_image (tcanvas_js)
/// \macro_code
///
/// \author Sergey Linev

void latex_url()
{
   auto c1 = new TCanvas("c1", "Use of #url in TLatex", 1200, 800);

   auto latex = new TLatex(0.5, 0.5, "Link on #color[4]{#url[https://root.cern]{root.cern}} web site");
   latex->SetTextSize(0.1);
   latex->SetTextAlign(22);
   latex->SetTextAngle(30.);
   c1->Add(latex);
   c1->Print("c1.svg");
}
