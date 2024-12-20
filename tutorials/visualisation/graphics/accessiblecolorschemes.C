/// \file
/// \ingroup tutorial_graphics
/// \notebook
/// \preview Choosing an appropriate color scheme is essential for making results easy to understand and interpret.
/// Factors like colorblindness and converting colors to grayscale for publications
/// can impact accessibility. Furthermore, results should be aesthetically pleasing. The following
/// three color schemes, recommended by M. Petroff in [arXiv:2107.02270v2](https://arxiv.org/pdf/2107.02270)
/// and available on [GitHub](https://github.com/mpetroff/accessible-color-cycles)
/// under the MIT License, meet these criteria.
///
/// \macro_image
/// \macro_code
///
/// \author Olivier Couet

void box(double x1, double y1, double x2, double y2, int col)
{
   auto b1 = new TBox(x1, y1, x2, y2);
   b1->SetFillColor(col);
   b1->Draw();

   TColor *c = gROOT->GetColor(col);
   auto tc = new TLatex((x2 + x1) / 2., 0.01 + (y2 + y1) / 2., Form("#splitline{%s}{%s}", c->GetName(), c->GetTitle()));
   tc->SetTextFont(42);
   tc->SetTextAlign(23);
   tc->SetTextSize(0.020);
   tc->Draw();
}

void accessiblecolorschemes()
{
   auto C = new TCanvas("C", "C", 600, 800);
   int c;
   double x, y;
   double w = 0.2;
   double h = 0.08;
   auto t = new TText();
   t->SetTextSize(0.025);
   t->SetTextFont(42);

   // 6-colors scheme
   x = 0.1;
   y = 0.1;
   t->DrawText(x, y - h / 2., "6-colors scheme");
   for (c = kP6Blue; c < kP6Blue + 6; c++) {
      box(x, y, x + w, y + h, c);
      y = y + h;
   }

   // 8-color scheme
   y = 0.1;
   x = 0.4;
   t->DrawText(x, y - h / 2., "8-colors scheme");
   for (c = kP8Blue; c < kP8Blue + 8; c++) {
      box(x, y, x + w, y + h, c);
      y = y + h;
   }

   // 10-color scheme
   y = 0.1;
   x = 0.7;
   t->DrawText(x, y - h / 2., "10-colors scheme");
   for (c = kP10Blue; c < kP10Blue + 10; c++) {
      box(x, y, x + w, y + h, c);
      y = y + h;
   }
}