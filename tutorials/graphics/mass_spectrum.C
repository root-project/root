// This macro makes use of some basic graphics primitives such as line, arrow
// and text. It as been written using the TCanvas ToolBar to produce a first
// draft and was then modified for fine adjustments. Note also the use
// of C funtions. They allow to simplify the macro reading and editing by
// avoiding code repetition or defining some graphics attributes in one single
// place. This technique to generate drawings may appear not very user friendly
// compare to all the "wysiwyg" graphics editors available. In some cases it can
// be more powerful than a GUI interface because it allows to generate very
// precise drawing and using computation to generate them.
//Author: Olivier Couet

void hline (Double_t x, Double_t y)
{
  Double_t dx = 0.1;
  TLine *l = new TLine(x,y,x+dx,y);
  l->Draw();
  l->SetLineWidth(4);
}

void arrow (Double_t x1, Double_t y1, Double_t x2, Double_t y2, Int_t ls)
{
  TArrow *arr = new TArrow(x1,y1,x2,y2,0.025,"|>");
  arr->SetFillColor(1);
  arr->SetFillStyle(1001);
  arr->SetLineStyle(ls);
  arr->SetAngle(19);
  arr->Draw();
}

void mass_spectrum()
{
   TCanvas *C = new TCanvas("C","C",800,500);

   hline (0.10,0.25);
   hline (0.10,0.80);
   hline (0.30,0.90);
   hline (0.30,0.35);
   hline (0.45,0.60);
   hline (0.58,0.68);
   hline (0.73,0.70);
   hline (0.89,0.75);

   arrow(0.32, 0.90, 0.32, 0.35, 1);
   arrow(0.34, 0.90, 0.34, 0.35, 1);
   arrow(0.36, 0.90, 0.36, 0.60, 1);
   arrow(0.38, 0.90, 0.38, 0.70, 1);

   arrow(0.30, 0.90, 0.18, 0.25, 1);
   arrow(0.30, 0.35, 0.19, 0.25, 1);
   arrow(0.40, 0.90, 0.47, 0.61, 1);

   arrow(0.15, 0.25, 0.15, 0.19, 1);
   arrow(0.15, 0.80, 0.15, 0.74, 1);

   arrow(0.50, 0.60, 0.50, 0.54, 1);
   arrow(0.60, 0.68, 0.60, 0.62, 1);
   arrow(0.94, 0.75, 0.94, 0.69, 1);

   arrow(0.32, 0.35, 0.32, 0.19, 1);
   arrow(0.36, 0.35, 0.36, 0.19, 1);
   arrow(0.38, 0.35, 0.38, 0.19, 1);

   arrow(0.40, 0.90, 0.60, 0.68, 1);
   arrow(0.40, 0.90, 0.90, 0.75, 1);
   arrow(0.45, 0.60, 0.35, 0.35, 1);
   arrow(0.30, 0.90, 0.18, 0.80, 2);
   arrow(0.67, 0.68, 0.36, 0.35, 1);
   arrow(0.78, 0.70, 0.37, 0.35, 2);
   arrow(0.91, 0.75, 0.39, 0.35, 1);

   TLatex l1;
   l1.SetTextSize(0.035);
   l1.SetTextAlign(22);
   l1.SetTextFont(132);
   l1.DrawLatex(0.15, 0.73, "hadrons");
   l1.DrawLatex(0.15, 0.18, "hadrons");
   l1.DrawLatex(0.32, 0.18, "hadrons");
   l1.DrawLatex(0.38, 0.59, "hadrons");
   l1.DrawLatex(0.50, 0.53, "hadrons");
   l1.DrawLatex(0.94, 0.68, "hadrons");
   l1.DrawLatex(0.58, 0.62, "hadrons");
   l1.DrawLatex(0.41, 0.18, "radiative");

   TLatex l2;
   l2.SetTextSize(0.038);
   l2.SetTextAlign(22);
   l2.SetTextFont(132);
   l2.DrawLatex(0.07, 0.08, "#font[12]{J^{PC}} =");
   l2.DrawLatex(0.15, 0.08, "0^{-+}");
   l2.DrawLatex(0.35, 0.08, "1^{--}");
   l2.DrawLatex(0.50, 0.08, "0^{++}");
   l2.DrawLatex(0.62, 0.08, "1^{++}");
   l2.DrawLatex(0.77, 0.08, "1^{+-}");
   l2.DrawLatex(0.93, 0.08, "2^{++}");
   l2.DrawLatex(0.15, 0.83, "#eta_{c}(2S)");
   l2.DrawLatex(0.15, 0.28, "#eta_{c}(1S)");
   l2.DrawLatex(0.35, 0.93, "#psi(2S)");
   l2.DrawLatex(0.45, 0.35, "#font[12]{J}/#psi(1S)");
   l2.DrawLatex(0.51, 0.63, "#chi_{c0}(1P)");
   l2.DrawLatex(0.63, 0.71, "#chi_{c1}(1P)");
   l2.DrawLatex(0.78, 0.73, "h_{c1}(1P)");
   l2.DrawLatex(0.94, 0.78, "#chi_{c2}(1P)");

   TLatex l3;
   l3.SetTextSize(0.037);
   l3.SetTextAlign(11);
   l3.SetTextFont(132);
   l3.DrawLatex(0.23, 0.86, "#font[152]{g}");
   l3.DrawLatex(0.23, 0.57, "#font[152]{g}");
   l3.DrawLatex(0.44, 0.77, "#font[152]{g}");
   l3.DrawLatex(0.40, 0.50, "#font[152]{g}");
   l3.DrawLatex(0.45, 0.46, "#font[152]{g}");
   l3.DrawLatex(0.71, 0.61, "#font[152]{g}");
   l3.DrawLatex(0.24, 0.31, "#font[152]{g}");
   l3.DrawLatex(0.38, 0.81, "#font[152]{g^{*}}");
   l3.DrawLatex(0.355, 0.16, "#font[152]{g^{*}}");
   l3.DrawLatex(0.295, 0.50, "#pi#pi");
   l3.DrawLatex(0.345, 0.53, "#eta,#pi^{0}");
   l3.DrawLatex(0.70, 0.65, "#pi^{0}");
}
