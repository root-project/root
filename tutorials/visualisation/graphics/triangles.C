/// \file
/// \ingroup tutorial_graphics
/// \notebook
///  \preview Create small triangles at random positions on the canvas.
/// Assign a unique ID to each triangle, and give each one a random color from the color palette.
///
/// ~~~{.cpp}
/// root > .x triangles.C
/// ~~~
///
/// When a triangle is clicked, a message displaying its unique number and color will be printed.
///
/// \macro_image
/// \macro_code
///
/// \author Rene Brun

void triangles(int ntriangles = 50)
{
   auto c1 = new TCanvas("c1", "triangles", 10, 10, 700, 700);
   gStyle->SetPalette(kCMYK);
   TRandom r;
   double dx = 0.2;
   double dy = 0.2;
   int ncolors = TColor::GetNumberOfColors();
   double x[4], y[4];
   for (int i = 0; i < ntriangles; i++) {
      x[0] = r.Uniform(.05, .95);
      y[0] = r.Uniform(.05, .95);
      x[1] = x[0] + dx * r.Rndm();
      y[1] = y[0] + dy * r.Rndm();
      x[2] = x[1] - dx * r.Rndm();
      y[2] = y[1] - dy * r.Rndm();
      x[3] = x[0];
      y[3] = y[0];
      auto pl = new TPolyLine(4, x, y);
      pl->SetUniqueID(i);
      int ci = ncolors * r.Rndm();
      TColor *c = gROOT->GetColor(TColor::GetColorPalette(ci));
      c->SetAlpha(r.Rndm());
      pl->SetFillColor(c->GetNumber());
      pl->Draw("f");
   }
   c1->AddExec("ex", "TriangleClicked()");
}

void TriangleClicked()
{
   // this action function is called whenever you move the mouse
   // it just prints the id of the picked triangle
   // you can add graphics actions instead
   int event = gPad->GetEvent();
   if (event != 11)
      return; // may be comment this line
   TObject *select = gPad->GetSelected();
   if (!select)
      return;
   if (select->InheritsFrom(TPolyLine::Class())) {
      TPolyLine *pl = (TPolyLine *)select;
      printf("You have clicked triangle %d, color=%d\n", pl->GetUniqueID(), pl->GetFillColor());
   }
}
