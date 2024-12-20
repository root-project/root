/// \file
/// \ingroup tutorial_graphics
/// \notebook
/// Simple example illustrating how to draw TGaxis objects in various formats.
///
/// \macro_image
/// \macro_code
///
/// \authors Rene Brun, Olivier Couet

void gaxis()
{
   auto c1 = new TCanvas("c1", "Examples of TGaxis", 10, 10, 700, 500);
   c1->Range(-10, -1, 10, 1);

   auto axis1 = new TGaxis(-4.5, -0.2, 5.5, -0.2, -6, 8, 510, "");
   axis1->Draw();

   auto axis2 = new TGaxis(-4.5, 0.2, 5.5, 0.2, 0.001, 10000, 510, "G");
   axis2->Draw();

   auto axis3 = new TGaxis(-9, -0.8, -9, 0.8, -8, 8, 50510, "");
   axis3->SetTitle("axis3");
   axis3->SetTitleOffset(0.5);
   axis3->Draw();

   auto axis4 = new TGaxis(-7, -0.8, -7, 0.8, 1, 10000, 50510, "G");
   axis4->SetTitle("axis4");
   axis4->Draw();

   auto axis5 = new TGaxis(-4.5, -0.6, 5.5, -0.6, 1.2, 1.32, 80506, "-+");
   axis5->SetLabelSize(0.03);
   axis5->SetTextFont(72);
   axis5->Draw();

   auto axis6 = new TGaxis(-4.5, 0.5, 5.5, 0.5, 100, 900, 50510, "-");
   axis6->Draw();

   auto axis7 = new TGaxis(-5.5, 0.85, 5.5, 0.85, 0, 4.3e-6, 510, "");
   axis7->Draw();

   auto axis8 = new TGaxis(8, -0.8, 8, 0.8, 0, 9000, 50510, "+L");
   axis8->Draw();

   // One can make a vertical axis going top->bottom.
   // However one need to adjust labels align to avoid overlapping.
   auto axis9 = new TGaxis(6.5, 0.8, 6.5, -0.8, 0, 90, 50510, "-L");
   axis9->Draw();
}
