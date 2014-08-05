// Draws the Klein bottle.
// Klein bottle is closed nonorientable surface that has no inside or
// outside. TF3 can be drawn in several styles:
//   default - like surface4
//   kMaple0 - very nice colours
//   kMaple1 - nice colours and outlines
//   kMaple2 - nice colour outlines.
// To switch between them, you can press 's' key.
// Author: Timur Pocheptsov

void gltf3()
{
   gStyle->SetCanvasPreferGL(1);
   TCanvas *cnv = new TCanvas("glc", "TF3: Klein bottle", 200, 10, 600, 600);
   // TCanvas *cnv = new TCanvas("glc", "TF3: Torus", 200, 10, 600, 600);

   TPaveLabel *title = new TPaveLabel(0.04, 0.86, 0.96, 0.98,
      "\"gl\" option for TF3. Select plot and press 's' to change the color.");
   title->SetFillColor(32);
   title->Draw();

   TPad *tf3Pad  = new TPad("box", "box", 0.04, 0.04, 0.96, 0.8);
   tf3Pad->Draw();

   TFormula f1 = TFormula("f1", "x*x + y*y + z*z + 2*y - 1");
   TFormula f2 = TFormula("f2", "x*x + y*y + z*z - 2*y - 1");

   // Klein bottle with cutted top&bottom parts
   // The Klein bottle is a closed nonorientable surface that has no
   // inside or outside.

   TF3 *tf3 = new TF3("Klein Bottle","f1*(f2*f2-8*z*z) + 16*x*z*f2",
                      -3.5, 3.5, -3.5, 3.5, -2.5, 2.5);
   // Torus
   // TF3 *tf3 = new TF3("Torus","4*(x^4 +
   // (y^2+z^2)^2)+17*x^2*(y^2+z^2)-20*(x^2+y^2+z^2)+17", -2.5., 2.5.,
   // -2.5., 2.5., -2.5., 2.5.);

   tf3->SetFillColor(kRed);
   tf3Pad->cd();
   tf3->Draw("gl");
}
