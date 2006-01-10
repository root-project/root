{
   // TF3 can be drawn in several styles,
   // default // (like surface4)
   // kMaple0 (very nice colours)
   // kMaple1 (nice colours and outlines)
   // kMaple2 (nice colour outlines).
   // To switch between them, you can press 's' key.

   gStyle->SetCanvasPreferGL(kTRUE);
   
   TCanvas *cnv = new TCanvas("c", "TF3 sample (press 's' to change colour)", 700, 10, 700, 700);
   TF3 *tf3 = new TF3("sample", "x*x+y*y-z*z-1", -2., 2., -2., 2., -2., 2.);
   
   tf3->SetFillColor(kGreen);
   tf3->Draw();
}
