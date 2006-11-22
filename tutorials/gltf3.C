void gltf3(){
  // TF3 can be drawn in several styles,
   // default // (like surface4)
   // kMaple0 (very nice colours)
   // kMaple1 (nice colours and outlines)
   // kMaple2 (nice colour outlines).
   // To switch between them, you can press 's' key.

   TCanvas *cnv = new TCanvas("glc", "TF3: Klein bottle", 200, 10, 600, 600);

   TPaveLabel *title = new TPaveLabel(0.04, 0.86, 0.96, 0.98, "\"gl\" option for TF3. Select plot and press 's' to change the color.");
   title->SetFillColor(32);
   title->Draw();

   TPad *tf3Pad  = new TPad("box", "box", 0.04, 0.04, 0.96, 0.8);   
   tf3Pad->Draw();

 //  TF3 *tf3 = new TF3("sample", "x*x+y*y-z*z-1", -2., 2., -2., 2., -2., 2.);
   TF3 *tf3 = new TF3("Klein Bottle","(x*x + y*y + z*z + 2*y -1)*((x*x + y*y + z*z - 2*y -1)*(x*x + y*y + z*z - 2*y -1)-8*z*z)+16*x*z*(x*x + y*y + z*z - 2*y -1)", -3.5, 3.5, -3.5, 3.5, -5.5, 5.5);
   tf3->SetFillColor(kRed);
   tf3Pad->cd();
   tf3->Draw("gl");
}
