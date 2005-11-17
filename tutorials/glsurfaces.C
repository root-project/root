{
   //
   // This example draws 4 surfaces using OpenGL in pad. The commands used
   // are exactly the same as with a normal pad. The only command to add is:
   
   gStyle->SetCanvasPreferGL(true);

   // after this command all legos surfaces are automatically rendered with
   // OpenGL.
   //

   TCanvas *c1 = new TCanvas("c1","Surfaces Drawing Options with gl",200,10,700,700);
   c1->SetFillColor(42);
   gStyle->SetFrameFillColor(42);

   c1->Divide(2, 2);
   TPad *pad = static_cast<TPad *>(c1->cd(1));

   TF2 *fun1 = new TF2("fun1","x * x - y * y - 1", -6., 6., -6., 6.);
   fun1->SetFillColor(kRed);
   fun1->Draw("surf4");

   pad = static_cast<TPad *>(c1->cd(2));

   TF2 *fun2 = new TF2("fun2","x * x + y * y - 1", -6., 6., -6., 6.);
   fun2->SetFillColor(kGreen);
   fun2->Draw("surf");

   pad = static_cast<TPad *>(c1->cd(3));
   TF2 *fun3 = new TF2("fun3","sin(x) / x * cos(y) * y", -6., 6., -6., 6.);
   fun3->SetFillColor(kWhite);
   fun3->Draw("surf");

   pad = static_cast<TPad *>(c1->cd(4));
   TF2 *fun4 = new TF2("fun4","sqrt(1 - y * y - x * x)", -1., 1., -1., 1.);
   fun4->SetFillColor(kMagenta);
   fun4->Draw("surf4");
}
