// Various surfaces rendered with GL.
// This example draws 6 surfaces using OpenGL in pad (one is remake
// of a classic surfaces.C, another contains 4 surfaces).
//
// The commands used are exactly the same as with a normal pad. 
// The only command to add is: gStyle->SetCanvasPreferGL(true);
// Authors: Rene Brun, Timur Pocheptsov

void glsurfaces()
{
   gStyle->SetPalette(0);
   
   // after this command all legos surfaces (surf/srf1/surf2/surf4/tf3
   // options) are automatically rendered with OpenGL.
   gStyle->SetCanvasPreferGL(kTRUE);

   TCanvas *c1 = new TCanvas("glc1","Surfaces Drawing Options",200,10,700,900);
   c1->SetFillColor(42);
   gStyle->SetFrameFillColor(42);
   title = new TPaveText(0.2, 0.96, 0.8, 0.995);
   title->SetFillColor(33);
   title->AddText("Examples of Surface options");
   title->Draw();

   TPad *pad1 = new TPad("pad1","Gouraud shading", 0.03, 0.50, 0.98, 0.95, 21);
   TPad *pad2 = new TPad("pad2","Color mesh", 0.03, 0.02, 0.98, 0.48, 21);
   pad1->Draw();
   pad2->Draw();
   // We generate a 2-D function
   TF2 *f2 = new TF2("f2","x**2 + y**2 - x**3 -8*x*y**4", -1., 1.2, -1.5, 1.5);
   // Draw this function in pad1 with Gouraud shading option
   pad1->cd();
   pad1->SetLogz();
   f2->SetFillColor(45);
   f2->Draw("glsurf4");

   TF2 *f2clone = new TF2("f2clone","x**2 + y**2 - x**3 -8*x*y**4",
                          -1., 1.2, -1.5, 1.5);
   // Draw this function in pad2 with color mesh option
   pad2->cd();
   pad2->SetLogz();
   f2clone->Draw("glsurf1");
   
   //add axis titles. The titles are set on the intermediate
   //histogram used for visualisation. We must force this histogram
   //to be created, then force the redrawing of the two pads
   pad2->Update();
   f2->GetHistogram()->GetXaxis()->SetTitle("x title");
   f2->GetHistogram()->GetYaxis()->SetTitle("y title");
   f2->GetHistogram()->GetXaxis()->SetTitleOffset(1.4);
   f2->GetHistogram()->GetYaxis()->SetTitleOffset(1.4);
   f2clone->GetHistogram()->GetXaxis()->SetTitle("x title");
   f2clone->GetHistogram()->GetYaxis()->SetTitle("y title");
   f2clone->GetHistogram()->GetXaxis()->SetTitleOffset(1.4);
   f2clone->GetHistogram()->GetYaxis()->SetTitleOffset(1.4);
   pad1->Modified();
   pad2->Modified();

   TCanvas *c2 = new TCanvas("glc2","Surfaces Drawing Options with gl",
                             700,10,700,700);
   c2->SetFillColor(42);
   gStyle->SetFrameFillColor(42);

   c2->Divide(2, 2);

   c2->cd(1);
   TF2 *fun1 = new TF2("fun1","1000*((sin(x)/x)*(sin(y)/y))+200",
                       -6., 6., -6., 6.);
   fun1->SetNpx(30);
   fun1->SetNpy(30);
   fun1->SetFillColor(kGreen);
   fun1->Draw("glsurf3");

   c2->cd(2);
   TF2 *fun2 = new TF2("fun2","cos(y)*sin(x)+cos(x)*sin(y)",
                       -6., 6., -6., 6.);
   fun2->Draw("glsurf1cyl");

   c2->cd(3);
   TF2 *fun3 = new TF2("fun3","sin(x) / x * cos(y) * y", -6., 6., -6., 6.);
   fun3->Draw("glsurfpol");

   c2->cd(4);
   TF3 *fun4 = new TF3("fun4","sin(x * x + y * y + z * z - 4)",
                       -2.5, 2.5, -2.5, 2.5, -2.5, 2.5);
   Int_t colInd = TColor::GetColor(1.f, 0.5f, 0.f);
   fun4->SetFillColor(colInd);
   fun4->Draw("gl");//tf3 option
}
