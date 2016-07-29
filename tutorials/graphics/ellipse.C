/// \file
/// \ingroup tutorial_graphics
/// \notebook -js
/// Draw ellipses.
///
/// \macro_image
/// \macro_code
///
/// \author Rene Brun

TCanvas *ellipse(){
   TCanvas *c1 = new TCanvas("c1");
   c1->Range(0,0,1,1);
   TPaveLabel *pel = new TPaveLabel(0.1,0.8,0.9,0.95,"Examples of Ellipses");
   pel->SetFillColor(42);
   pel->Draw();
   TEllipse *el1 = new TEllipse(0.25,0.25,.1,.2);
   el1->Draw();
   TEllipse *el2 = new TEllipse(0.25,0.6,.2,.1);
   el2->SetFillColor(6);
   el2->SetFillStyle(3008);
   el2->Draw();
   TEllipse *el3 = new TEllipse(0.75,0.6,.2,.1,45,315);
   el3->SetFillColor(2);
   el3->SetFillStyle(1001);
   el3->SetLineColor(4);
   el3->Draw();
   TEllipse *el4 = new TEllipse(0.75,0.25,.2,.15,45,315,62);
   el4->SetFillColor(5);
   el4->SetFillStyle(1001);
   el4->SetLineColor(4);
   el4->SetLineWidth(6);
   el4->Draw();
   return c1;
}
