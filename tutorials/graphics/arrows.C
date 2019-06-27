/// \file
/// \ingroup tutorial_graphics
/// \notebook -js
/// Draw arrows.
///
/// \macro_image (js)
/// \macro_code
///
/// \author Rene Brun

void arrows(){
   TCanvas *c1 = new TCanvas("c1");
   c1->Range(0,0,1,1);

   TPaveLabel *par = new TPaveLabel(0.1,0.8,0.9,0.95,"Examples of various arrows formats");
   par->SetFillColor(42);
   par->Draw();

   TArrow *ar1 = new TArrow(0.1,0.1,0.1,0.7);
   ar1->Draw();
   TArrow *ar2 = new TArrow(0.2,0.1,0.2,0.7,0.05,"|>");
   ar2->SetAngle(40);
   ar2->SetLineWidth(2);
   ar2->Draw();
   TArrow *ar3 = new TArrow(0.3,0.1,0.3,0.7,0.05,"<|>");
   ar3->SetAngle(40);
   ar3->SetLineWidth(2);
   ar3->Draw();
   TArrow *ar4 = new TArrow(0.46,0.7,0.82,0.42,0.07,"|>");
   ar4->SetAngle(60);
   ar4->SetLineWidth(2);
   ar4->SetFillColor(2);
   ar4->Draw();
   TArrow *ar5 = new TArrow(0.4,0.25,0.95,0.25,0.15,"<|>");
   ar5->SetAngle(60);
   ar5->SetLineWidth(4);
   ar5->SetLineColor(4);
   ar5->SetFillStyle(3008);
   ar5->SetFillColor(2);
   ar5->Draw();
}
