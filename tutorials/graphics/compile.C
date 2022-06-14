/// \file
/// \ingroup tutorial_graphics
/// \notebook -js
/// This macro produces the flowchart of TFormula::Compile
///
/// \macro_image
/// \macro_code
///
/// \author Rene Brun

void compile(){
   TCanvas *c1 = new TCanvas("c1");
   c1->Range(0,0,1,1);
   TPaveLabel *ptc = new TPaveLabel(0.02,0.42,0.2,0.58,"Compile");
   ptc->SetTextSize(0.40);
   ptc->SetFillColor(32);
   ptc->Draw();
   TPaveText *psub = new TPaveText(0.28,0.4,0.65,0.6);
   psub->Draw();
   TText *t2 = psub->AddText("Substitute some operators");
   TText *t3 = psub->AddText("to C++ style");
   TPaveLabel *panal = new TPaveLabel(0.73,0.42,0.98,0.58,"Analyze");
   panal->SetTextSize(0.40);
   panal->SetFillColor(42);
   panal->Draw();
   TArrow *ar1 = new TArrow(0.2,0.5,0.27,0.5,0.02,"|>");
   ar1->SetLineWidth(6);
   ar1->SetLineColor(4);
   ar1->Draw();
   TArrow *ar2 = new TArrow(0.65,0.5,0.72,0.5,0.02,"|>");
   ar2->SetLineWidth(6);
   ar2->SetLineColor(4);
   ar2->Draw();
}
