/// \file
/// \ingroup tutorial_graphics
/// \notebook
/// Draw a pave text.
/// The text lines are added in order using the AddText method
/// Line separator can be added using AddLine
/// Once the TPaveText is build the text of each line can be retrieved as a
/// TText wich is useful to modify the text attributes of a line.
///
/// \macro_image
/// \macro_code
///
/// \author Olivier Couet

TCanvas *pavetext(){
   TCanvas *c = new TCanvas("c");
   TPaveText *pt = new TPaveText(.05,.1,.95,.8);

   pt->AddText("A TPaveText can contain severals line of text.");
   pt->AddText("They are added to the pave using the AddText method.");
   pt->AddLine(.0,.5,1.,.5);
   pt->AddText("Even complex TLatex formulas can be added:");
   pt->AddText("F(t) = #sum_{i=-#infty}^{#infty}A(i)cos#[]{#frac{i}{t+i}}");

   pt->Draw();

   TText *t = pt->GetLineWith("Even");
   t->SetTextColor(kOrange+1);

   return c;
}
