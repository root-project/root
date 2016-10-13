/// \file
/// \ingroup tutorial_graphics
/// \notebook
/// Draw a pave text.
/// The text lines are added in order using the AddText method
/// Line separator can be added using AddLine.
///
/// AddText returns a TText corresponding to the line added to the pave. This
/// return value can be used to modify the text attributes.
///
/// Once the TPaveText is build the text of each line can be retrieved as a
/// TText with GetLine and GetLineWith wich is also useful to modify the text
/// attributes of a line.
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
   TText *t1 = pt->AddText("F(t) = #sum_{i=-#infty}^{#infty}A(i)cos#[]{#frac{i}{t+i}}");

   t1->SetTextColor(kBlue);

   pt->Draw();

   TText *t2 = pt->GetLineWith("Even");
   t2->SetTextColor(kOrange+1);

   return c;
}
