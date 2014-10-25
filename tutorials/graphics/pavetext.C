//Draw a pave text
//Author: Olivier Couet
TCanvas *pavetext(){
   TCanvas *c = new TCanvas("c");
   TPaveText *pt = new TPaveText(.05,.1,.95,.8);

   pt->AddText("A TPaveText can contain severals line of text.");
   pt->AddText("They are added to the pave using the AddText method.");
   pt->AddLine(.0,.5,1.,.5);
   pt->AddText("Even complex TLatex formulas can be added:");
   pt->AddText("F(t) = #sum_{i=-#infty}^{#infty}A(i)cos#[]{#frac{i}{t+i}}");

   pt->Draw();
   return c;
}
