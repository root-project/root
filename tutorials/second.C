void second() {
   
   TCanvas *nut = new TCanvas("nut", "Second Session",100,10,750,1000);
   nut->Range(0,0,20,24);
   nut->SetFillColor(18);

   TPaveLabel *pl = new TPaveLabel(3,22,17,23.7,"My second ROOT interactive session","br");
   pl->SetFillColor(18);
   pl->Draw();

   TText t(0,0,"a");
   t.SetTextFont(62);
   t.SetTextSize(0.025);
   t.SetTextAlign(12);
   t.DrawText(2,21,"Very large C/C++ files can be interpreted (50,000 lines+).");
   t.DrawText(2,20,"Functions in macros can reference other functions, etc.");
   t.DrawText(2,19,"Let's make a file \"graph.C\" with the following statements:");

   TPaveText *macro = new TPaveText(2,11,18,18);
   macro->SetFillColor(10);
   macro->SetTextColor(kBlue);
   macro->SetBorderSize(6);
   macro->SetTextAlign(12);
   macro->SetTextFont(61);
   macro->AddText("{");
   macro->AddText("   TCanvas *c1 = new TCanvas(\"c1\",\"A Simple Graph Example\",200,10,700,500);");
   macro->AddText("   c1->Range(-0.5,-2,2.5,12);");
   macro->AddText("   const Int_t n = 20");
   macro->AddText("   Float_t x[n], y[n];");
   macro->AddText("   for (Int_t i=0;i<n;i++) {");
   macro->AddText("        x[i] = i*0.1;");
   macro->AddText("        y[i] = 10*sin(x[i]+0.2);");
   macro->AddText("   }");
   macro->AddText("   gr = new TGraph(n,x,y);");
   macro->AddText("   gr->Draw(\"AC*\");");
   macro->AddText("}");
   macro->AddText(" ");
   macro->Draw();

   t.SetTextFont(72);
   t.SetTextColor(kRed);
   t.SetTextSize(0.026);
   t.DrawText(3,10,"Root > .x graph.C");

   TPad *pad = new TPad("pad","pad",.15,.05,.85,.40);
   pad->SetFillColor(41);
   pad->SetFrameFillColor(33);
   pad->Draw();
   pad->cd();
   pad->SetGrid();
   pad->Range(-0.5,-2,2.5,12);
   const Int_t n = 20;
   Float_t x[n], y[n];
   for (Int_t i=0;i<n;i++) {
     x[i] = i*0.1;
     y[i] = 10*sin(x[i]+0.2);
   }
   gr = new TGraph(n,x,y);
   gr->Draw("AC*");
   nut->cd();
}
