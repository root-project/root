// example illustrating divided pads and Latex
// Author: Rene Brun
void quarks () {
   TCanvas *c1 = new TCanvas("c1", "c1",10,10,630,760);
   c1->SetFillColor(kBlack);
   Int_t quarkColor  = 50;
   Int_t leptonColor = 16;
   Int_t forceColor  = 38;
   Int_t titleColor  = kYellow;
   Int_t border = 8;
   
   TLatex *texf = new TLatex(0.90,0.455,"Force Carriers");
   texf->SetTextColor(forceColor);
   texf->SetTextAlign(22); texf->SetTextSize(0.07); 
   texf->SetTextAngle(90);
   texf->Draw();
   
   TLatex *texl = new TLatex(0.11,0.288,"Leptons");
   texl->SetTextColor(leptonColor);
   texl->SetTextAlign(22); texl->SetTextSize(0.07); 
   texl->SetTextAngle(90);
   texl->Draw();
   
   TLatex *texq = new TLatex(0.11,0.624,"Quarks");
   texq->SetTextColor(quarkColor);
   texq->SetTextAlign(22); texq->SetTextSize(0.07); 
   texq->SetTextAngle(90);
   texq->Draw();
   
   TLatex tex(0.5,0.5,"u");
   tex.SetTextColor(titleColor); tex.SetTextFont(32); 
   tex.SetTextAlign(22);
   tex.SetTextSize(0.14); 
   tex.DrawLatex(0.5,0.93,"Elementary");
   tex.SetTextSize(0.12); 
   tex.DrawLatex(0.5,0.84,"Particles");
   tex.SetTextSize(0.05); 
   tex.DrawLatex(0.5,0.067,"Three Generations of Matter");

   tex.SetTextColor(kBlack); tex.SetTextSize(0.8);
         
// ------------>Create main pad and its subdivisions
   TPad *pad = new TPad("pad", "pad",0.15,0.11,0.85,0.79);
   pad->Draw();
   pad->cd();
   pad->Divide(4,4,0.0003,0.0003);
  
   pad->cd(1); gPad->SetFillColor(quarkColor);   
   gPad->SetBorderSize(border);
   tex.DrawLatex(.5,.5,"u");

   pad->cd(2); gPad->SetFillColor(quarkColor);   
   gPad->SetBorderSize(border);
   tex.DrawLatex(.5,.5,"c");
  
   pad->cd(3); gPad->SetFillColor(quarkColor);   
   gPad->SetBorderSize(border);
   tex.DrawLatex(.5,.5,"t");
  
   pad->cd(4); gPad->SetFillColor(forceColor);   
   gPad->SetBorderSize(border);
   tex.DrawLatex(.5,.55,"#gamma");
  
   pad->cd(5); gPad->SetFillColor(quarkColor);   
   gPad->SetBorderSize(border);
   tex.DrawLatex(.5,.5,"d");
  
   pad->cd(6); gPad->SetFillColor(quarkColor);   
   gPad->SetBorderSize(border);
   tex.DrawLatex(.5,.5,"s");
  
   pad->cd(7); gPad->SetFillColor(quarkColor);   
   gPad->SetBorderSize(border);
   tex.DrawLatex(.5,.5,"b");
  
   pad->cd(8); gPad->SetFillColor(forceColor);   
   gPad->SetBorderSize(border);
   tex.DrawLatex(.5,.55,"g");
  
   pad->cd(9); gPad->SetFillColor(leptonColor);  
   gPad->SetBorderSize(border);
   tex.DrawLatex(.5,.5,"#nu_{e}");
  
   pad->cd(10); gPad->SetFillColor(leptonColor); 
   gPad->SetBorderSize(border);
   tex.DrawLatex(.5,.5,"#nu_{#mu}");
  
   pad->cd(11); gPad->SetFillColor(leptonColor); 
   gPad->SetBorderSize(border);
   tex.DrawLatex(.5,.5,"#nu_{#tau}");
  
   pad->cd(12); gPad->SetFillColor(forceColor);  
   gPad->SetBorderSize(border);
   tex.DrawLatex(.5,.5,"Z");
  
   pad->cd(13); gPad->SetFillColor(leptonColor); 
   gPad->SetBorderSize(border);
   tex.DrawLatex(.5,.5,"e");
  
   pad->cd(14); gPad->SetFillColor(leptonColor); 
   gPad->SetBorderSize(border);
   tex.DrawLatex(.5,.56,"#mu");
  
   pad->cd(15); gPad->SetFillColor(leptonColor); 
   gPad->SetBorderSize(border);
   tex.DrawLatex(.5,.5,"#tau");
  
   pad->cd(16); gPad->SetFillColor(forceColor);  
   gPad->SetBorderSize(border);
   tex.DrawLatex(.5,.5,"W");
  
   c1->cd();
}
