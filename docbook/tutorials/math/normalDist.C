// tutorial illustrating the new statistical  distributions functions (pdf, cdf and quantile) 

#ifndef __CINT__
#include "Math/DistFunc.h"
#include "TF1.h"
#include "TCanvas.h"
#include "TSystem.h"
#include "TLegend.h"
#include "TAxis.h"
#endif



void normalDist() { 

#ifdef __CINT__
   gSystem->Load("libMathCore");
   gSystem->Load("libMathMore");
#endif

   TF1 *pdfunc  = new TF1("pdf","ROOT::Math::normal_pdf(x, [0],[1])",-5,5);
   TF1 *cdfunc  = new TF1("cdf","ROOT::Math::normal_cdf(x, [0],[1])",-5,5);
   TF1 *ccdfunc = new TF1("cdf_c","ROOT::Math::normal_cdf_c(x, [0])",-5,5);
   TF1 *qfunc   = new TF1("quantile","ROOT::Math::normal_quantile(x, [0])",0,1);
   TF1 *cqfunc  = new TF1("quantile_c","ROOT::Math::normal_quantile_c(x, [0])",0,1);

   pdfunc->SetParameters(1.0,0.0);  // set sigma to 1 and mean to zero
   pdfunc->SetTitle("");
   pdfunc->SetLineColor(kBlue);

   pdfunc->GetXaxis()->SetLabelSize(0.06);
   pdfunc->GetXaxis()->SetTitle("x");
   pdfunc->GetXaxis()->SetTitleSize(0.07);
   pdfunc->GetXaxis()->SetTitleOffset(0.55);
   pdfunc->GetYaxis()->SetLabelSize(0.06);
 




   cdfunc->SetParameters(1.0,0.0);  // set sigma to 1 and mean to zero
   cdfunc->SetTitle("");
   cdfunc->SetLineColor(kRed);

   cdfunc->GetXaxis()->SetLabelSize(0.06);
   cdfunc->GetXaxis()->SetTitle("x");
   cdfunc->GetXaxis()->SetTitleSize(0.07);
   cdfunc->GetXaxis()->SetTitleOffset(0.55);
   
   cdfunc->GetYaxis()->SetLabelSize(0.06);
   cdfunc->GetYaxis()->SetTitle("p");
   cdfunc->GetYaxis()->SetTitleSize(0.07);
   cdfunc->GetYaxis()->SetTitleOffset(0.55);
 
   ccdfunc->SetParameters(1.0,0.0);  // set sigma to 1 and mean to zero
   ccdfunc->SetTitle("");
   ccdfunc->SetLineColor(kGreen);

   qfunc->SetParameter(0, 1.0);  // set sigma to 1 
   qfunc->SetTitle("");
   qfunc->SetLineColor(kRed);
   qfunc->SetNpx(1000); // to get more precision for p close to 0 or 1

   qfunc->GetXaxis()->SetLabelSize(0.06);
   qfunc->GetXaxis()->SetTitle("p");
   qfunc->GetYaxis()->SetLabelSize(0.06);
   qfunc->GetXaxis()->SetTitleSize(0.07);
   qfunc->GetXaxis()->SetTitleOffset(0.55);
   qfunc->GetYaxis()->SetTitle("x");
   qfunc->GetYaxis()->SetTitleSize(0.07);
   qfunc->GetYaxis()->SetTitleOffset(0.55);

   cqfunc->SetParameter(0, 1.0);  // set sigma to 1 
   cqfunc->SetTitle("");
   cqfunc->SetLineColor(kGreen);
   cqfunc->SetNpx(1000); 
   

   TCanvas * c1 = new TCanvas("c1","Normal Distributions",100,10,600,800);

   c1->Divide(1,3);
   c1->cd(1);

   pdfunc->Draw();
   TLegend *legend1 = new TLegend(0.583893,0.601973,0.885221,0.854151);
   legend1->AddEntry(pdfunc,"normal_pdf","l");
   legend1->Draw();

   c1->cd(2);
   cdfunc->Draw();
   ccdfunc->Draw("same");
   TLegend *legend2 = new TLegend(0.585605,0.462794,0.886933,0.710837);
   legend2->AddEntry(cdfunc,"normal_cdf","l");
   legend2->AddEntry(ccdfunc,"normal_cdf_c","l");
   legend2->Draw();

   c1->cd(3);
   qfunc->Draw();
   cqfunc->Draw("same");
   TLegend *legend3 = new TLegend(0.315094,0.633668,0.695179,0.881711);
   legend3->AddEntry(qfunc,"normal_quantile","l");
   legend3->AddEntry(cqfunc,"normal_quantile_c","l");
   legend3->Draw();


   
}
