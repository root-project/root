{
   //Example of macro featuring two sliders
   TFile *f = new TFile("hsimple.root");
   TH2F *hpxpy = (TH2F*)f->Get("hpxpy");
   TCanvas *c1 = new TCanvas("c1");
   TPad *pad = new TPad("pad","lego pad",0.1,0.1,0.98,0.98);
   pad->SetFillColor(33);
   pad->Draw();
   pad->cd();
   gStyle->SetFrameFillColor(42);
   hpxpy->SetFillColor(46);
   hpxpy->Draw("lego1");
   c1->cd();

   //Create two sliders in main canvas. When button1 will be released
   //the macro xysliderAction.C will be called.
   TSlider *xslider = new TSlider("xslider","x",0.1,0.02,0.98,0.08);
   xslider->SetMethod(".x xysliderAction.C");
   TSlider *yslider = new TSlider("yslider","y",0.02,0.1,0.06,0.98);
   yslider->SetMethod(".x xysliderAction.C");
}

