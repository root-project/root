{
   Int_t nx = hpxpy->GetXaxis()->GetNbins();
   Int_t ny = hpxpy->GetYaxis()->GetNbins();
   Int_t binxmin = nx*xslider->GetMinimum();
   Int_t binxmax = nx*xslider->GetMaximum();
   hpxpy->GetXaxis()->SetRange(binxmin,binxmax);
   Int_t binymin = ny*yslider->GetMinimum();
   Int_t binymax = ny*yslider->GetMaximum();
   hpxpy->GetYaxis()->SetRange(binymin,binymax);
   pad->cd();
   pad->Modified();
   //hpxpy->Draw(hpxpy->GetDrawOption());
   c1->Update();
}
