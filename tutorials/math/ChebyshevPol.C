// example of Chebyshev polynomials
// using new TFormula pre-defined definitions of chebyshev polynomials

void ChebyshevPol() {


   auto legend = new TLegend(0.88,0.4,1.,1.);

   int colors[] = { kRed, kRed+3, kMagenta, kMagenta+3, kBlue, kBlue+3, kCyan+3, kGreen, kGreen+3, kYellow, kOrange };
   
   for (int degree=0; degree <=10; ++degree) {
      auto f1 = new TF1("f1",TString::Format("cheb%d",degree),-1,1);
      // all parameters are zero apart from the one corresponding to the degree
      f1->SetParameter(degree,1);
      f1->SetLineColor( colors[degree]);
      f1->SetMinimum(-1.2);
      f1->SetMaximum(1.2);
      TString opt = (degree == 0) ? "" : "same";
      //f1->Print("V");
      f1->SetNpx(1000);
      f1->SetTitle("Chebyshev Polynomial");
      f1->Draw(opt);
      legend->AddEntry(f1,TString::Format("N=%d",degree),"L");
   }
   legend->Draw();
   
}
 
