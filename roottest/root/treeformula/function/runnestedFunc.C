bool runnestedFunc() {
   TF1 *feta1 = new TF1("feta1", "TMath::Power(TMath::CosH(x), 2.0)",
	-5.4, 5.4);
   TF1 *feta2 = new TF1("feta2", "(TMath::CosH(x))*(TMath::CosH(x))",
	-5.4, 5.4);
   cout <<feta1->Eval(2.0) << endl;
   cout <<feta2->Eval(2.0) << endl;
   double a = feta1->Eval(2.0);
   double b = feta2->Eval(2.0);
   Bool_t result = (TMath::Abs( a-b )/a) < 1e-7; 
   return !(result);
}


