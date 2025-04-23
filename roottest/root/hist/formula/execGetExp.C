{
   TF1 *f = new TF1("f1","[0]+[1]*cos(x*TMath::DegToRad())+[2]*sin(x*TMath::DegToRad())");
   cout << f->GetExpFormula() << endl;
#ifdef ClingWorkAroundBrokenUnnamedReturn
   int res = 0;
#else
   return 0;
#endif
}

