#ifdef ClingWorkAroundBrokenUnnamedReturn
int execOutOfBounds() {
#else
{
#endif
  TFile *_file0 = TFile::Open("run133.root");
  auto t = _file0->Get<TTree>("t");
  // ThetaDeg is a dynamic array with indices going from 0 to (ringtouche_DE1 - 1)
  TTreeFormula tf("tf", "ThetaDeg", t); 
  t->GetEntry(0); // ringtouche for entry 0 is 0
  auto res = tf.EvalInstance(1); // ThetaDeg[1] goes out of bonds
  if (!TMath::IsNaN(res)) {
     printf("Error: evaluated instance is %f rather than NaN\n", res);
     return 1;
  }
  TH2F *h2 = new TH2F("h2","h2",5,0,5,100,0,100);
  t->Draw("ThetaDeg[ringtouche_DE1]:ringtouche_DE1>>h2","","colz goff");
  // Filling NaNs increases entry number but does not change any bin content
  if ( (int)(h2->GetMean(1)*1000) != 0) {
     printf("Error: x mean is %f rather than 0.000\n",h2->GetMean(1));
     return 1;
  } 
  if ( (int)(h2->GetMean(2)*10) != 0) {
     printf("Error: y mean is %f rather than 0\n",h2->GetMean(2));
     return 1;
  }
  return 0;
}

  
