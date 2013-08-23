{
  TFile *_file0 = TFile::Open("run133.root");
  TTree *t; _file0->GetObject("t",t);
  TH2F *h2 = new TH2F("h2","h2",5,0,5,100,0,100);
  t->Draw("ThetaDeg[ringtouche_DE1]:ringtouche_DE1>>h2","","colz goff");
  if ( (int)(h2->GetMean(1)*1000) != 3) {
     printf("Error: x mean is %f rather than 0.003\n",h2->GetMean(1));
     return 1;
  } 
  if ( (int)(h2->GetMean(2)*10) != 0) {
     printf("Error: y mean is %f rather than 0\n",h2->GetMean(2));
     return 1;
  }
  return 0;
}

  
