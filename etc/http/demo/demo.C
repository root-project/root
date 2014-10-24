// macro used to generate several JSON files,
// used to demonstrate online features of JSROOT
// Files used in demo.htm page.
// In real application JSON data could be produced by THttpServer on-the-fly

void demo() {

   TH1I* h1 = new TH1I("histo1","histo title", 100, 0., 100.);
   // h1->FillRandom("gaus",10000);

   TCanvas *c1 = new TCanvas("c1","Dynamic Filling Example",200,10,700,500);

   for (int n=0;n<20;n++) {
      double ampl1 = 15000 + 10000 * cos(n/20.*2.*TMath::Pi());
      double ampl2 = 15000 + 10000 * sin(n/20.*2.*TMath::Pi());

      h1->Reset();

      for (int cnt=0;cnt<ampl1;cnt++)
         h1->Fill(gRandom->Gaus(25,10));

      for (int cnt=0;cnt<ampl2;cnt++)
         h1->Fill(gRandom->Gaus(75,10));

      if (n==0) h1->Draw();
      c1->Modified();
      c1->Update();

      TString json = TBufferJSON::ConvertToJSON(h1);

      FILE* f = fopen(Form("root%d.json",n), "w");
      fputs(json.Data(),f);
      fclose(f);

      // gSystem->Exec(Form("gzip root%d.json",n));

   }



}
