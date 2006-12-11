void timeonaxis2() {
   // Define the time offset as 2003, January 1st
   //Author: Olivier Couet
   
   TDatime T0(2003,01,01,00,00,00);
   int X0 = T0.Convert();
   gStyle->SetTimeOffset(X0);
   
   // Define the lowest histogram limit as 2002, September 23rd
   TDatime T1(2002,09,23,00,00,00);
   int X1 = T1.Convert()-X0;

   // Define the highest histogram limit as 2003, March 7th
   TDatime T2(2003,03,07,00,00,00);
   int X2 = T2.Convert(1)-X0;

   TH1F * h1 = new TH1F("h1","test",100,X1,X2);      

   TRandom r;
   for (Int_t i=0;i<30000;i++) {
      Double_t noise = r.Gaus(0.5*(X1+X2),0.1*(X2-X1));
      h1->Fill(noise);
   }
   
   h1->GetXaxis()->SetTimeDisplay(1);
   h1->GetXaxis()->SetLabelSize(0.03);
   h1->GetXaxis()->SetTimeFormat("%Y\/%m\/%d");
   h1->Draw();
}
