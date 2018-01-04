void pal() {
   TCanvas *can = new TCanvas("can","can", 600,400);
   TH2F *hcontz = new TH2F("hcontz","example", 40,-4,4, 40, -20, 20);
   Float_t px, py;
   for (int i=0;i<25000;++i) {
      gRandom->Rannor(px,py);
      hcontz->Fill(px-1, 1.5*py);
      hcontz->Fill(2+0.5*px, 2*py-10,0.1);
   }
   Double_t R[3] = {1,0,0};
   Double_t G[3] = {0,1,0};
   Double_t B[3] = {1,0,1};
   Double_t L[3] = {0,0.5,1};
   Int_t n= 50;
   TColor::CreateGradientColorTable(3,L,R,G,B,n);
   // TColor::CreateGradientColorTable(3,L,R,G,B,n);
   hcontz->Draw("COLZ");
   
   can->Modified();
   can->Update();
   
   //can->Draw();
   
   printf("Defined %d\n", TColor::DefinedColors());

   //can->SaveAs("can.json");
   //can->SaveAs("can.png");
   //can->SaveAs("can2.json");
}
