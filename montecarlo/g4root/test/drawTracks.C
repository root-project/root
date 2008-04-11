void drawTracks()
{
// Draw simulated tracks on top of the geometry.
   TGeoManager::Import("ex06geom.root");
   gGeoManager->GetTopVolume()->SetVisContainers();
   gGeoManager->SetTopVisible();
   TCanvas *c1 = new TCanvas("c1", "Tank with bubble", 800,600);
   TCanvas *c2 = new TCanvas("c2", "YZ plots", 800,600);
   c2->Divide(2,1,0.01,0.01);
   c1->cd();
   gGeoManager->GetTopVolume()->Draw("ogl");
   TFile file1("tracks_tgeo.root");
   if (file1.IsZombie()) return;
   TFile file2("tracks_g4.root");
   if (file2.IsZombie()) return;
   TObjArray *tracks1 = (TObjArray*)file1.Get("tracks");
   TObjArray *tracks2 = (TObjArray*)file2.Get("tracks");
   file1.Close();
   file2.Close();
   Int_t ntracks1 = tracks1->GetEntries();
   Int_t ntracks2 = tracks2->GetEntries();
   TPolyLine3D *track;
   TH2F *hist1 = new TH2F("steps_tgeo", "Step coordinates TGeo", 100, -800.,800., 100, -800.,800.);
   TH2F *hist2 = new TH2F("steps_g4", "Step coordinates G4", 100, -800.,800., 100, -800.,800.);
   Int_t i,j,n;
   Float_t *arr;
   for (i=0; i<ntracks1; i++) {
      track = (TPolyLine3D*)tracks1->At(i);
      n = track->GetLastPoint();
      arr = track->GetP();
      for (j=0; j<=n; j++) hist1->Fill(arr[3*j+1],arr[3*j+2]);
      track->Draw("SAME");
   }
   for (i=0; i<ntracks2; i++) {
      track = (TPolyLine3D*)tracks2->At(i);
      n = track->GetLastPoint();
      arr = track->GetP();
      for (j=0; j<=n; j++) hist2->Fill(arr[3*j+1],arr[3*j+2]);
   }

   c2->cd(1);
   hist1->Draw();
   c2->cd(2);
   hist2->Draw(); 
}
      
