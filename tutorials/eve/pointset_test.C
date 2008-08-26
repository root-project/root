// Author: Matevz Tadel

TEvePointSet* pointset_test(Int_t npoints = 512)
{
   TEveManager::Create();

   TRandom r(0);
   Float_t s = 100;

   TEvePointSet* ps = new TEvePointSet();
   ps->SetOwnIds(kTRUE);

   for(Int_t i = 0; i<npoints; i++)
   {
      ps->SetNextPoint(r.Uniform(-s,s), r.Uniform(-s,s), r.Uniform(-s,s));
      ps->SetPointId(new TNamed(Form("Point %d", i), ""));
   }

   ps->SetMarkerColor(5);
   ps->SetMarkerSize(1.5);
   ps->SetMarkerStyle(4);

   gEve->AddElement(ps);
   gEve->Redraw3D();

   return ps;
}

TEvePointSetArray* pointsetarray_test()
{
   TEveManager::Create();

   TRandom r(0);

   TEvePointSetArray* l = new TEvePointSetArray("TPC hits - Charge Slices", "");
   l->SetSourceCS(TEvePointSelectorConsumer::kTVT_RPhiZ);
   l->SetMarkerColor(3);
   l->SetMarkerStyle(4); // antialiased circle
   l->SetMarkerSize(0.8);

   gEve->AddElement(l);
   l->InitBins("Charge", 9, 10, 100);

   TColor::SetPalette(1, 0); // Spectrum palette
   const Int_t nCol = TColor::GetNumberOfColors();
   for (Int_t i = 1; i <= 9; ++i)
      l->GetBin(i)->SetMainColor(TColor::GetColorPalette(i * nCol / 10));

   l->GetBin(0) ->SetMainColor(kGray);
   l->GetBin(10)->SetMainColor(kWhite);

   Double_t rad, phi, z;
   for (Int_t i = 0; i < 10000; ++i)
   {
      rad = r.Uniform(60, 180);
      phi = r.Uniform(0, TMath::TwoPi());
      z   = r.Uniform(-250, 250);
      l->Fill(rad*TMath::Cos(phi), rad*TMath::Sin(phi), z,
              r.Uniform(0, 110));
   }

   l->CloseBins();

   gEve->Redraw3D();

}
