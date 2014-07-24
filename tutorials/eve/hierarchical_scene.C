const Int_t Ns = 7;

void add_blobs(TEveElement* p, Float_t rad, Float_t height, Float_t size,
               Int_t level)
{
  if (level <= 0) return;

  for (Int_t i = 0; i < Ns; ++i)
  {
    TEveGeoShape* x = new TEveGeoShape("SS");
    x->SetShape(new TGeoSphere(0, size));
    Double_t phi = TMath::TwoPi() * i / Ns;
    x->RefMainTrans().SetPos(rad*TMath::Cos(phi),
                             rad*TMath::Sin(phi),
                             height);
    x->SetMainColor(TColor::GetColorPalette
                    (gRandom->Integer(TColor::GetNumberOfColors())));
    p->AddElement(x);

    add_blobs(x, 0.8 * rad, 0.8 * height, 0.8 * size, level - 1);
  }
}

void hierarchical_scene()
{
  TEveManager::Create();

  TColor::SetPalette(1, 0);
  gRandom = new TRandom3(0);

  TEveScene *s = gEve->SpawnNewScene("Hierachical Scene", "OoogaDooga");
  s->SetHierarchical(kTRUE);

  gEve->GetDefaultViewer()->AddScene(s);

  add_blobs(s, 6, 4, 0.5, 4);

  gEve->Redraw3D(kTRUE);
}
