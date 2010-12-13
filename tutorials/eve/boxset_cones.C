// @(#)root/eve:$Id$
// Author: Alja Mrak-Tadel

// Demonstrates usage of 'cone' mode in TEveBoxSet class.

TEveBoxSet* boxset_cones(Float_t x=0, Float_t y=0, Float_t z=0,
		      Int_t num=100, Bool_t register=kTRUE)
{
  TEveManager::Create();

  using namespace TMath;

  TEveStraightLineSet* lines = new TEveStraightLineSet("StraightLines");
  lines->SetLineColor(kYellow);
  lines->SetLineWidth(2);

  TRandom r(0);
  gStyle->SetPalette(1, 0);
  TEveRGBAPalette* pal = new TEveRGBAPalette(0, 500);
  TEveBoxSet* cones = new TEveBoxSet("ConeSet");
  cones->SetPalette(pal);
  cones->Reset(TEveBoxSet::kBT_Cone, kFALSE, 64);

  Float_t a = 40; // max distance between cones
  TEveVector dir, pos;
  Float_t theta, phi, height, rad;
  for (Int_t i=0; i<num; ++i)
  {
    theta  = r.Uniform(0,TMath::Pi());
    phi    = r.Uniform (-TMath::Pi(), TMath::Pi());
    height = r.Uniform(5, 15);
    rad    = r.Uniform(3, 5);
    dir.Set(Cos(phi)*Cos(theta), Sin(phi)*Cos(theta), Sin(theta));
    dir *= height;
    pos.Set(r.Uniform(-a,a), r.Uniform(-a, a), r.Uniform(-a, a));

    cones->AddCone(pos, dir, rad);
    cones->DigitValue(r.Uniform(0, 500));

    // draw axis line 30% longer than cone height
    TEveVector end = pos + dir*1.3;
    lines->AddLine(pos.fX, pos.fY, pos.fZ, end.fX, end.fY, end.fZ);
  }

  // by default cone cap not drawn
  if (r.Integer(2)>0)  cones->SetDrawConeCap(kTRUE);

  cones->RefitPlex();
  TEveTrans& t = cones->RefMainTrans();
  t.SetPos(x, y, z);

  gEve->AddElement(cones);
  gEve->AddElement(lines);

  gEve->Redraw3D(kTRUE);

  return cones;
}

TEveBoxSet*
elliptic_boxset_cones(Float_t x=0, Float_t y=0, Float_t z=0,
                   Int_t num=100, Bool_t register=kTRUE)
{
  TEveManager::Create();

  using namespace TMath;

  TEveManager::Create();

  TEveStraightLineSet* lines = new TEveStraightLineSet("StraightLines");
  lines->SetLineColor(kYellow);
  lines->SetLineWidth(2);

  TRandom r(0);

  TEveBoxSet* cones = new TEveBoxSet("EllipticConeSet");
  cones->Reset(TEveBoxSet::kBT_EllipticCone, kTRUE, 64);

  cones->SetPickable(kTRUE);

  Float_t a = 40; // max distance between cones
  TEveVector dir, pos;
  Float_t theta, phi, height, rad;
  for (Int_t i=0; i<num; ++i)
  {
    theta  = r.Uniform(0,TMath::Pi());
    phi    = r.Uniform (-TMath::Pi(), TMath::Pi());
    height = r.Uniform(5, 15);
    rad    = r.Uniform(3, 5);
    dir.Set(Cos(phi)*Cos(theta), Sin(phi)*Cos(theta), Sin(theta));
    dir *= height;
    pos.Set(r.Uniform(-a,a), r.Uniform(-a, a), r.Uniform(-a, a));

    cones->AddEllipticCone(pos, dir, rad, 0.5*rad, r.Uniform(0,360));
    cones->DigitColor(r.Uniform(20, 255), r.Uniform(20, 255),
                      r.Uniform(20, 255), r.Uniform(20, 255));

    // draw axis line 30% longer than cone height
    TEveVector end = pos + dir*1.3;
    lines->AddLine(pos.fX, pos.fY, pos.fZ, end.fX, end.fY, end.fZ);
  }

  // by default cone cap not drawn
  if (r.Integer(2)>0)  cones->SetDrawConeCap(kTRUE);

  cones->RefitPlex();
  TEveTrans& t = cones->RefMainTrans();
  t.SetPos(x, y, z);

  gEve->AddElement(cones);
  gEve->AddElement(lines);

  gEve->Redraw3D(kTRUE);

  return cones;
}
