// @(#)root/eve:$Id$
// Author: Matevz Tadel
// Demonstrates usage of TEveBoxSet class.

TEveBoxSet* boxset_test(Float_t x=0, Float_t y=0, Float_t z=0,
                        Int_t num=100, Bool_t register=kTRUE)
{
   TEveManager::Create();

   TRandom r(0);
   gStyle->SetPalette(1, 0);

   TEveRGBAPalette* pal = new TEveRGBAPalette(0, 130);

   TEveBoxSet* q = new TEveBoxSet("BoxSet");
   q->SetPalette(pal);
   q->Reset(TEveBoxSet::kBT_AABox, kFALSE, 64);
   for (Int_t i=0; i<num; ++i) {
      q->AddBox(r.Uniform(-10, 10), r.Uniform(-10, 10), r.Uniform(-10, 10),
                r.Uniform(0.2, 1),  r.Uniform(0.2, 1),  r.Uniform(0.2, 1));
      q->DigitValue(r.Uniform(0, 130));
   }
   q->RefitPlex();

   TEveTrans& t = q->RefMainTrans();
   t.SetPos(x, y, z);

   if (register)
   {
      gEve->AddElement(q);
      gEve->Redraw3D(kTRUE);
   }

   return q;
}



TEveBoxSet* cone_test(Float_t x=0, Float_t y=0, Float_t z=0, Int_t num=1000)
{
  using namespace TMath;

  TEveManager::Create();

  TRandom r(0);
  gStyle->SetPalette(1, 0);

  TEveRGBAPalette* pal = new TEveRGBAPalette(0, 500);

  TEveBoxSet* q = new TEveBoxSet("ConeSet");
  q->SetPalette(pal);
  q->Reset(TEveBoxSet::kBT_Cone, kFALSE, 64);

  Float_t a = 400;

  TEveVector dir, pos;
  Float_t theta, phi, height;

  for (Int_t i=0; i<num; ++i)
  {
    theta  = TMath::Pi()/4;
    if (i%2) theta = TMath::Pi()-theta;
    phi    = r.Uniform (-TMath::Pi(), TMath::Pi());
    height = r.Uniform(5, 15);

    pos.Set(r.Uniform(-a,a), r.Uniform(-a, a), r.Uniform(-a, a));
    dir.Set(Cos(phi)*Sin(theta), Sin(phi)*Sin(theta), Cos(theta));
    dir *= height;

    q->AddCone(pos, dir, r.Uniform(3, 5));
    q->DigitValue(r.Uniform(0, 500));
  }

  q->RefitPlex();
  TEveTrans& t = q->RefMainTrans();
  t.SetPos(x, y, z);

  if (r.Integer(2)>0) q->SetDrawConeCap(kTRUE);

  gEve->AddElement(q);
  gEve->Redraw3D(kTRUE);

  return q;
}
