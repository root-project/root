/// \file
/// \ingroup tutorial_eve_7
/// Demonstrates usage of 'cone' mode in REveBoxSet class.
///
/// \image html eve_boxset_cones.png
/// \macro_code
///
/// \author Alja Mrak-Tadel

using namespace ROOT::Experimental;

REveBoxSet *elliptic_boxset_cones(Float_t x = 0, Float_t y = 0, Float_t z = 0, Int_t num = 100)
{
   auto eveMng = REveManager::Create();

   using namespace TMath;

   REveManager::Create();

   auto lines = new REveStraightLineSet("StraightLines");
   lines->SetLineColor(kYellow);
   lines->SetLineWidth(2);

   TRandom r(0);

   auto cones = new REveBoxSet("EllipticConeSet");
   bool valIsColor = true;
   cones->Reset(REveBoxSet::kBT_InstancedScaledRotated, valIsColor, 64);

   cones->SetPickable(kTRUE);

   Float_t a = 40; // max distance between cones
   REveVector dir, pos;
   Float_t theta, phi, height, rad;
   for (Int_t i = 0; i < num; ++i) {
      theta = r.Uniform(0, TMath::Pi());
      phi = r.Uniform(-TMath::Pi(), TMath::Pi());
      height = r.Uniform(5, 15);
      rad = r.Uniform(3, 5);
      dir.Set(Cos(phi) * Cos(theta), Sin(phi) * Cos(theta), Sin(theta));
      dir *= height;
      pos.Set(r.Uniform(-a, a), r.Uniform(-a, a), r.Uniform(-a, a));

      cones->AddEllipticCone(pos, dir, rad, 0.5 * rad, r.Uniform(0, 360));
      if (i % 2 == 0)
         cones->DigitColor(kRed);
      else
         cones->DigitColor(kGreen);
      // draw axis line 30% longer than cone height
      REveVector end = pos + dir * 1.3f;
      lines->AddLine(pos.fX, pos.fY, pos.fZ, end.fX, end.fY, end.fZ);
   }

   // by default cone cap not drawn
   if (r.Integer(2) > 0)
      cones->SetDrawConeCap(kTRUE);

   cones->RefitPlex();
   REveTrans &t = cones->RefMainTrans();
   t.SetPos(x, y, z);
   cones->SetPickable(1);
   cones->SetAlwaysSecSelect(1);

   eveMng->GetEventScene()->AddElement(cones);
   eveMng->GetEventScene()->AddElement(lines);

   REveViewer *v = (REveViewer *)eveMng->GetViewers()->FirstChild();
   v->SetAxesType(REveViewer::kAxesOrigin);

   eveMng->Show();
   return cones;
}

REveBoxSet *boxset_cones(Float_t x = 0, Float_t y = 0, Float_t z = 0, Int_t num = 100)
{
   auto eveMng = REveManager::Create();

   using namespace TMath;

   auto lines = new REveStraightLineSet("StraightLines");
   lines->SetLineColor(kYellow);
   lines->SetLineWidth(2);

   TRandom r(0);
   auto pal = new REveRGBAPalette(0, 500);
   auto cones = new REveBoxSet("ConeSet");
   cones->SetPalette(pal);
   bool valIsColor = false;
   cones->Reset(REveBoxSet::kBT_InstancedScaledRotated, valIsColor, 64);

   Float_t a = 40; // max distance between cones
   REveVector dir, pos;
   Float_t theta, phi, height, rad;
   for (Int_t i = 0; i < num; ++i) {
      theta = r.Uniform(0, TMath::Pi());
      phi = r.Uniform(-TMath::Pi(), TMath::Pi());
      height = r.Uniform(5, 15);
      rad = r.Uniform(3, 5);
      dir.Set(Cos(phi) * Cos(theta), Sin(phi) * Cos(theta), Sin(theta));
      dir *= height;
      pos.Set(r.Uniform(-a, a), r.Uniform(-a, a), r.Uniform(-a, a));

      cones->AddCone(pos, dir, rad);
      cones->DigitValue(r.Uniform(0, 500));

      // draw axis line 30% longer than cone height
      REveVector end = pos + dir * 1.3f;
      lines->AddLine(pos.fX, pos.fY, pos.fZ, end.fX, end.fY, end.fZ);
   }

   cones->SetPickable(1);
   cones->SetAlwaysSecSelect(1);
   // by default cone cap not drawn
   if (r.Integer(2) > 0)
      cones->SetDrawConeCap(kTRUE);

   cones->RefitPlex();
   REveTrans &t = cones->RefMainTrans();
   t.SetPos(x, y, z);

   eveMng->GetEventScene()->AddElement(cones);
   eveMng->GetEventScene()->AddElement(lines);
   eveMng->GetEventScene()->AddElement(pal);
   eveMng->Show();

   return cones;
}
