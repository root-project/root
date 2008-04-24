// @(#)root/eve:$Id$
// Author: Matevz Tadel

TEveQuadSet* quad_test(Float_t x=0, Float_t y=0, Float_t z=0,
                       Int_t num=100, Bool_t register=kTRUE)
{
   TEveManager::Create();

   TRandom r(0);
   gStyle->SetPalette(1, 0);

   TEveRGBAPalette* pal = new TEveRGBAPalette(0, 130);

   TEveQuadSet* q = new TEveQuadSet("RectangleXY");
   q->SetPalette(pal);
   q->Reset(TEveQuadSet::kQT_RectangleXY, kFALSE, 32);
   for (Int_t i=0; i<num; ++i) {
      q->AddQuad(r.Uniform(-10, 10), r.Uniform(-10, 10), r.Uniform(-10, 10),
                 r.Uniform(0.2, 1), r.Uniform(0.2, 1));
      q->QuadValue(r.Uniform(0, 130));
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

TEveQuadSet* quad_test_emc(Float_t x=0, Float_t y=0, Float_t z=0,
                                Int_t num=100)
{
   TEveManager::Create();

   TRandom r(0);
   gStyle->SetPalette(1, 0);

   TEveQuadSet* q = new TEveQuadSet("EMC Supermodule");
   q->SetOwnIds(kTRUE);
   q->Reset(TEveQuadSet::kQT_RectangleXZFixedDimY, kFALSE, 32);
   q->SetDefWidth(8);
   q->SetDefHeight(8);

   for (Int_t i=0; i<num; ++i) {
      q->AddQuad(r.Uniform(-100, 100), r.Uniform(-100, 100));
      q->QuadValue(r.Uniform(0, 130));
      q->AddId(new TNamed(Form("Cell %d", i)));
   }
   q->RefitPlex();

   TEveTrans& t = q->RefMainTrans();
   t.SetPos(x, y, z);

   gEve->AddElement(q);
   gEve->Redraw3D();

   return q;
}

TEveQuadSet* quad_test_circ()
{
   TEveManager::Create();

   TRandom r(0);
   gStyle->SetPalette(1, 0);

   TEveQuadSet* q = new TEveQuadSet("Pepe");
   q->Reset(TEveQuadSet::kQT_RectangleXY, kFALSE, 32);

   Float_t R = 10, dW = 1, dH = .5;
   for (Int_t i=0; i<12; ++i) {
      Float_t x = R * TMath::Cos(TMath::TwoPi()*i/12);
      Float_t y = R * TMath::Sin(TMath::TwoPi()*i/12);
      q->AddQuad(x-dW, y-dH, r.Uniform(-1, 1), 2*dW, 2*dH);
      q->QuadValue(r.Uniform(0, 130));
   }
   q->RefitPlex();

   TEveTrans& t = q->RefMainTrans();
   t.SetPos(0, 0, 300);

   gEve->AddElement(q);
   gEve->Redraw3D();

   return q;
}

TEveQuadSet* quad_test_hex(Float_t x=0, Float_t y=0, Float_t z=0,
                                Int_t num=100, Bool_t register=kTRUE)
{
   TEveManager::Create();

   TRandom r(0);
   gStyle->SetPalette(1, 0);

   {
      TEveQuadSet* q = new TEveQuadSet("HexagonXY");
      q->Reset(TEveQuadSet::kQT_HexagonXY, kFALSE, 32);
      for (Int_t i=0; i<num; ++i) {
         q->AddHexagon(r.Uniform(-10, 10), r.Uniform(-10, 10), r.Uniform(-10, 10),
                       r.Uniform(0.2, 1));
         q->QuadValue(r.Uniform(0, 120));
      }
      q->RefitPlex();

      TEveTrans& t = q->RefMainTrans();
      t.SetPos(x, y, z);

      if (register)
      {
         gEve->AddElement(q);
         gEve->Redraw3D();
      }
   }

   {
      TEveQuadSet* q = new TEveQuadSet("HexagonYX");
      q->Reset(TEveQuadSet::kQT_HexagonYX, kFALSE, 32);
      for (Int_t i=0; i<num; ++i) {
         q->AddHexagon(r.Uniform(-10, 10), r.Uniform(-10, 10), r.Uniform(-10, 10),
                       r.Uniform(0.2, 1));
         q->QuadValue(r.Uniform(0, 120));
      }
      q->RefitPlex();

      TEveTrans& t = q->RefMainTrans();
      t.SetPos(x, y, z);

      if (register)
      {
         gEve->AddElement(q);
         gEve->Redraw3D();
      }
   }

   return q;
}

TEveQuadSet* quad_test_hexid(Float_t x=0, Float_t y=0, Float_t z=0,
                                  Int_t num=100, Bool_t register=kTRUE)
{
   TEveManager::Create();

   TRandom r(0);
   gStyle->SetPalette(1, 0);

   {
      TEveQuadSet* q = new TEveQuadSet("HexagonXY");
      q->SetOwnIds(kTRUE);
      q->Reset(TEveQuadSet::kQT_HexagonXY, kFALSE, 32);
      for (Int_t i=0; i<num; ++i) {
         q->AddHexagon(r.Uniform(-10, 10), r.Uniform(-10, 10), r.Uniform(-10, 10),
                       r.Uniform(0.2, 1));
         q->QuadValue(r.Uniform(0, 120));
         q->QuadId(new TNamed(Form("Quad with idx=%d", i), "This title is not confusing."));
      }
      q->RefitPlex();

      TEveTrans& t = q->RefMainTrans();
      t.SetPos(x, y, z);

      if (register)
      {
         gEve->AddElement(q);
         gEve->Redraw3D();
      }
   }

   return q;
}

void quad_test_hierarchy(Int_t n=4)
{
   TEveManager::Create();

   gStyle->SetPalette(1, 0);

   TEveRGBAPalette* pal = new TEveRGBAPalette(20, 100);
   pal->SetLimits(0, 120);

   TEveFrameBox* box = new TEveFrameBox();
   box->SetAABox(-10, -10, -10, 20, 20, 20);
   box->SetFrameColor((Color_t) 33);

   TEveElementList* l = new TEveElementList("Parent/Dir");
   l->SetTitle("Tooltip");
   //  l->SetMainColor((Color_t)3);
   gEve->AddElement(l);

   for (Int_t i=0; i<n; ++i)
   {
      TEveQuadSet* qs = quad_test_hexid(0, 0, 50*i, 50, kFALSE);
      qs->SetPalette(pal);
      qs->SetFrame(box);
      gEve->AddElement(qs, l);
   }

   gEve->Redraw3D();
}
