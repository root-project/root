/// \file
/// \ingroup tutorial_eve
/// Demonstates usage of 2D digit class TEveQuadSet.
///
/// \image html eve_quadset.png
/// \macro_code
///
/// \author Matevz Tadel

void    quadset_callback(TEveDigitSet* ds, Int_t idx, TObject* obj);
TString quadset_tooltip_callback(TEveDigitSet* ds, Int_t idx);
void    quadset_set_callback(TEveDigitSet* ds);

//------------------------------------------------------------------------------

TEveQuadSet* quadset(Float_t x=0, Float_t y=0, Float_t z=0,
                       Int_t num=100, Bool_t registerSet=kTRUE)
{
   TEveManager::Create();

   TRandom r(0);

   TEveRGBAPalette *pal = new TEveRGBAPalette(0, 130);
   TEveFrameBox    *box = new TEveFrameBox();
   box->SetAAQuadXY(-10, -10, 0, 20, 20);
   box->SetFrameColor(kGray);

   TEveQuadSet* q = new TEveQuadSet("RectangleXY");
   q->SetOwnIds(kTRUE);
   q->SetPalette(pal);
   q->SetFrame(box);
   q->Reset(TEveQuadSet::kQT_RectangleXY, kFALSE, 32);
   for (Int_t i=0; i<num; ++i)
   {
      q->AddQuad(r.Uniform(-10, 9), r.Uniform(-10, 9), 0,
                 r.Uniform(0.2, 1), r.Uniform(0.2, 1));
      q->QuadValue(r.Uniform(0, 130));
      q->QuadId(new TNamed(Form("QuadIdx %d", i),
                           "TNamed assigned to a quad as an indentifier."));
   }
   q->RefitPlex();

   TEveTrans& t = q->RefMainTrans();
   t.RotateLF(1, 3, 0.5*TMath::Pi());
   t.SetPos(x, y, z);

   TGLViewer* v = gEve->GetDefaultGLViewer();
   v->SetCurrentCamera(TGLViewer::kCameraOrthoZOY);
   TGLCameraOverlay* co = v->GetCameraOverlay();
   co->SetShowOrthographic(kTRUE);
   co->SetOrthographicMode(TGLCameraOverlay::kGridFront);

   // Uncomment these two lines to get internal highlight / selection.
   // q->SetPickable(1);
   // q->SetAlwaysSecSelect(1);

   TEveRGBAPaletteOverlay *po = new TEveRGBAPaletteOverlay(pal, 0.55, 0.1, 0.4, 0.05);
   v = gEve->GetDefaultGLViewer();
   v->AddOverlayElement(po);

   // To set user-interface (GUI + overlay) to display real values
   // mapped with a linear function: r = 0.1 * i + 0;
   // pal->SetUIDoubleRep(kTRUE, 0.1, 0);

   if (registerSet)
   {
      gEve->AddElement(q);
      gEve->Redraw3D(kTRUE);
   }

   Info("quadset", "use alt-left-mouse to select individual digits.");

   return q;
}

TEveQuadSet* quadset_emc(Float_t x=0, Float_t y=0, Float_t z=0, Int_t num=100)
{
   TEveManager::Create();

   TRandom r(0);

   TEveQuadSet* q = new TEveQuadSet("EMC Supermodule");
   q->SetOwnIds(kTRUE);
   q->Reset(TEveQuadSet::kQT_RectangleXZFixedDimY, kFALSE, 32);
   q->SetDefWidth(8);
   q->SetDefHeight(8);

   for (Int_t i=0; i<num; ++i)
   {
      q->AddQuad(r.Uniform(-100, 100), r.Uniform(-100, 100));
      q->QuadValue(r.Uniform(0, 130));
      q->QuadId(new TNamed(Form("Cell %d", i), "Dong!"));
   }
   q->RefitPlex();

   TEveTrans& t = q->RefMainTrans();
   t.SetPos(x, y, z);

   gEve->AddElement(q);
   gEve->Redraw3D();

   return q;
}

TEveQuadSet* quadset_circ()
{
   TEveManager::Create();

   TRandom rnd(0);

   Float_t R = 10, dW = 1, dH = .5;

   TEveFrameBox *box = new TEveFrameBox();
   {
      Float_t  frame[3*36];
      Float_t *p = frame;
      for (Int_t i = 0; i < 36; ++i, p += 3) {
         p[0] = 11 * TMath::Cos(TMath::TwoPi()*i/36);
         p[1] = 11 * TMath::Sin(TMath::TwoPi()*i/36);
         p[2] = 0;
      }
      box->SetQuadByPoints(frame, 36);
   }
   box->SetFrameColor(kGray);

   TEveQuadSet* q = new TEveQuadSet("Pepe");
   q->SetFrame(box);
   q->Reset(TEveQuadSet::kQT_HexagonXY, kFALSE, 32);

   for (Float_t r = R; r > 2; r *= 0.8)
   {
      Int_t maxI = 2.0*TMath::Pi()*r / 2;
      for (Int_t i = 0; i < maxI; ++i)
      {
         Float_t x = r * TMath::Cos(TMath::TwoPi()*i/maxI);
         Float_t y = r * TMath::Sin(TMath::TwoPi()*i/maxI);
         q->AddHexagon(x, y, rnd.Uniform(-1, 1), rnd.Uniform(0.2, 1));
         q->QuadValue(rnd.Uniform(0, 130));
      }
   }
   q->RefitPlex();

   TEveTrans& t = q->RefMainTrans();
   t.RotateLF(1, 3, 0.5*TMath::Pi());

   gEve->AddElement(q);
   gEve->Redraw3D();

   return q;
}

TEveQuadSet* quadset_hex(Float_t x=0, Float_t y=0, Float_t z=0,
                           Int_t num=100, Bool_t registerSet=kTRUE)
{
   TEveManager::Create();

   TRandom r(0);

   {
      TEveQuadSet* q = new TEveQuadSet("HexagonXY");
      q->Reset(TEveQuadSet::kQT_HexagonXY, kFALSE, 32);
      for (Int_t i=0; i<num; ++i)
      {
         q->AddHexagon(r.Uniform(-10, 10),
                       r.Uniform(-10, 10),
                       r.Uniform(-10, 10),
                       r.Uniform(0.2, 1));
         q->QuadValue(r.Uniform(0, 120));
      }
      q->RefitPlex();

      TEveTrans& t = q->RefMainTrans();
      t.SetPos(x, y, z);

      if (registerSet)
      {
         gEve->AddElement(q);
         gEve->Redraw3D();
      }
   }

   {
      TEveQuadSet* q = new TEveQuadSet("HexagonYX");
      q->Reset(TEveQuadSet::kQT_HexagonYX, kFALSE, 32);
      for (Int_t i=0; i<num; ++i)
      {
         q->AddHexagon(r.Uniform(-10, 10),
                       r.Uniform(-10, 10),
                       r.Uniform(-10, 10),
                       r.Uniform(0.2, 1));
         q->QuadValue(r.Uniform(0, 120));
      }
      q->RefitPlex();

      TEveTrans& t = q->RefMainTrans();
      t.SetPos(x, y, z);

      if (registerSet)
      {
         gEve->AddElement(q);
         gEve->Redraw3D();
      }

      return q;
   }
}

TEveQuadSet* quadset_hexid(Float_t x=0, Float_t y=0, Float_t z=0,
                             Int_t num=100, Bool_t registerSet=kTRUE)
{
   TEveManager::Create();

   TRandom r(0);

   TEveQuadSet* q = new TEveQuadSet("HexagonXY");

   {

      q->SetOwnIds(kTRUE);
      q->Reset(TEveQuadSet::kQT_HexagonXY, kFALSE, 32);
      for (Int_t i=0; i<num; ++i)
      {
         q->AddHexagon(r.Uniform(-10, 10),
                       r.Uniform(-10, 10),
                       r.Uniform(-10, 10),
                       r.Uniform(0.2, 1));
         q->QuadValue(r.Uniform(0, 120));
         q->QuadId(new TNamed(Form("Quad with idx=%d", i),
                              "This title is not confusing."));
      }
      q->RefitPlex();

      TEveTrans& t = q->RefMainTrans();
      t.SetPos(x, y, z);

      if (registerSet)
      {
         gEve->AddElement(q);
         gEve->Redraw3D();
      }
   }

   quadset_set_callback(q);

   // With the following you get per digit highlight with tooltip.
   //q->SetPickable(1);
   //q->SetAlwaysSecSelect(1);
   // Otherwise you need to Alt - left click to get info printout.

   return q;
}

void quadset_hierarchy(Int_t n=4)
{
   TEveManager::Create();


   TEveRGBAPalette* pal = new TEveRGBAPalette(20, 100);
   pal->SetLimits(0, 120);

   TEveFrameBox* box = new TEveFrameBox();
   box->SetAABox(-10, -10, -10, 20, 20, 20);
   box->SetFrameColor(33);

   TEveElementList* l = new TEveElementList("Parent/Dir");
   l->SetTitle("Tooltip");
   //  l->SetMainColor(3);
   gEve->AddElement(l);

   for (Int_t i=0; i<n; ++i)
   {
      TEveQuadSet* qs = quadset_hexid(0, 0, 50*i, 50, kFALSE);
      qs->SetPalette(pal);
      qs->SetFrame(box);
      l->AddElement(qs);
   }

   gEve->Redraw3D();
}

//------------------------------------------------------------------------------

void quadset_callback(TEveDigitSet* ds, Int_t idx, TObject* obj)
{
   printf("dump_digit_set_hit - 0x%lx, id=%d, obj=0x%lx\n",
          (ULong_t) ds, idx, (ULong_t) obj);
}

TString quadset_tooltip_callback(TEveDigitSet* ds, Int_t idx)
{
   // This gets called for tooltip if the following is set:
   //   q->SetPickable(1);
   //   q->SetAlwaysSecSelect(1);

   return TString::Format("callback tooltip for '%s' - 0x%lx, id=%d\n",
                          ds->GetElementName(), (ULong_t) ds, idx);
}

void quadset_set_callback(TEveDigitSet* ds)
{
   ds->SetCallbackFoo(quadset_callback);
   ds->SetTooltipCBFoo(quadset_tooltip_callback);
}
