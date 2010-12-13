// @(#)root/eve:$Id$
// Author: Matevz Tadel

// Demonstrates usage of TEveBoxSet class.

TEveBoxSet* boxset(Float_t x=0, Float_t y=0, Float_t z=0,
		   Int_t num=100, Bool_t register=kTRUE)
{
   TEveManager::Create();

   TRandom r(0);
   gStyle->SetPalette(1, 0);

   TEveRGBAPalette* pal = new TEveRGBAPalette(0, 130);

   TEveFrameBox* frm = new TEveFrameBox();
   frm->SetAABoxCenterHalfSize(0, 0, 0, 12, 12, 12);
   frm->SetFrameColor(kCyan);
   frm->SetBackColorRGBA(120,120,120,20);
   frm->SetDrawBack(kTRUE);

   TEveBoxSet* q = new TEveBoxSet("BoxSet");
   q->SetPalette(pal);
   q->SetFrame(frm);
   q->Reset(TEveBoxSet::kBT_AABox, kFALSE, 64);
   for (Int_t i=0; i<num; ++i) {
      q->AddBox(r.Uniform(-10, 10), r.Uniform(-10, 10), r.Uniform(-10, 10),
                r.Uniform(0.2, 1),  r.Uniform(0.2, 1),  r.Uniform(0.2, 1));
      q->DigitValue(r.Uniform(0, 130));
   }
   q->RefitPlex();

   TEveTrans& t = q->RefMainTrans();
   t.SetPos(x, y, z);

   // Uncomment these two lines to get internal highlight / selection.
   // q->SetPickable(1);
   // q->SetAlwaysSecSelect(1);

   if (register)
   {
      gEve->AddElement(q);
      gEve->Redraw3D(kTRUE);
   }

   return q;
}

TEveBoxSet* boxset_colisval(Float_t x=0, Float_t y=0, Float_t z=0,
			    Int_t num=100, Bool_t register=kTRUE)
{
   TEveManager::Create();

   TRandom r(0);

   TEveBoxSet* q = new TEveBoxSet("BoxSet");
   q->Reset(TEveBoxSet::kBT_AABox, kTRUE, 64);
   for (Int_t i=0; i<num; ++i) {
      q->AddBox(r.Uniform(-10, 10), r.Uniform(-10, 10), r.Uniform(-10, 10),
                r.Uniform(0.2, 1),  r.Uniform(0.2, 1),  r.Uniform(0.2, 1));
      q->DigitColor(r.Uniform(20, 255), r.Uniform(20, 255),
                    r.Uniform(20, 255), r.Uniform(20, 255));
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

TEveBoxSet* boxset_single_color(Float_t x=0, Float_t y=0, Float_t z=0,
                                Int_t num=100, Bool_t register=kTRUE)
{
   TEveManager::Create();

   TRandom r(0);

   TEveBoxSet* q = new TEveBoxSet("BoxSet");
   q->UseSingleColor();
   q->SetMainColor(kCyan-2);
   q->SetMainTransparency(50);
   q->Reset(TEveBoxSet::kBT_AABox, kFALSE, 64);
   for (Int_t i=0; i<num; ++i) {
      q->AddBox(r.Uniform(-10, 10), r.Uniform(-10, 10), r.Uniform(-10, 10),
                r.Uniform(0.2, 1),  r.Uniform(0.2, 1),  r.Uniform(0.2, 1));
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

TEveBoxSet* boxset_freebox(Int_t num=100, Bool_t register=kTRUE)
{
   TEveManager::Create();

   TRandom r(0);
   gStyle->SetPalette(1, 0);

   TEveRGBAPalette* pal = new TEveRGBAPalette(0, 130);

   TEveBoxSet* q = new TEveBoxSet("BoxSet");
   q->SetPalette(pal);
   q->Reset(TEveBoxSet::kBT_FreeBox, kFALSE, 64);

#define RND_BOX(x) r.Uniform(-(x), (x))

   Float_t verts[24];
   for (Int_t i=0; i<num; ++i) {
      Float_t x = RND_BOX(10);
      Float_t y = RND_BOX(10);
      Float_t z = RND_BOX(10);
      Float_t a = r.Uniform(0.2, 0.5);
      Float_t d = 0.05;
      Float_t verts[24] = {
         x - a + RND_BOX(d), y - a + RND_BOX(d), z - a + RND_BOX(d),
         x - a + RND_BOX(d), y + a + RND_BOX(d), z - a + RND_BOX(d),
         x + a + RND_BOX(d), y + a + RND_BOX(d), z - a + RND_BOX(d),
         x + a + RND_BOX(d), y - a + RND_BOX(d), z - a + RND_BOX(d),
         x - a + RND_BOX(d), y - a + RND_BOX(d), z + a + RND_BOX(d),
         x - a + RND_BOX(d), y + a + RND_BOX(d), z + a + RND_BOX(d),
         x + a + RND_BOX(d), y + a + RND_BOX(d), z + a + RND_BOX(d),
         x + a + RND_BOX(d), y - a + RND_BOX(d), z + a + RND_BOX(d) };
      q->AddBox(verts);
      q->DigitValue(r.Uniform(0, 130));
   }
   q->RefitPlex();

#undef RND_BOX

   // Uncomment these two lines to get internal highlight / selection.
   // q->SetPickable(1);
   // q->SetAlwaysSecSelect(1);

   if (register)
   {
      gEve->AddElement(q);
      gEve->Redraw3D(kTRUE);
   }

   return q;
}
