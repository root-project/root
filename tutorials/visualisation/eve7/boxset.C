/// \file
/// \ingroup tutorial_eve_7
/// Demonstrates usage of REveBoxSet class.
///
/// \image html eve_boxset.png
/// \macro_code
///
/// \author Matevz Tadel

#include "TRandom.h"
#include <ROOT/REveElement.hxx>
#include <ROOT/REveScene.hxx>
#include <ROOT/REveManager.hxx>
#include <ROOT/REveBoxSet.hxx>

using namespace ROOT::Experimental;

std::string customTooltip(const ROOT::Experimental::REveDigitSet *digitSet, int n)
{
   auto d = digitSet->GetDigit(n);
   return TString::Format("Custom tooltip:\n value %d idx %d\n", d->fValue, n).Data();
}

REveBoxSet *boxset_free(Int_t num = 100)
{
   TRandom r(0);

   auto pal = new REveRGBAPalette(0, 130);
   pal->SetMin(80);

   auto q = new REveBoxSet("BoxSet");
   q->SetPalette(pal);
   q->Reset(REveBoxSet::kBT_FreeBox, kFALSE, 64);

#define RND_BOX(x) (Float_t) r.Uniform(-(x), (x))

   const float R = 500;
   const float A = 40;
   const float D = 1;
   for (int i = 0; i < num; ++i) {
      float x = RND_BOX(R);
      float y = RND_BOX(R);
      float z = RND_BOX(R);
      float a = r.Uniform(0.2 * A, A);
      float d = D;
      float verts[24] = {x - a + RND_BOX(d), y - a + RND_BOX(d), z - a + RND_BOX(d), x - a + RND_BOX(d),
                         y + a + RND_BOX(d), z - a + RND_BOX(d), x + a + RND_BOX(d), y + a + RND_BOX(d),
                         z - a + RND_BOX(d), x + a + RND_BOX(d), y - a + RND_BOX(d), z - a + RND_BOX(d),
                         x - a + RND_BOX(d), y - a + RND_BOX(d), z + a + RND_BOX(d), x - a + RND_BOX(d),
                         y + a + RND_BOX(d), z + a + RND_BOX(d), x + a + RND_BOX(d), y + a + RND_BOX(d),
                         z + a + RND_BOX(d), x + a + RND_BOX(d), y - a + RND_BOX(d), z + a + RND_BOX(d)};
      q->AddFreeBox(verts);
      q->DigitValue(r.Uniform(0, 130));
   }
   q->RefitPlex();

#undef RND_BOX

   // Uncomment these two lines to get internal highlight / selection.
   q->SetPickable(1);
   q->SetAlwaysSecSelect(1);
   q->SetTooltipCBFoo(customTooltip);

   return q;
}

REveBoxSet *boxset_axisaligned(Float_t x = 0, Float_t y = 0, Float_t z = 0, Int_t num = 100)
{
   TRandom r(0);

   auto pal = new REveRGBAPalette(0, 130);

   auto frm = new REveFrameBox();
   frm->SetAABoxCenterHalfSize(0, 0, 0, 12, 12, 12);
   frm->SetFrameColor(kCyan);
   frm->SetBackColorRGBA(120, 120, 120, 20);
   frm->SetDrawBack(kTRUE);

   auto q = new REveBoxSet("BoxSet");
   q->SetPalette(pal);
   q->SetFrame(frm);
   q->Reset(REveBoxSet::kBT_InstancedScaled, kFALSE, 64);

   for (Int_t i = 0; i < num; ++i) {
      q->AddInstanceScaled(r.Uniform(-10, 10), r.Uniform(-10, 10), r.Uniform(-10, 10), r.Uniform(0.2, 1),
                           r.Uniform(0.2, 1), r.Uniform(0.2, 1));
      q->DigitValue(r.Uniform(0, 130));
   }
   q->RefitPlex();

   REveTrans &t = q->RefMainTrans();
   t.SetPos(x, y, z);

   // Uncomment these two lines to get internal highlight / selection.
   q->SetPickable(1);
   q->SetAlwaysSecSelect(1);
   q->SetTooltipCBFoo(customTooltip);

   return q;
}

REveBoxSet *boxset_colisval(Float_t x = 0, Float_t y = 0, Float_t z = 0, Int_t num = 100)
{
   TRandom r(0);

   auto q = new REveBoxSet("BoxSet");
   q->Reset(REveBoxSet::kBT_InstancedScaled, kTRUE, 64);
   for (Int_t i = 0; i < num; ++i) {
      q->AddInstanceScaled(r.Uniform(-10, 10), r.Uniform(-10, 10), r.Uniform(-10, 10), r.Uniform(0.2, 1),
                           r.Uniform(0.2, 1), r.Uniform(0.2, 1));
      q->DigitColor(r.Uniform(20, 255), r.Uniform(20, 255), r.Uniform(20, 255), r.Uniform(20, 255));
   }
   q->RefitPlex();

   REveTrans &t = q->RefMainTrans();
   t.SetPos(x, y, z);

   return q;
}

REveBoxSet *boxset_single_color(Float_t x = 0, Float_t y = 0, Float_t z = 0, Int_t num = 100)
{
   TRandom r(0);

   auto q = new REveBoxSet("BoxSet");
   q->UseSingleColor();
   q->SetMainColorPtr(new Color_t);
   q->SetMainColor(kRed);
   q->SetMainTransparency(50);
   q->Reset(REveBoxSet::kBT_InstancedScaled, kFALSE, 64);

   for (Int_t i = 0; i < num; ++i) {
      q->AddInstanceScaled(r.Uniform(-10, 10), r.Uniform(-10, 10), r.Uniform(-10, 10), r.Uniform(0.2, 1),
                           r.Uniform(0.2, 1), r.Uniform(0.2, 1));
   }
   q->RefitPlex();

   REveTrans &t = q->RefMainTrans();
   t.SetPos(x, y, z);

   return q;
}

REveBoxSet *boxset_fixed_dim(Float_t x = 0, Float_t y = 0, Float_t z = 0, Int_t num = 100)
{
   TRandom r(0);

   auto q = new REveBoxSet("BoxSet");
   q->UseSingleColor();
   q->SetMainColorPtr(new Color_t);
   q->SetMainColor(kRed);
   q->SetMainTransparency(50);
   q->Reset(REveBoxSet::kBT_Instanced, kFALSE, 64);

   for (Int_t i = 0; i < num; ++i) {
      q->AddInstance(r.Uniform(-10, 10), r.Uniform(-10, 10), r.Uniform(-10, 10));
   }
   q->RefitPlex();

   q->SetPickable(1);
   q->SetAlwaysSecSelect(1);
   REveTrans &t = q->RefMainTrans();
   t.SetPos(x, y, z);
   return q;
}

REveBoxSet *boxset_hex(Float_t x = 0, Float_t y = 0, Float_t z = 0, Int_t num = 100)
{
   TRandom r(0);

   auto q = new REveBoxSet("BoxSet");
   q->Reset(REveBoxSet::kBT_InstancedScaledRotated, kTRUE, 64);

   for (Int_t i = 0; i < num; ++i) {
      q->AddHex(REveVector(r.Uniform(-10, 10), r.Uniform(-10, 10), r.Uniform(-10, 10)), r.Uniform(0.2, 1),
                r.Uniform(0, 60), r.Uniform(0.2, 5));
      q->DigitColor(r.Uniform(20, 255), r.Uniform(20, 255), r.Uniform(20, 255), r.Uniform(20, 255));
   }
   q->RefitPlex();

   q->SetPickable(true);
   q->SetAlwaysSecSelect(true);

   REveTrans &t = q->RefMainTrans();
   t.SetPos(x, y, z);
   return q;
}

REveBoxSet *boxset_gentrans(Float_t x = 0, Float_t y = 0, Float_t z = 0, int num = 10)
{
   auto q = new REveBoxSet("BoxSet-GenTrans");
   q->Reset(REveBoxSet::kBT_InstancedScaledRotated, kTRUE, 64);

   TRandom r(0);
   for (Int_t i = 0; i < num; ++i) {
      // Create per digit transformation
      REveTrans t;
      float x = 50 - i * 10;
      t.Move3LF(x, 0, 0);
      t.Scale(1, 1, 10);
      t.RotateLF(1, 2, r.Uniform(3.14));
      // t.Print();
      float farr[16];
      for (int m = 0; m < 16; m++)
         farr[m] = t.Array()[m];

      q->AddInstanceMat4(farr);
      q->DigitColor(255, 0, 0, 100); // AMT how the treansparency handled, last ergument alpha
   }
   q->RefitPlex();
   q->SetPickable(true);
   q->SetAlwaysSecSelect(true);

   REveTrans &t = q->RefMainTrans();
   t.SetPos(x, y, z);

   return q;
}

void boxset()
{
   enum EBoxDemo_t {
      ScaledRotated,
      Free,
      AxisAligned,
      Hexagon,
      FixedDimension,
      SingleColor
   };

   // EBoxDemo_t demo = ScaledRotated;
   EBoxDemo_t demo = AxisAligned;

   auto eveMng = REveManager::Create();
   REveBoxSet *b = nullptr;
   switch (demo) {
   case ScaledRotated: b = boxset_gentrans(); break;
   case Free: b = boxset_free(); break;
   case AxisAligned: b = boxset_axisaligned(); break;
   case FixedDimension: b = boxset_fixed_dim(); break;
   case Hexagon: b = boxset_hex(); break;
   case SingleColor: b = boxset_single_color(); break;
   default: printf("Unsupported demo type. \n"); return;
   }

   eveMng->GetEventScene()->AddElement(b);

   // Add palette to scene to be streamed and edited in controller
   if (b->GetPalette())
      eveMng->GetEventScene()->AddElement(b->GetPalette());

   eveMng->Show();

   REveViewer *v = ROOT::Experimental::gEve->GetDefaultViewer();
   v->SetAxesType(REveViewer::kAxesOrigin);

   // AMT temporary solution: add geo-shape to set scene bounding box
   // digits at the moment have no bounding box calculation and lights are consequently
   // in inital empty position at extend size of 1
   auto b1 = new REveGeoShape("Bounding Box Barrel");
   b1->SetShape(new TGeoTube(30, 32, 10));
   b1->SetMainColor(kCyan);
   b1->SetNSegments(80);
   b1->SetMainTransparency(95);
   ROOT::Experimental::gEve->GetGlobalScene()->AddElement(b1);
}
