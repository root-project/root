// @(#)root/eve:$Id$
// Author: Matevz Tadel

// Demonstrates usage of TEveBox class.


TEveBox* box(Float_t a=10, Float_t d=5, Float_t x=0, Float_t y=0, Float_t z=0)
{
   TEveManager::Create();

   TRandom& r = * gRandom;

   TEveBox* b = new TEveBox;
   b->SetMainColor(kCyan);
   b->SetMainTransparency(0);

   b->SetVertex(0, x - a + r.Uniform(-d, d), y - a + r.Uniform(-d, d), z - a + r.Uniform(-d, d));
   b->SetVertex(1, x - a + r.Uniform(-d, d), y + a + r.Uniform(-d, d), z - a + r.Uniform(-d, d));
   b->SetVertex(2, x + a + r.Uniform(-d, d), y + a + r.Uniform(-d, d), z - a + r.Uniform(-d, d));
   b->SetVertex(3, x + a + r.Uniform(-d, d), y - a + r.Uniform(-d, d), z - a + r.Uniform(-d, d));
   b->SetVertex(4, x - a + r.Uniform(-d, d), y - a + r.Uniform(-d, d), z + a + r.Uniform(-d, d));
   b->SetVertex(5, x - a + r.Uniform(-d, d), y + a + r.Uniform(-d, d), z + a + r.Uniform(-d, d));
   b->SetVertex(6, x + a + r.Uniform(-d, d), y + a + r.Uniform(-d, d), z + a + r.Uniform(-d, d));
   b->SetVertex(7, x + a + r.Uniform(-d, d), y - a + r.Uniform(-d, d), z + a + r.Uniform(-d, d));

   gEve->AddElement(b);
   gEve->Redraw3D(kTRUE);
}
