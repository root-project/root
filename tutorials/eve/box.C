// @(#)root/eve:$Id$
// Author: Matevz Tadel

// Demonstrates usage of TEveBox class.


TEveBox* box()
{
   TEveManager::Create();

   TRandom r(0);

   TEveBox* b = new TEveBox;
   b->SetMainColor(kCyan);
   b->SetMainTransparency(0);

   Float_t a = 10, d = 5;
   b->SetVertex(0, -a + r.Uniform(-d, d), -a + r.Uniform(-d, d), -a + r.Uniform(-d, d));
   b->SetVertex(1, -a + r.Uniform(-d, d),  a + r.Uniform(-d, d), -a + r.Uniform(-d, d));
   b->SetVertex(2,  a + r.Uniform(-d, d),  a + r.Uniform(-d, d), -a + r.Uniform(-d, d));
   b->SetVertex(3,  a + r.Uniform(-d, d), -a + r.Uniform(-d, d), -a + r.Uniform(-d, d));
   b->SetVertex(4, -a + r.Uniform(-d, d), -a + r.Uniform(-d, d),  a + r.Uniform(-d, d));
   b->SetVertex(5, -a + r.Uniform(-d, d),  a + r.Uniform(-d, d),  a + r.Uniform(-d, d));
   b->SetVertex(6,  a + r.Uniform(-d, d),  a + r.Uniform(-d, d),  a + r.Uniform(-d, d));
   b->SetVertex(7,  a + r.Uniform(-d, d), -a + r.Uniform(-d, d),  a + r.Uniform(-d, d));

   gEve->AddElement(b);
   gEve->Redraw3D(kTRUE);
}
