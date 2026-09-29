/// \file
/// \ingroup tutorial_eve_7
/// Move an element from the server by changing only its transformation.
///
/// One box stands still. Every 1.5 s the other is moved by SetTransMatrix(),
/// which streams only the new matrix. The client moves the object it already
/// has and rebuilds no geometry.
///
/// \macro_code
///
/// \author Matevz Tadel

#include <ROOT/REveManager.hxx>
#include <ROOT/REveScene.hxx>
#include <ROOT/REveBox.hxx>
#include <ROOT/REveTrans.hxx>

#include <TTimer.h>
#include <TMath.h>

using namespace ROOT::Experimental;

class REveTransMover : public TTimer {
   REveElement *fEl{nullptr};
   int          fStep{0};

public:
   REveTransMover(REveElement *el, Long_t ms) : TTimer(ms, kTRUE), fEl(el) {}

   Bool_t Notify() override
   {
      REveManager::ChangeGuard ch;

      REveTrans t;
      t.SetPos(30.0 * TMath::Sin(0.4 * fStep), 0.0, 0.0);
      fEl->SetTransMatrix(t.Array());   // a transformation-only change

      ++fStep;
      Reset();
      return kTRUE;
   }
};

static REveBox *make_box(const char *name, Color_t col, Float_t a, Float_t z)
{
   auto b = new REveBox(name);
   b->SetMainColor(col);
   b->SetMainTransparency(0);

   b->SetVertex(0, -a, -a, z - a);   b->SetVertex(1, -a,  a, z - a);
   b->SetVertex(2,  a,  a, z - a);   b->SetVertex(3,  a, -a, z - a);
   b->SetVertex(4, -a, -a, z + a);   b->SetVertex(5, -a,  a, z + a);
   b->SetVertex(6,  a,  a, z + a);   b->SetVertex(7,  a, -a, z + a);

   return b;
}

void trans_update()
{
   auto eveMng = REveManager::Create();
   eveMng->AllowMultipleRemoteConnections(false, false);

   auto still = make_box("Still", kCyan, 10, -20);
   eveMng->GetEventScene()->AddElement(still);

   auto mover = make_box("Mover", kMagenta, 10, 20);
   // Give the box a transformation from the start, so the first stream already
   // carries its matrix.
   mover->InitMainTrans();
   eveMng->GetEventScene()->AddElement(mover);

   eveMng->Show();

   (new REveTransMover(mover, 1500))->TurnOn();
}
