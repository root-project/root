/// \file
/// \ingroup tutorial_eve_7
/// Move an element from the server by changing only its transformation.
///
/// One box stands still in the centre. Four boxes go round a square about it,
/// a step every 200 ms, each moved by SetTransMatrix(), which streams only the
/// new matrix. The client moves the objects it already has and rebuilds no
/// geometry.
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

#include <vector>

using namespace ROOT::Experimental;

/// Position (`x`, `y`) at `s` along the perimeter of a square of half-side
/// `a`, s in [0, 4) going counter-clockwise from the corner (a, -a).
static void square_pos(Double_t s, Double_t a, Double_t &x, Double_t &y)
{
   const Double_t f = 2 * a * (s - TMath::Floor(s)) - a; // -a .. a along the side
   switch (Int_t(TMath::Floor(s)) % 4) {
   case 0:
      x = a;
      y = f;
      break;
   case 1:
      x = -f;
      y = a;
      break;
   case 2:
      x = -a;
      y = -f;
      break;
   default:
      x = f;
      y = -a;
      break;
   }
}

class REveTransMover : public TTimer {
   // clang-format off
   std::vector<REveElement*> fEls;
   int                       fStep{0};
   // clang-format on

public:
   REveTransMover(std::vector<REveElement *> els, Long_t ms) : TTimer(ms, kTRUE), fEls(std::move(els)) {}

   Bool_t Notify() override
   {
      REveManager::ChangeGuard ch;

      for (size_t i = 0; i < fEls.size(); ++i) {
         Double_t x, y;
         square_pos(0.02 * fStep + i, 40, x, y);
         REveTrans t;
         t.SetPos(0.0, y, x);                // in the yz plane, which faces the default camera
         fEls[i]->SetTransMatrix(t.Array()); // a transformation-only change
      }

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

   // clang-format off
   b->SetVertex(0, -a, -a, z - a);   b->SetVertex(1, -a,  a, z - a);
   b->SetVertex(2,  a,  a, z - a);   b->SetVertex(3,  a, -a, z - a);
   b->SetVertex(4, -a, -a, z + a);   b->SetVertex(5, -a,  a, z + a);
   b->SetVertex(6,  a,  a, z + a);   b->SetVertex(7,  a, -a, z + a);
   // clang-format on

   return b;
}

void trans_update()
{
   auto eveMng = REveManager::Create();
   eveMng->AllowMultipleRemoteConnections(false, false);

   auto still = make_box("Still", kCyan, 10, 0);
   eveMng->GetEventScene()->AddElement(still);

   const Color_t colors[] = {kMagenta, kOrange, kGreen + 1, kBlue};
   std::vector<REveElement *> movers;
   for (int i = 0; i < 4; ++i) {
      auto mover = make_box(Form("Mover %d", i), colors[i], 6, 0);
      // Give the box a transformation from the start, so the first stream
      // already carries its matrix.
      mover->InitMainTrans();
      eveMng->GetEventScene()->AddElement(mover);
      movers.push_back(mover);
   }

   eveMng->Show();

   auto timer = new REveTransMover(movers, 200);
   timer->Notify();
   timer->TurnOn();
}
