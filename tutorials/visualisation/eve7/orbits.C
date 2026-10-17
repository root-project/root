/// \file
/// \ingroup tutorial_eve_7
/// Bodies orbiting bodies orbiting bodies, built as an element hierarchy.
///
/// The shapes are the random spheres, tubes, tube segments and cut tubes of
/// gl/glViewerLOD.C. Each body sits at a distance from the origin of a pivot,
/// a plain REveElement that spins about its local z axis, so the body moves on
/// a circle in the pivot's xy plane. Each body spins about its own z axis as
/// well. The pivots of a body's children are the body's children, so the
/// children orbit in the body's frame and follow both its orbit and its spin.
///
/// The spins are set with REveTrans::SetMotion() and extrapolated on the
/// client. A timer advances each matrix to the current angle and sends the
/// motion again every kRefreshSec, which also brings the motion to a client
/// that connects later.
///
/// \macro_code
///
/// \author Matevz Tadel

#include <ROOT/REveElement.hxx>
#include <ROOT/REveGeoShape.hxx>
#include <ROOT/REveManager.hxx>
#include <ROOT/REveScene.hxx>
#include <ROOT/REveTrans.hxx>
#include <ROOT/REveUtil.hxx>
#include <ROOT/REveVector.hxx>

#include "TGeoSphere.h"
#include "TGeoTube.h"
#include "TMath.h"
#include "TRandom.h"
#include "TTimer.h"

#include <iterator>
#include <vector>

using namespace ROOT::Experimental;

namespace {

const Double_t kRefreshSec = 2.;

const Color_t kColors[] = {kRed,        kGreen + 1, kBlue,   kOrange, kMagenta,  kCyan + 1,
                           kYellow + 1, kAzure + 1, kViolet, kTeal,   kPink + 1, kSpring};

/// An element that spins about its local z axis at fRate rad/s, starting from
/// fBase at server time fT0Ms.
struct Spinner {
   REveElement *fEl;
   REveTrans fBase;
   Double_t fRate;
   Double_t fT0Ms;
};

std::vector<Spinner> gSpinners;

/// A random shape of the kinds glViewerLOD.C uses, about `size` across.
TGeoShape *RandomShape(Double_t size)
{
   TRandom &r = *gRandom;
   const Double_t rMin = r.Rndm() * 0.5 * size;
   const Double_t rMax = rMin + (0.2 + r.Rndm() * 0.5) * size;
   const Double_t dz = (0.2 + r.Rndm()) * size;
   const Double_t phi1 = r.Rndm() * 90.;
   const Double_t phi2 = phi1 + 90. + r.Rndm() * 180.;

   switch (r.Integer(4)) {
   case 0: return new TGeoSphere(0., rMax);
   case 1: return new TGeoTube(rMin, rMax, dz);
   case 2: return new TGeoTubeSeg(rMin, rMax, dz, phi1, phi2);
   default:
      return new TGeoCtub(rMin, rMax, dz, phi1, phi2, r.Rndm() * .5, r.Rndm() * .5, -1. + r.Rndm() * .5, r.Rndm() * .5,
                          r.Rndm() * .5, 1. - r.Rndm() * .5);
   }
}

/// Give `t` a random orientation, keeping its position.
void RandomOrientation(REveTrans &t)
{
   TRandom &r = *gRandom;
   t.RotateLF(1, 2, r.Uniform(0, TMath::TwoPi()));
   t.RotateLF(2, 3, r.Uniform(0, TMath::Pi()));
   t.RotateLF(1, 2, r.Uniform(0, TMath::TwoPi()));
}

void AddSpinner(REveElement *el, Double_t rate)
{
   gSpinners.push_back({el, el->RefMainTrans(), rate, REveUtil::ServerTimeMs()});
}

/// Under `parent`, add `n` pivots, each with a body at `radius`, and recurse
/// for `levels` - 1 more levels with smaller bodies and orbits.
void AddLevel(REveElement *parent, Int_t levels, Int_t n, Double_t radius, Double_t size)
{
   TRandom &r = *gRandom;
   for (Int_t i = 0; i < n; ++i) {
      auto pivot = new REveElement(Form("%s pivot %d", parent->GetCName(), i));
      RandomOrientation(pivot->RefMainTrans());
      parent->AddElement(pivot);
      AddSpinner(pivot, r.Uniform(0.2, 0.6) * (r.Rndm() < 0.5 ? -1 : 1));

      auto body = new REveGeoShape(Form("%s body %d", parent->GetCName(), i));
      body->SetShape(RandomShape(size));
      body->SetMainColor(kColors[r.Integer(std::size(kColors))]);
      body->RefMainTrans().SetPos(radius * r.Uniform(0.7, 1.3), 0, 0);
      RandomOrientation(body->RefMainTrans());
      pivot->AddElement(body);
      AddSpinner(body, r.Uniform(0.5, 1.5) * (r.Rndm() < 0.5 ? -1 : 1));

      if (levels > 1)
         AddLevel(body, levels - 1, n, 0.35 * radius, 0.4 * size);
   }
}

/// Advances every spinner to the current angle and sends the spin again.
class OrbitRefresher : public TTimer {
public:
   OrbitRefresher() : TTimer(Long_t(1000 * kRefreshSec), kTRUE) {}

   Bool_t Notify() override
   {
      REveManager::ChangeGuard ch;
      const Double_t now = REveUtil::ServerTimeMs();
      for (auto &s : gSpinners) {
         REveTrans &t = s.fEl->RefMainTrans();
         t = s.fBase;
         t.RotateLF(1, 2, s.fRate * (now - s.fT0Ms) / 1000.);
         t.SetMotion(REveVectorD(), REveVectorD(), REveVectorD(0, 0, 1), s.fRate, 2 * kRefreshSec);
         s.fEl->StampTrans();
      }
      Reset();
      return kTRUE;
   }
};

} // namespace

void orbits(Int_t levels = 3, Int_t n = 6, UInt_t seed = 4357)
{
   auto eveMng = REveManager::Create();
   eveMng->AllowMultipleRemoteConnections(false, false);

   gRandom->SetSeed(seed);
   gSpinners.clear();

   auto system = new REveElement("Orbits");
   eveMng->GetEventScene()->AddElement(system);
   AddLevel(system, levels, n, 1000, 150);

   eveMng->Show();

   auto timer = new OrbitRefresher;
   timer->Notify();
   timer->TurnOn();
}
