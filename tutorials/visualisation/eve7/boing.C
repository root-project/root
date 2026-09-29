/// \file
/// \ingroup tutorial_eve_7
/// The Amiga Boing ball, bouncing inside the 3D axis box, driven from the server.
///
/// Each timer tick sends only the transformations of the ball and its shadow.
/// The ball also carries a trajectory, set with REveTrans::SetMotion(), which
/// the client extrapolates between updates. The shadow carries none and steps
/// to each new position. Run with a long period, e.g. boing(500), to see the
/// difference. Select the ball to edit its REveSMorph parameters while it
/// bounces. The Motion panel of the viewer's editor sets the update and redraw
/// rates and switches the extrapolation off.
///
/// \macro_code
///
/// \author Matevz Tadel

#include <ROOT/REveManager.hxx>
#include <ROOT/REveScene.hxx>
#include <ROOT/REveSMorph.hxx>
#include <ROOT/REveViewer.hxx>
#include <ROOT/REveTrans.hxx>

#include <TTimer.h>
#include <TMath.h>

#include <chrono>

using namespace ROOT::Experimental;

// Room half-extents and ball radius. Y is up, which is the up axis of the
// default REve camera, so the scene needs no camera setup.
const Float_t kBX = 40, kBY = 30, kBZ = 40;
const Float_t kR  = 8;

////////////////////////////////////////////////////////////////////////////////
/// Moves the ball and its shadow on every timer tick.
///
/// Each tick integrates the motion and sets two transformation matrices with
/// SetTransMatrix(). No geometry is rebuilt, so the cost per tick does not
/// depend on how finely the ball is tessellated.

class Boinger : public TTimer {
   REveSMorph *fBall{nullptr};
   REveSMorph *fShadow{nullptr};

   Double_t fX{0}, fY{kBY - kR}, fZ{0};       // position; elastic, so fY is the apex
   Double_t fVx{34}, fVy{0}, fVz{21};         // velocity; fVy is the falling one
   Double_t fSpin{0};                         // angle about the ball's polar axis

   std::chrono::steady_clock::time_point fLast{std::chrono::steady_clock::now()};
   std::chrono::steady_clock::time_point fT0{std::chrono::steady_clock::now()};
   int fSent{0};

   static constexpr Double_t kGrav = -160;    // units / s^2, along -y
   static constexpr Double_t kTilt = 0.30;    // polar axis tipped out of vertical
   static constexpr Double_t kSpinRate = 2.4; // rad / s

   /// Time until the ball next hits a wall, the floor or the ceiling, in seconds.
   /// SetMotion() gets it as the time the trajectory may be trusted, because a
   /// bounce is the one change the client cannot predict.
   Double_t TimeToNextBounce() const
   {
      Double_t t = 1e9;

      auto linear = [&](Double_t p, Double_t v, Double_t lim) {
         if (v > 1e-9)       t = TMath::Min(t, ( lim - p) / v);
         else if (v < -1e-9) t = TMath::Min(t, (-lim - p) / v);
      };
      linear(fX, fVx, kBX - kR);
      linear(fZ, fVz, kBZ - kR);

      // y: solve 0.5*g*t^2 + v*t + (p - lim) = 0 for the next positive root,
      // against whichever of floor or ceiling it is heading for.
      const Double_t ylim = kBY - kR;
      for (Double_t lim : {ylim, -ylim}) {
         Double_t c = fY - lim, b = fVy, a = 0.5 * kGrav;
         Double_t disc = b * b - 4 * a * c;
         if (disc < 0) continue;
         Double_t sq = TMath::Sqrt(disc);
         for (Double_t r : {(-b + sq) / (2 * a), (-b - sq) / (2 * a)})
            if (r > 1e-6) t = TMath::Min(t, r);
      }

      // Cap the window at 2 s, so that the ball stops soon after the timer does.
      return TMath::Min(t, 2.0);
   }

   /// Reflects p and v off the wall at +-lim. The bounce is perfectly elastic,
   /// so the ball returns to the same height every time.
   static void Bounce(Double_t &p, Double_t &v, Double_t lim)
   {
      if (p > lim)       { p = 2 * lim - p;  v = -TMath::Abs(v); }
      else if (p < -lim) { p = -2 * lim - p; v =  TMath::Abs(v); }
   }

public:
   Boinger(REveSMorph *ball, REveSMorph *shadow, Long_t ms)
      : TTimer(ms, kTRUE), fBall(ball), fShadow(shadow)
   {}

   int GetSent() const { return fSent; }

   Bool_t Notify() override
   {
      // The timer needs no throttling of its own. A transformation-only change
      // goes out on the motion channel, which skips a client that has not yet
      // taken the previous message. Each message carries the absolute position,
      // so the next one replaces a skipped one.
      ++fSent;

      // Integrate on the wall clock rather than the timer period, because
      // timer ticks can arrive late.
      auto now = std::chrono::steady_clock::now();
      Double_t dt = std::chrono::duration<double>(now - fLast).count();
      fLast = now;
      // Clamp dt, so that a stall of the event loop does not make the ball jump.
      if (dt > 0.1) dt = 0.1;

      // Report the tick rate every 100 ticks. Compare it with the timer period
      // to see whether the event loop keeps up.
      if (fSent % 100 == 0) {
         Double_t el = std::chrono::duration<double>(now - fT0).count();
         ::Info("boing", "%d ticks, %.1f/s over %.1f s", fSent, fSent / el, el);
      }

      // Integrate in fixed 5 ms sub-steps, so that the simulation does not
      // depend on the timer period. One step over a long period can carry the
      // ball past a wall, and the reflection then adds energy.
      for (Double_t rem = dt; rem > 0; ) {
         const Double_t h = TMath::Min(rem, 0.005);
         rem -= h;

         fVy += kGrav * h;
         fX += fVx * h;  fY += fVy * h;  fZ += fVz * h;

         Bounce(fX, fVx, kBX - kR);
         Bounce(fY, fVy, kBY - kR);
         Bounce(fZ, fVz, kBZ - kR);

         fSpin += kSpinRate * h;
      }


      // The ball spins at a constant rate about its polar axis, which for an
      // REveSMorph is the local x. The axis is stood up and tilted by kTilt
      // from vertical. The spin is independent of the flight, so the ball does
      // not roll.
      Double_t cs = TMath::Cos(fSpin), sn = TMath::Sin(fSpin);
      // Rotate the polar axis by a quarter turn to stand it up (x -> y), then
      // lean it over by kTilt.
      Double_t al = TMath::PiOver2() + kTilt;
      Double_t ct = TMath::Cos(al), st = TMath::Sin(al);

      // Columns of Rz(al) * Rx(spin), scaled to the radius.
      Double_t e1[3] = {  ct,       st,      0   };
      Double_t e2[3] = { -st * cs,  ct * cs, sn  };
      Double_t e3[3] = {  st * sn, -ct * sn, cs  };

      REveManager::ChangeGuard ch;

      REveTrans t;
      t.SetBaseVec(1, kR * e1[0], kR * e1[1], kR * e1[2]);
      t.SetBaseVec(2, kR * e2[0], kR * e2[1], kR * e2[2]);
      t.SetBaseVec(3, kR * e3[0], kR * e3[1], kR * e3[2]);
      t.SetPos(fX, fY, fZ);
      fBall->SetTransMatrix(t.Array());

      // Declare the trajectory as well as the position. The client evaluates it
      // on its own frame clock, so the ball moves smoothly between updates.
      // Under constant gravity the second-order form is the exact path. The
      // trajectory is trusted until the next bounce, after which the client
      // stops extrapolating until the next update.
      REveVectorD vel(fVx, fVy, fVz);
      REveVectorD acc(0., kGrav, 0.);

      // The spin axis is in the ball's local frame, so it is the same (1,0,0)
      // on every update. The orientation is already in the matrix.
      REveVectorD spin_axis(1., 0., 0.);

      fBall->RefMainTrans().SetMotion(vel, acc, spin_axis, kSpinRate,
                                      TimeToNextBounce());

      // The shadow: a shallow dome under the ball that shrinks as the ball
      // rises. It gets no SetMotion(), so the client moves it to each new
      // matrix as it arrives, while the ball moves smoothly in between.
      //
      // It is a hemisphere (SetThetaMax(0.5)) flattened along its polar axis.
      // A flattened whole sphere would z-fight with itself. The basis is set by
      // hand because the polar axis, the local x, has to point up.
      Double_t h = (fY + kBY) / (2 * kBY);        // 0 at the floor, 1 at the ceiling

      // Never wider than the ball, so the shadow stays inside the room when the
      // ball is at a wall.
      Double_t s = kR * (1.0 - 0.3 * h);

      REveTrans sh;
      sh.SetBaseVec(1, 0, 0.02 * kR, 0);          // polar axis up, and squashed
      sh.SetBaseVec(2, s, 0, 0);
      sh.SetBaseVec(3, 0, 0, s);
      // Raise the shadow by half its flattened thickness, 0.02 * kR, so the
      // bottom of its bounding box is on the floor. REveSMorph's box spans the
      // whole sphere, so its lower half is below the drawn dome.
      sh.SetPos(fX, -kBY + 0.02 * kR, fZ);
      fShadow->SetTransMatrix(sh.Array());

      Reset();
      return kTRUE;
   }
};

void boing(Long_t period_ms = 40)
{
   auto eveMng = REveManager::Create();
   eveMng->AllowMultipleRemoteConnections(false, false);

   // The edge axes frame the room and carry the scale.
   auto viewer = eveMng->GetDefaultViewer();
   viewer->SetAxesType(REveViewer::kAxesEdge);
   // Y is up, so the axes rule the floor, the surface the ball bounces off.
   viewer->SetAxesUpAxis(1);
   // Make the axes span the room. Otherwise they span the scene content, which
   // here is only the ball and its shadow.
   viewer->SetAxesBBox(-kBX, -kBY, -kBZ, kBX, kBY, kBZ);

   auto scene = eveMng->GetEventScene();


   auto ball = new REveSMorph("Boing ball");
   ball->SetTLevel(32);
   ball->SetPLevel(48);
   ball->SetTexture("checker_8.png");
   ball->SetMainColor(kWhite);
   ball->SetPickable(kTRUE);
   // Size it before the first frame, or the unit-size surface shows at the
   // origin for the one tick before the timer first fires.
   ball->SetRadius(kR);
   scene->AddElement(ball);

   auto shadow = new REveSMorph("Shadow");
   shadow->SetTLevel(6);
   shadow->SetPLevel(32);
   shadow->SetThetaMax(0.5);      // a hemisphere; see the comment in Notify()
   shadow->SetMainColor(kBlack);
   // Fairly opaque, because the lit surface's specular highlight lightens even
   // a black shadow.
   shadow->SetMainTransparency(20);
   shadow->SetPickable(kFALSE);
   scene->AddElement(shadow);

   eveMng->Show();

   (new Boinger(ball, shadow, period_ms))->TurnOn();
}
