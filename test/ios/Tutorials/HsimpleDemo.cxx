#include <stdexcept>

#include "TRandom.h"
#include "TFrame.h"
#include "IOSPad.h"
#include "TH1.h"

#include "HsimpleDemo.h"


namespace ROOT {
namespace iOS {
namespace Demos {

////////////////////////////////////////////////////////////////////////////////

HsimpleDemo::HsimpleDemo()
                  : fHist(new TH1F("hpx", "This is the px distribution", 100, -4.f, 4.f))
{
   if (!gRandom)
      throw std::runtime_error("gRandom is null");

   fHist->SetFillColor(48);
}

////////////////////////////////////////////////////////////////////////////////
///For auto-ptr dtor only.

HsimpleDemo::~HsimpleDemo()
{
}

////////////////////////////////////////////////////////////////////////////////
///Clear old contents of the histogram.

void HsimpleDemo::ResetDemo()
{
   fHist->Reset();
}

////////////////////////////////////////////////////////////////////////////////

Bool_t HsimpleDemo::IsAnimated()const
{
   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////

unsigned HsimpleDemo::NumOfFrames() const
{
   return 25;
}

////////////////////////////////////////////////////////////////////////////////
///0.5 second of animation.

double HsimpleDemo::AnimationTime() const
{
   return 0.5;
}

////////////////////////////////////////////////////////////////////////////////

void HsimpleDemo::StartAnimation()
{
   fHist->Reset();
   gRandom->SetSeed();
}

////////////////////////////////////////////////////////////////////////////////
///Fill histograms randomly (2D Rannor is taken from original code sample).

void HsimpleDemo::NextStep()
{
   Float_t x = 0.f, dummyY = 0.f;

   for (UInt_t i = 0; i < 1000; ++i) {
      gRandom->Rannor(x, dummyY);
      fHist->Fill(x);
   }
}

////////////////////////////////////////////////////////////////////////////////

void HsimpleDemo::StopAnimation()
{
}

////////////////////////////////////////////////////////////////////////////////

void HsimpleDemo::AdjustPad(Pad *pad)
{
   pad->SetFillColor(42);
   pad->GetFrame()->SetFillColor(21);
   pad->GetFrame()->SetBorderSize(6);
   pad->GetFrame()->SetBorderMode(-1);
}

////////////////////////////////////////////////////////////////////////////////

void HsimpleDemo::PresentDemo()
{
   fHist->Draw();
}

}
}
}
