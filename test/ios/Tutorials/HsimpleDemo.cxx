#include <iostream>
#include <stdexcept>

#include "TRandom.h"
#include "TFrame.h"
#include "TH1.h"

#include "HsimpleDemo.h"
#include "IOSPad.h"

namespace ROOT {
namespace iOS {
namespace Demos {

//______________________________________________________________________________
HsimpleDemo::HsimpleDemo()
                  : fHist(new TH1F("hpx", "This is the px distribution", 100, -4.f, 4.f))
{
   if (!gRandom)
      throw std::runtime_error("HsimpleDemo::HsimpleDemo: gRandom is 0");
      
   fHist->SetFillColor(48);
}

//______________________________________________________________________________
HsimpleDemo::~HsimpleDemo()
{
   //For auto-ptr dtor only.
}

//______________________________________________________________________________
void HsimpleDemo::ResetDemo()
{
   //Clear old contents of the histogram.
   fHist->Reset();
}

//______________________________________________________________________________
Bool_t HsimpleDemo::IsAnimated()const
{
   return kTRUE;
}

//______________________________________________________________________________
unsigned HsimpleDemo::NumOfFrames() const
{
   return 25;
}

//______________________________________________________________________________
double HsimpleDemo::AnimationTime() const
{
   //0.5 second of animation.
   return 0.5;
}

//______________________________________________________________________________
void HsimpleDemo::StartAnimation()
{
   fHist->Reset();
   gRandom->SetSeed();
}

//______________________________________________________________________________
void HsimpleDemo::NextStep()
{
   //Fill histograms randomly (2D Rannor is taken from original code sample).
   Float_t x = 0.f, dummyY = 0.f;

   for (UInt_t i = 0; i < 1000; ++i) {
      gRandom->Rannor(x, dummyY);
      fHist->Fill(x);
   }
}

//______________________________________________________________________________
void HsimpleDemo::StopAnimation()
{
}

//______________________________________________________________________________
void HsimpleDemo::AdjustPad(Pad *pad)
{
   pad->SetFillColor(42);
   pad->GetFrame()->SetFillColor(21);
   pad->GetFrame()->SetBorderSize(6);
   pad->GetFrame()->SetBorderMode(-1);
}

//______________________________________________________________________________
void HsimpleDemo::PresentDemo()
{
   fHist->Draw();
}

}
}
}
