//
//  HsimpleDemo.h
//  Tutorials
//
//  Created by Timur Pocheptsov on 7/10/11.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
//
#ifndef ROOT_HsimpleDemo
#define ROOT_HsimpleDemo

#include <memory>

#ifndef ROOT_DemoBase
#include "DemoBase.h"
#endif

class TH1F;

namespace ROOT {
namespace iOS {
namespace Demos {

class HsimpleDemo : public DemoBase {
public:
   HsimpleDemo();
   ~HsimpleDemo();
   
   //Overriders.
   void ResetDemo();
   bool IsAnimated()const;
   
   unsigned NumOfFrames() const;
   double AnimationTime() const;

   void StartAnimation();
   void NextStep();
   void StopAnimation();

   void AdjustPad(Pad *pad);

   void PresentDemo();
   
   bool Supports3DRotation() const {return false;}

private:
   std::auto_ptr<TH1F> fHist;
   
   HsimpleDemo(const HsimpleDemo &rhs);
   HsimpleDemo &operator = (const HsimpleDemo &rhs);
};

}
}
}

#endif
