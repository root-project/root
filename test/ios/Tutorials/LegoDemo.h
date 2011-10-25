#ifndef ROOT_LegoDemo
#define ROOT_LegoDemo

#include <memory>

#include "DemoBase.h"

class TF2;

namespace ROOT {
namespace iOS {
namespace Demos {

class LegoDemo : public DemoBase {
public:
   LegoDemo();
   ~LegoDemo();
   
   //overriders.
   void ResetDemo() {}
   bool IsAnimated()const {return false;}
   unsigned NumOfFrames()const {return 1;}
   double AnimationTime()const {return 0.;}
   
   void StartAnimation(){}
   void NextStep(){}
   void StopAnimation(){}

   void AdjustPad(Pad *pad);
   
   void PresentDemo();
   
   bool Supports3DRotation() const {return true;}
private:
   std::auto_ptr<TF2> fLego;
   
   LegoDemo(const LegoDemo &rhs);
   LegoDemo &operator = (const LegoDemo &rhs);

};

}
}
}

#endif
