#ifndef ROOT_SurfaceDemo
#define ROOT_SurfaceDemo

#include <memory>

#include "DemoBase.h"

class TF2;

namespace ROOT {
namespace iOS {
namespace Demos {

class SurfaceDemo : public DemoBase {
public:
   SurfaceDemo();
   ~SurfaceDemo();
   
   //overriders.
   void ResetDemo() {}
   bool IsAnimated()const {return false;}
   unsigned NumOfFrames()const {return 1;}
   double AnimationTime()const {return 0.;}
   
   void StartAnimation(){}
   void NextStep(){}
   void StopAnimation(){}

   void AdjustPad(Pad *);

   void PresentDemo();
   
   bool Supports3DRotation() const {return true;}
private:
   std::auto_ptr<TF2> fSurface;
   
   SurfaceDemo(const SurfaceDemo &rhs);
   SurfaceDemo &operator = (const SurfaceDemo &rhs);
};

}
}
}

#endif
