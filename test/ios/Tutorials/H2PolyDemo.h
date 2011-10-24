#ifndef ROOT_H2PolyDemo
#define ROOT_H2PolyDemo

#include <memory>

#include "DemoBase.h"

class TH2Poly;

namespace ROOT {
namespace iOS {
namespace Demos {

class H2PolyDemo : public DemoBase {
public:
   H2PolyDemo(const char *fileName);
   ~H2PolyDemo();
   
   //overriders.
   void ResetDemo() {}
   bool IsAnimated()const {return false;}
   unsigned NumOfFrames()const {return 1;}
   double AnimationTime()const {return 0.;}
   
   void StartAnimation(){}
   void NextStep(){}
   void StopAnimation(){}

   void AdjustPad(Pad *) {}

   void PresentDemo();
   
   bool Supports3DRotation() const {return false;}
private:
   std::auto_ptr<TH2Poly> fPoly;
   
   H2PolyDemo(const H2PolyDemo &rhs);
   H2PolyDemo &operator = (const H2PolyDemo &rhs);
};

}
}
}

#endif
