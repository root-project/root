#ifndef ROOT_PolarGraphDemo
#define ROOT_PolarGraphDemo

#include <memory>

#include "DemoBase.h"

class TGraphPolar;

namespace ROOT {
namespace iOS {
namespace Demos {

class PolarGraphDemo : public DemoBase {
private:
   enum {
      kNPointsAFL = 1000,
      kNPointsCP = 20
   };
public:
   PolarGraphDemo();
   ~PolarGraphDemo();
   
   //overriders.
   void ResetDemo() {}
   bool IsAnimated()const {return false;}
   unsigned NumOfFrames() const {return 1;}
   double AnimationTime() const {return 0.;}

   void StartAnimation() {}
   void NextStep() {}
   void StopAnimation() {}

   void AdjustPad(Pad *);

   void PresentDemo();
   
   bool Supports3DRotation() const {return false;}
private:
   std::auto_ptr<TGraphPolar> fPolarAFL; //polar graph with draw option "AFL"
   std::auto_ptr<TGraphPolar> fPolarCP;  //polar graph with draw option "CP"
   
   PolarGraphDemo(const PolarGraphDemo &rhs);
   PolarGraphDemo &operator = (const PolarGraphDemo &rhs);
};

}
}
}

#endif
