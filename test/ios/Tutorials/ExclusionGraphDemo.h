#ifndef ROOT_ExclusionGraphDemo
#define ROOT_ExclusionGraphDemo

#include <memory>

#include "DemoBase.h"

class TMultiGraph;

namespace ROOT {
namespace iOS {
namespace Demos {

class ExclusionGraphDemo : public DemoBase {
private:
   enum {
      kNPoints = 35
   };
public:
   ExclusionGraphDemo();
   ~ExclusionGraphDemo();

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

   bool Supports3DRotation() const {return false;}
private:
   std::auto_ptr<TMultiGraph> fMultiGraph;

   ExclusionGraphDemo(const ExclusionGraphDemo &rhs);
   ExclusionGraphDemo &operator = (const ExclusionGraphDemo &rhs);
};

}
}
}

#endif
