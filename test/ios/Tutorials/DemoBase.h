#ifndef ROOT_DemoBase
#define ROOT_DemoBase

namespace ROOT {
namespace iOS {

class Pad;

namespace Demos {

class DemoBase {
public:
   virtual ~DemoBase();

   virtual void ResetDemo() = 0;
   virtual bool IsAnimated() const = 0;
   virtual unsigned NumOfFrames() const = 0;
   virtual double AnimationTime() const = 0;
   virtual void StartAnimation() = 0;
   virtual void NextStep() = 0;
   virtual void StopAnimation() = 0;

   virtual void AdjustPad(Pad *pad) = 0;

   virtual void PresentDemo() = 0;

   virtual bool Supports3DRotation() const = 0;
};

}
}
}

#endif
