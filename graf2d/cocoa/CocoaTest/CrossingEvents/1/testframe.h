#ifndef TESTFRAME_INCLUDED
#define TESTFRAME_INCLUDED

#include "GuiTypes.h"
#include "TGFrame.h"

namespace ROOT {
namespace CocoaTest {

class TestFrame : public TGFrame {
public:
   TestFrame(TestFrame *parent, UInt_t width, UInt_t height,
             UInt_t options, Pixel_t background);
   ~TestFrame();

   void DoRedraw();

   //Event handlers:   
   Bool_t HandleButton(Event_t *btnEvent);
   Bool_t HandleCrossing(Event_t *crossingEvent);
   Bool_t HandleMotion(Event_t *motionEvent);
   

private:
   void PrintFrameInfo()const;
   void PrintEventCoordinates(const Event_t *event)const;

   TestFrame(const TestFrame &rhs) = delete;
   TestFrame(TestFrame &&rhs) = delete;
   TestFrame &operator = (const TestFrame &rhs) = delete;
   TestFrame &operator = (TestFrame &&rhs) = delete;
   
   //To be used by all test frames.
   static FontStruct_t font_;
   static GContext_t textContext_;
};

}
}

#endif
