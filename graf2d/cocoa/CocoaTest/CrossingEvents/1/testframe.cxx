#include <iostream>
#include <cassert>

#include "TVirtualX.h"
#include "RConfigure.h"

#include "testframe.h"

namespace ROOT {
namespace CocoaTest {

FontStruct_t TestFrame::font_ = kNone;
GContext_t TestFrame::textContext_ = kNone;

//_____________________________________________________
TestFrame::TestFrame(TestFrame *parent, UInt_t width, UInt_t heihght,
                     UInt_t options, Pixel_t background)
               : TGFrame(parent, width, heihght, options, background)
{
   std::cout<<"TestFrame::TestFrame:\n";
   PrintFrameInfo();

   if (font_ == kNone) {//Init font.
      assert(textContext_ == kNone);
#ifdef R__HAS_COCOA      
      font_ = gVirtualX->LoadQueryFont("-*-courier-*-*-*-*-14");
#else
      font_ = gVirtualX->LoadQueryFont("fixed");
#endif
      GCValues_t gcVals;
      gcVals.fFont = gVirtualX->GetFontHandle(font_);
      
      gcVals.fMask = kGCFont | kGCForeground;
      textContext_ = gVirtualX->CreateGC(GetId(), &gcVals);
   }
}

//_____________________________________________________
TestFrame::~TestFrame()
{
   std::cout<<"TestFrame::~TestFrame:\n";
   PrintFrameInfo();
}

//_____________________________________________________
void TestFrame::DoRedraw()
{
   TGFrame::DoRedraw();
   
   const TString text(TString::Format("id : %u, w : %u, h : %u", unsigned(GetId()), unsigned(GetWidth()), unsigned(GetHeight())));
   
   gVirtualX->DrawString(GetId(), textContext_, 0, 30, text.Data(), text.Length());
}

//_____________________________________________________
Bool_t TestFrame::HandleButton(Event_t *btnEvent)
{
   assert(btnEvent);
   std::cout<<"Button event:\n";
   PrintFrameInfo();
   PrintEventCoordinates(btnEvent);

   return kTRUE;
}

//_____________________________________________________
Bool_t TestFrame::HandleCrossing(Event_t *crossingEvent)
{
   assert(crossingEvent);
   
   if (crossingEvent->fType == kEnterNotify)
      std::cout<<"Enter notify event:\n";
   else
      std::cout<<"Leave notify event:\n";

   PrintFrameInfo();
   
   return kTRUE;
}

//_____________________________________________________
Bool_t TestFrame::HandleMotion(Event_t *motionEvent)
{
   assert(motionEvent);
   std::cout<<"Motion event:\n";
   PrintFrameInfo();
   
   return kTRUE;
}

//_____________________________________________________
void TestFrame::PrintFrameInfo()const
{
   std::cout<<"this == "<<this<<" window id == "<<GetId()<<std::endl;
}

//_____________________________________________________
void TestFrame::PrintEventCoordinates(const Event_t *event)const
{
   assert(event);

   std::cout<<"event.x == "<<event->fX<<" event.y == "<<event->fY<<std::endl;
   std::cout<<"event.xroot == "<<event->fXRoot<<" event.yroot == "<<event->fYRoot<<std::endl;
}

}
}
