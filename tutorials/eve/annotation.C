/// \file
/// \ingroup tutorial_eve
/// Demonstrates usage of TGLAnnotation class.
///
/// \image html eve_annotation.png
/// \macro_code
///
/// \author Alja Mrak-Tadel

#include <TEveManager.h>
#include <TGLViewer.h>
#include <TGLAnnotation.h>
#include <TEveBox.h>
#include <TDatime.h>
#include <TTimer.h>
#include <TDatime.h>

class MyTimer : public TTimer
{
private:
   TGLAnnotation* m_label;

public:
   MyTimer(TGLAnnotation* x) : TTimer(1000), m_label(x)
   {
   }

   virtual Bool_t Notify()
   {
      // stop timer
      TurnOff();

      // so some action here
      TDatime d;
      m_label->SetText(d.AsString());
      gEve->GetDefaultGLViewer()->RequestDraw();

      // start timer
      SetTime(1000);
      Reset();
      TurnOn();
      return true;
   }
};

void annotation(Float_t a=10, Float_t d=5, Float_t x=0, Float_t y=0, Float_t z=0)
{
   TEveManager::Create();

   // add a box in scene
   auto b = new TEveBox("Box", "Test Title");
   b->SetMainColor(kCyan);
   b->SetMainTransparency(0);
   b->SetVertex(0, x - a, y - a, z - a);
   b->SetVertex(1, x - a, y + a, z - a);
   b->SetVertex(2, x + a, y + a, z - a);
   b->SetVertex(3, x + a, y - a, z - a);
   b->SetVertex(4, x - a, y - a, z + a);
   b->SetVertex(5, x - a, y + a, z + a);
   b->SetVertex(6, x + a, y + a, z + a);
   b->SetVertex(7, x + a, y - a, z + a);
   gEve->AddElement(b);
   gEve->Redraw3D(kTRUE);

   // add overlay text
   auto v = gEve->GetDefaultGLViewer();
   TDatime time;
   auto ann = new TGLAnnotation(v, time.AsString(), 0.1, 0.9);
   ann->SetTextSize(0.1);// % of window diagonal

   // set timer to update text every second
   auto timer = new MyTimer(ann);
   timer->SetTime(1000);
   timer->Reset();
   timer->TurnOn();
}
