#ifndef ROOT_TGLRenderArea
#define ROOT_TGLRenderArea

#include <TGFrame.h>

class TGLWindow : public TGCompositeFrame {
private:
   ULong_t fCtx;
public:
   TGLWindow(Window_t id, const TGWindow *parent);
   ~TGLWindow();
   Bool_t HandleConfigureNotify(Event_t *event);//*SIGNAL*
   Bool_t HandleButton(Event_t *event);//*SIGNAL*
   Bool_t HandleKey(Event_t *event);//*SIGNAL*
   Bool_t HandleMotion(Event_t *event);//*SIGNAL*
   Bool_t HandleExpose(Event_t *event);//*SIGNAL*
   void Refresh();
   void MakeCurrent();
private:
   TGLWindow(const TGLWindow &);
   TGLWindow & operator = (const TGLWindow &);

   ClassDef(TGLWindow, 0)
};

class TGLRenderArea {
public:
   TGLRenderArea();
   TGLRenderArea(Window_t wid, const TGWindow *parent);
   virtual ~TGLRenderArea();
   TGLWindow * GetGLWindow()const{ return fArea; }
private:
   TGLRenderArea(const TGLRenderArea &);
   TGLRenderArea & operator = (const TGLRenderArea &);

   TGLWindow * fArea;

   ClassDef(TGLRenderArea, 0)
};

#endif
