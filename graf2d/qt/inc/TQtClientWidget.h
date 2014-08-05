// @(#)root/qt:$Id$
/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * Copyright (C) 2002 by Valeri Fine.                                    *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TQtClientWidget
#define ROOT_TQtClientWidget


#ifndef __CINT__
# include <QKeySequence>
# include <QFrame>
# include <QCursor>
# include <map>
#else
  class QFrame;
  class QColor;
  class QPixmap;
#endif

#include "GuiTypes.h"

//________________________________________________________________________
//
// TQtClientWidget  is a QFrame implemantation backing  ROOT TGWindow objects
// It tries to mimic the X11 Widget behaviour, that kind the ROOT Gui relies on heavily.
//________________________________________________________________________

class QCursor;
class QCloseEvent;
class QPaintEvent;
class TQtClientGuard;
class TQtWidget;
class QShortcut;
class TGWindow;

class TQtClientWidget: public QFrame {
#ifndef __CINT__
     Q_OBJECT
#endif

private:
         void  operator=(const TQtClientWidget&);
         TQtClientWidget(const TQtClientWidget&);
protected:

       UInt_t fGrabButtonMask;        // modifier button mask for TVirtualX::GrabButton
       UInt_t fGrabEventPointerMask;  // input event mask for TVirtualX::GrabPointer
       UInt_t fGrabEventButtonMask;   // input event mask for TVirtualX::GrabButton
       UInt_t fSelectEventMask;       // input mask for SelectInput
       UInt_t fSaveSelectInputMask;   // To save dutinr the grabbing the selectInput
       EMouseButton fButton;
#ifndef __CINT__
       std::map<QKeySequence,QShortcut*>  fGrabbedKey;
#endif
       Bool_t   fPointerOwner;
       QCursor *fNormalPointerCursor;
       QCursor *fGrabPointerCursor;
       QCursor *fGrabButtonCursor;
       bool     fIsClosing;
       bool     fDeleteNotify;
       TQtClientGuard  *fGuard;
       TQtWidget       *fCanvasWidget;
       TGWindow *fMyRootWindow;  // back pointer to the host window object
       QColor   *fEraseColor;    // Color to paint widget background with our PainEvent
       QPixmap  *fErasePixmap;   // pixmap to paint widget background with our PainEvent
       friend class TQtClientGuard;
       friend class TGQt;
#ifndef __CINT__
      TQtClientWidget(TQtClientGuard *guard, QWidget* parent=0, const char* name=0, Qt::WFlags f=0);
#else
      TQtClientWidget(TQtClientGuard *guard, QWidget* parent=0, const char* name=0, WFlags f=0);
#endif
      void SetCanvasWidget(TQtWidget *widget);
      virtual void paintEvent       ( QPaintEvent * );
public:
    enum {kRemove = -1, kTestKey = 0, kInsert = 1};
    virtual ~TQtClientWidget();
    virtual void closeEvent(QCloseEvent *ev);
    virtual void setEraseColor(const QColor &color);
    virtual void setErasePixmap (const QPixmap &pixmap);
    bool   DeleteNotify();
    TQtWidget *GetCanvasWidget() const;
    void   GrabEvent(Event_t &ev,bool own=TRUE);
//    Q3Accel *HasAccel() const ;
    bool   IsClosing();
    bool   IsGrabbed       (Event_t &ev);
    bool   IsGrabPointerSelected(UInt_t evmask) const;
    bool   IsGrabButtonSelected (UInt_t evmask) const;
    TQtClientWidget *IsKeyGrabbed(const Event_t &ev);
    UInt_t IsEventSelected (UInt_t evmask) const;
    bool   IsGrabOwner()   { return fPointerOwner;}
    void   SetAttributeEventMask(UInt_t evmask);
    void   SetButtonMask   (UInt_t modifier=kAnyModifier,EMouseButton button=kAnyButton);
    void   SetClosing(bool flag=kTRUE);
    void   SetCursor();
    void   SetCursor(Cursor_t cursor);
    void   SetDeleteNotify(bool flag=true);
    void   SetButtonEventMask(UInt_t evmask,Cursor_t cursor=0);
    void   SelectInput       (UInt_t evmask);
    Bool_t SetKeyMask        (Int_t keycode = 0, UInt_t modifier=kAnyModifier,int insert=kInsert);
    void   UnSetButtonMask   (bool dtor=false);
    void   UnSetKeyMask(Int_t keycode = 0, UInt_t modifier=kAnyModifier);
    QCursor *GrabButtonCursor() const;
    QCursor *GrabPointerCursor() const;
    UInt_t ButtonMask  ()    const;
    UInt_t ButtonEventMask() const;
    UInt_t SelectEventMask() const;
    EMouseButton Button()    const;
    UInt_t PointerMask ()    const;
#ifndef __CINT__
protected slots:
      void Disconnect();
#endif
public slots:
    virtual void Accelerate();
#ifndef Q_MOC_RUN
    ClassDef(TQtClientWidget,0) // QFrame implementation backing  ROOT TGWindow objects
#endif
};

//______________________________________________________________________________
inline bool TQtClientWidget::DeleteNotify(){return fDeleteNotify; }

//______________________________________________________________________________
inline TQtWidget *TQtClientWidget::GetCanvasWidget() const
{ return fCanvasWidget;}
//______________________________________________________________________________
//inline Q3Accel *TQtClientWidget::HasAccel() const
//{  return fGrabbedKey; }

//______________________________________________________________________________
inline bool  TQtClientWidget::IsClosing(){ return fIsClosing; }

//______________________________________________________________________________
inline UInt_t TQtClientWidget::IsEventSelected (UInt_t evmask) const
{
   //if (evmask & (kButtonPressMask | kButtonMotionMask) )
   //   fprintf(stderr,"TQtClientWidget::IsEventSelected event %x, mask %x. match %x\n"
   //   , evmask, fSelectEventtMask, evmask & (kButtonPressMask | kButtonMotionMask));
   return  (evmask & fSelectEventMask); //  || (IsGrabPointerSelected(evmask)) ;
}

//______________________________________________________________________________
inline void TQtClientWidget::SetCursor()
{ // Set this widget pre-defined cursor
   if (fNormalPointerCursor) setCursor(*fNormalPointerCursor);
}
//______________________________________________________________________________
inline void TQtClientWidget::SetCursor(Cursor_t crsr)
{
   // Change the pre-define curos shape and set it
   fNormalPointerCursor = (QCursor *)crsr;
   SetCursor();
}

//______________________________________________________________________________
inline void  TQtClientWidget::SetClosing(bool flag) { fIsClosing = flag;}
//______________________________________________________________________________
inline void  TQtClientWidget::SetDeleteNotify(bool flag){fDeleteNotify = flag;}

//______________________________________________________________________________
inline void TQtClientWidget::SetAttributeEventMask(UInt_t evmask) { SelectInput (evmask);}

//______________________________________________________________________________
inline void TQtClientWidget::SetButtonEventMask(UInt_t evmask,Cursor_t crsr)
{ fGrabEventButtonMask = evmask; fGrabButtonCursor =(QCursor *) crsr; }

//______________________________________________________________________________
inline EMouseButton TQtClientWidget::Button() const { return fButton;          }

//______________________________________________________________________________
inline UInt_t TQtClientWidget::ButtonEventMask() const { return fGrabEventButtonMask;}

//______________________________________________________________________________
inline UInt_t TQtClientWidget::ButtonMask()  const { return fGrabButtonMask;   }

//______________________________________________________________________________
inline UInt_t TQtClientWidget::PointerMask() const { return fGrabEventPointerMask;}

//______________________________________________________________________________
inline UInt_t TQtClientWidget::SelectEventMask() const {return fSelectEventMask;}

//______________________________________________________________________________
inline QCursor *TQtClientWidget::GrabButtonCursor() const
{      return   fGrabButtonCursor;                                                }

//______________________________________________________________________________
inline QCursor *TQtClientWidget::GrabPointerCursor() const
{      return   fGrabPointerCursor;                                                }

//______________________________________________________________________________
inline bool TQtClientWidget::IsGrabPointerSelected(UInt_t evmask) const
{  return  evmask & PointerMask(); }

//______________________________________________________________________________
inline bool  TQtClientWidget::IsGrabButtonSelected (UInt_t evmask) const
{ return  evmask & ButtonEventMask(); }

#endif
