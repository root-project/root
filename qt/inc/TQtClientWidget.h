#ifndef ROOT_TQTCLIENTWIDGET
#define ROOT_TQTCLIENTWIDGET
// Author: Valeri Fine   01/03/2003
/****************************************************************************
** $Id: TQtClientWidget.h,v 1.30 2004/07/09 00:17:48 fine Exp $
**
** Copyright (C) 2003 by Valeri Fine. Brookhaven National Laboratory.
**                                    All rights reserved.
**
** This file may be distributed under the terms of the Q Public License
** as defined by Trolltech AS of Norway and appearing in the file
** LICENSE.QPL included in the packaging of this file.      
**
*****************************************************************************/

#include <qframe.h>
#include <qcursor.h>
#include "GuiTypes.h"

// #include "TGQt.h"

class QAccel;
class QCursor;
class QCloseEvent;
class TQtClientGuard;
class TQtWidget;

class TQtClientWidget : public QFrame {
       Q_OBJECT
protected:
       
       UInt_t fGrabButtonMask;
       UInt_t fGrabPointerMask;
       UInt_t fEventMask;
       UInt_t fSelectEventMask;
       EMouseButton fButton;
       QAccel *fGrabbedKey;
       Bool_t fPointerOwner;
       QCursor *fPointerCursor;
       bool   fIsClosing;
       bool   fDeleteNotify;
       TQtClientGuard  *fGuard;
       TQtWidget       *fCanvasWidget;
       friend class TQtClientGuard;
       friend class TGQt;

       TQtClientWidget(TQtClientGuard *guard, QWidget* parent=0, const char* name=0, WFlags f=0 ): 
          QFrame(parent,name,f)
         ,fGrabButtonMask(kAnyModifier),fGrabPointerMask(kAnyModifier)
         ,fEventMask(kNoEventMask),fSelectEventMask(0) // ,fAttributeEventMask(0)
         ,fButton(kAnyButton),fGrabbedKey(0),fPointerOwner(kFALSE),fPointerCursor(0),fIsClosing(false)
         ,fDeleteNotify(false),fGuard(guard), fCanvasWidget(0)
          { }
       void SetCanvasWidget(TQtWidget *widget);
public: 
    virtual ~TQtClientWidget();
    virtual void closeEvent(QCloseEvent *ev);  
    bool   DeleteNotify();
    TQtWidget *GetCanvasWidget() const;
    void   GrabEvent(Event_t &ev,bool own=TRUE);
    bool   IsClosing();
    bool   IsKeyGrabbed    (Event_t &ev);
    bool   IsGrabbed       (Event_t &ev);
    bool   IsPointerGrabbed(Event_t &ev);
    UInt_t IsEventSelected (UInt_t evmask);
    bool   IsGrabOwner()   { return fPointerOwner;}
    void   SetAttributeEventMask(UInt_t evmask);
    void   SetButtonMask   (UInt_t modifier=kAnyModifier,EMouseButton button=kAnyButton);
    void   SetClosing(bool flag=kTRUE);
    void   SetCursor();
    void   SetCursor(Cursor_t cursor);
    void   SetDeleteNotify(bool flag=true);
    void   SetEventMask    (UInt_t evmask);
    void   SelectInput     (UInt_t evmask);
    void   SetPointerMask  (UInt_t modifier, Cursor_t cursor, Bool_t owner_events);
    void   SetKeyMask      (Int_t keycode = 0, UInt_t modifier=kAnyModifier,bool insert=true);
    void   UnSetButtonMask (bool dtor=false);
    void   UnSetPointerMask(bool dtor=false);
    void   UnSetKeyMask(Int_t keycode = 0, UInt_t modifier=kAnyModifier);
    UInt_t ButtonMask  () const;
    EMouseButton Button() const ;
    UInt_t PointerMask () const;
    UInt_t KeyMask()      const;
    Int_t  KeyCode()      const;
protected slots:
      void Disconnect();
public slots:
    virtual void Accelerate(int id);
    virtual void polish();
};
//______________________________________________________________________________
inline  bool TQtClientWidget::IsPointerGrabbed(Event_t &ev)
{
   //            
   //    grab     ( -owner_event && id == current id  -)         *
   //  o------>o---------------------------------->o--------------->o-->
   //          |             *                     |  grab pointer  |
   //          |                                   |                | 
   //          |           evmask                  |                |
   //          |---------------------------------->|                |
   //          |             *                                      |
   //          |                      *                             |
   //          |--------------------------------------------------->|
   //                             discard event
   //      

   bool isGrabbed = ev.fState & fGrabPointerMask;
   //fprintf(stderr," TQtClientWidget::IsPointerGrabbed %p grabbed=%d\n", this, isGrabbed);
   //fprintf(stderr,"                                 wid= %p mask=0x%x\n", (TQtClientWidget *)TGQt::wid(ev.fWindow), fGrabPointerMask);
   //fprintf(stderr,"                                 fPointerOwner= %d ev.fState=0x%x\n", fPointerOwner, ev.fState);
   
   return isGrabbed;
}
//______________________________________________________________________________
inline bool TQtClientWidget::DeleteNotify(){return fDeleteNotify; }

//______________________________________________________________________________
inline TQtWidget *TQtClientWidget::GetCanvasWidget() const
{ return fCanvasWidget;}

//______________________________________________________________________________
inline bool  TQtClientWidget::IsClosing(){ return fIsClosing; }

//______________________________________________________________________________
inline UInt_t TQtClientWidget::IsEventSelected (UInt_t evmask)
{ 
   //if (evmask & (kButtonPressMask | kButtonMotionMask) ) 
   //   fprintf(stderr,"TQtClientWidget::IsEventSelected event %x, mask %x. match %x\n"
   //   , evmask, fSelectEventtMask, evmask & (kButtonPressMask | kButtonMotionMask));
   return  (evmask & fSelectEventMask);
}

//______________________________________________________________________________
inline void TQtClientWidget::SetCursor()
{ // Set this widget pre-defined cursor
   if (fPointerCursor) setCursor(*fPointerCursor); 
}
//______________________________________________________________________________
inline void TQtClientWidget::SetCursor(Cursor_t cursor)
{ 
   // Change the pre-define curos shape and set it
   fPointerCursor = (QCursor *)cursor;
   SetCursor();
}

//______________________________________________________________________________
inline void  TQtClientWidget::SetClosing(bool flag) { fIsClosing = flag;}
//______________________________________________________________________________
inline void  TQtClientWidget::SetDeleteNotify(bool flag){fDeleteNotify = flag;}

//______________________________________________________________________________
inline void TQtClientWidget::SetAttributeEventMask(UInt_t evmask) { SelectInput (evmask);}

//______________________________________________________________________________
inline void TQtClientWidget::SetEventMask(UInt_t evmask) { fEventMask = evmask;}

//______________________________________________________________________________
inline void TQtClientWidget::SelectInput (UInt_t evmask) {fSelectEventMask=evmask;}

//______________________________________________________________________________
inline EMouseButton TQtClientWidget::Button()const { return fButton;           }

//______________________________________________________________________________
inline UInt_t TQtClientWidget::ButtonMask()  const { return fGrabButtonMask;   } 

//______________________________________________________________________________
inline UInt_t TQtClientWidget::PointerMask() const { return fGrabPointerMask;  }

#endif

