// @(#)root/qt:$Name:  $:$Id: TQtWidget.h,v 1.14 2006/03/24 15:31:10 antcheva Exp $
// Author: Valeri Fine   23/01/2003

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * Copyright (C) 2003 by Valeri Fine.                                    *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TQtWidget
#define ROOT_TQtWidget

// Definition of TQtWidget class
// "double-buffered" widget

#include <assert.h>
#include "Rtypes.h"
#include "TCanvas.h"

#ifndef __CINT__
#  include <qwidget.h>
#  if (QT_VERSION > 0x039999)
//Added by qt3to4:
#     include <QMouseEvent>
#     include <QCustomEvent>
#     include <QShowEvent>
#     include <QFocusEvent>
#     include <QKeyEvent>
#     include <QResizeEvent>
#     include <QEvent>
#     include <QPaintEvent>
#  endif
#  include <qpixmap.h>
#else
  // List of the fake classes to the fake RootCint happy.
  class QWidget;
  class QPixmap;
  class QMouseEvent;
  class QFocusEvent;
  class QCustomEvent;
  class QKeyEvent;
  class QShowEvent;
  class QPaintEvent;
  class QResizeEvent;
  class QSize;  
  class QString;
  class QEvent;
  class QSizePolicy;
#endif
  class TApplication;
//
// TQtWidget is a custom QWidget to back ROOT TCanvas.
//
// It can be used within Qt-based program and with Qt Designer as a "regular"
// Qt QWidget to create the Qt widget wihe builtin TCanvas'
//
enum EEventTrackingBits {
       kMousePressEvent       = BIT(0), // emit signal as soon as TCanvas processed mousePressEvent       QMouseEvent
       kMouseMoveEvent        = BIT(1), // emit signal as soon as TCanvas processed mouseMoveEvent        QMouseEvent
       kMouseReleaseEvent     = BIT(2), // emit signal as soon as TCanvas processed mouseReleaseEvent     QMouseEvent
       kMouseDoubleClickEvent = BIT(3), // emit signal as soon as TCanvas processed mouseDoubleClickEvent QMouseEvent
       kKeyPressEvent         = BIT(4), // emit signal as soon as TCanvas processed keyPressEvent         QKeyEvent
       kEnterEvent            = BIT(5), // emit signal as soon as TCanvas processed enterEvent            QEvent
       kLeaveEvent            = BIT(6)  // emit signal as soon as TCanvas processed leaveEvent            QEvent
};

//___________________________________________________________________
class TQtWidgetBuffer : public QPixmap
{
  private:
    QWidget *fWidget;

  public:
    TQtWidgetBuffer(QWidget *w=0) :  QPixmap(), fWidget(w)
    { if (w) resize(w->size()); }
    inline QRect rect () const { return fWidget->rect();}
};
//___________________________________________________________________
class  TQtWidget : public QWidget {
#ifndef __CINT__   
 Q_OBJECT
#endif
private:
#if !defined(_MSC_VER)  || _MSC_VER >= 1310
      void operator=(const TQtWidget&) const {}
#endif
		void operator=(const TQtWidget&) {}
      TQtWidget(const TQtWidget&) :QWidget() {}
   //----- Private bits, clients can only test but not change them
   UInt_t         fBits;       //bit field status word
   enum {
      kBitMask       = 0x00ffffff
   };
public:
   enum {
      kEXITSIZEMOVE,
      kENTERSIZEMOVE,
      kFORCESIZE
   };
#ifndef __CINT__
   TQtWidget( QWidget* parent=0, const char* name=0, Qt::WFlags f=Qt::WStyle_NoBorder, bool embedded=TRUE);
#else
  TQtWidget( QWidget* parent=0);
#endif  
  virtual ~TQtWidget();
  void SetCanvas(TCanvas *c)                 { fCanvas = c;}
  inline TCanvas  *GetCanvas() const         { return fCanvas;}
  inline QPixmap  &GetBuffer()               { return fPixmapID;}
  inline const QPixmap  &GetBuffer()  const  { return fPixmapID;}

  // overloaded methods
  virtual void adjustSize();
  virtual void resize (int w, int h);
  virtual void erase ();
  bool    IsDoubleBuffered() { return fDoubleBufferOn; }
  void    SetDoubleBuffer(bool on=TRUE){ fDoubleBufferOn = on;}
  virtual void SetSaveFormat(const char *format);

protected:
   friend class TGQt;
   TCanvas         *fCanvas;
   TQtWidgetBuffer  fPixmapID; // Double buffer of this widget
   bool        fPaint;
   bool        fSizeChanged;
   bool        fDoubleBufferOn;
   bool        fEmbedded;
   QSize       fSizeHint;
   QWidget    *fWrapper;
   QString     fSaveFormat;
   void SetRootID(QWidget *wrapper);
   QWidget *GetRootID() const;
   virtual void EmitCanvasPainted() { emit CanvasPainted(); }
   TCanvas  *Canvas();
   bool paintFlag(bool mode=TRUE);
   void AdjustBufferSize();

   // overloaded QWidget methods
   bool paintingActive () const;

   virtual void enterEvent       ( QEvent *      );
   virtual void customEvent      ( QCustomEvent *);
   virtual void focusInEvent     ( QFocusEvent * );
   virtual void focusOutEvent    ( QFocusEvent * );
   virtual void leaveEvent       ( QEvent *      );
   virtual void mouseDoubleClickEvent(QMouseEvent* );
   virtual void mouseMoveEvent   ( QMouseEvent * );
   virtual void mousePressEvent  ( QMouseEvent * );
   virtual void mouseReleaseEvent( QMouseEvent * );
   virtual void keyPressEvent    ( QKeyEvent *   );
   virtual void keyReleaseEvent  ( QKeyEvent *   );
   virtual void showEvent        ( QShowEvent *  );
   virtual void paintEvent       ( QPaintEvent * );
   virtual void resizeEvent      ( QResizeEvent *);
   //  Layout methods:
   virtual void        SetSizeHint (const QSize &size);
public:
   virtual QSize       sizeHint () const;        // returns the preferred size of the widget.
   virtual QSize       minimumSizeHint () const; // returns the smallest size the widget can have.
   virtual QSizePolicy sizePolicy () const;      // returns a QSizePolicy; a value describing the space requirements of the
protected:
   // -- A special event handler
   virtual void exitSizeEvent ();
   virtual void stretchWidget(QResizeEvent *e);
public:
   //----- bit manipulation (a'la TObject )
   void     SetBit     (UInt_t f, Bool_t set);
   void     SetBit     (UInt_t f);
   void     ResetBit   (UInt_t f);
   Bool_t   TestBit    (UInt_t f) const;
   Int_t    TestBits   (UInt_t f) const;
   void     InvertBit  (UInt_t f);
   void     EmitSignal (UInt_t f);
   void     EmitTestedSignal();
   UInt_t   GetAllBits() const;
   void     SetAllBits(UInt_t f);
   
public:
   // Static method to inmitate ROOT as needed
   static TApplication *InitRint(Bool_t prompt=kFALSE, const char *appClassName="QtRint", int *argc=0, char **argv=0,
          void *options = 0, int numOptions = 0, Bool_t noLogo = kFALSE);
   //  Proxy methods to access the TCanvas selected TObject 
   //  and last processed ROOT TCanvas event
   Int_t             GetEvent()       const;
   Int_t             GetEventX()      const;
   Int_t             GetEventY()      const;
   TObject          *GetSelected()    const;
   Int_t             GetSelectedX()   const;
   Int_t             GetSelectedY()   const;
   TVirtualPad      *GetSelectedPad() const;

   //----- bit Qt signal emitting the Qt signal to track mouse movement
   void     EnableSignalEvents  (UInt_t f);
   void     DisableSignalEvents (UInt_t f);
   Bool_t   IsSignalEventEnabled(UInt_t f) const;
   
   static TCanvas   *Canvas(TQtWidget *widget);
   static TQtWidget *Canvas(const TCanvas *canvas);
   static TQtWidget *Canvas(Int_t id);

public slots:
   virtual void cd();
   virtual void cd(int subpadnumber);
   void Disconnect();
   void Refresh();
   virtual bool Save(const QString &fileName) const;
   virtual bool Save(const char    *fileName) const;
   virtual bool Save(const QString &fileName,const char *format,int quality=60) const;
   virtual bool Save(const char    *fileName,const char *format,int quality=60) const;
#ifndef __CINT__
signals:
   // emit the Qt signal when the double buffer of the TCamvas has been filled up
   void CanvasPainted();  // Signal the TCanvas has been painted onto the screen
   void Saved(bool ok);   // Signal the TCanvas has been saved into the file
   void RootEventProcessed(TObject *selected, unsigned int event, TCanvas *c);
#endif

#ifndef Q_MOC_RUN
//MOC_SKIP_BEGIN
   ClassDef(TQtWidget,0) // QWidget to back ROOT TCanvas (Can be used with Qt designer)
//MOC_SKIP_END
#endif
};

//______________________________________________________________________________
inline void TQtWidget::AdjustBufferSize()
   {  if (fPixmapID.size() != size() ) fPixmapID.resize(size()); }

//______________________________________________________________________________
inline bool TQtWidget::paintingActive () const {
  return QWidget::paintingActive() || fPixmapID.paintingActive();
}
//______________________________________________________________________________
inline void TQtWidget::SetRootID(QWidget *wrapper) { fWrapper = wrapper;}
//______________________________________________________________________________
inline QWidget *TQtWidget::GetRootID() const { return fWrapper;}

//______________________________________________________________________________
//
//  Proxy methods to access the TCanvas selected TObject
//  and last processed event
//______________________________________________________________________________
inline Int_t        TQtWidget::GetEvent()       const { return GetCanvas()->GetEvent();       }
//______________________________________________________________________________
inline Int_t        TQtWidget::GetEventX()      const { return GetCanvas()->GetEventX();      }
//______________________________________________________________________________
inline Int_t        TQtWidget::GetEventY()      const { return GetCanvas()->GetEventY();      }
//______________________________________________________________________________
inline TObject     *TQtWidget::GetSelected()    const { return GetCanvas()->GetSelected();    }
//______________________________________________________________________________
inline Int_t        TQtWidget::GetSelectedX()   const { return GetCanvas()->GetSelectedX();   }
//______________________________________________________________________________
inline Int_t        TQtWidget::GetSelectedY()   const { return GetCanvas()->GetSelectedY();   }
//______________________________________________________________________________
inline TVirtualPad *TQtWidget::GetSelectedPad() const { return GetCanvas()->GetSelectedPad(); }

//----- bit manipulation
inline UInt_t TQtWidget::GetAllBits() const       { return fBits;                       }
inline void   TQtWidget::SetAllBits(UInt_t f)     { fBits = f;                          }
inline void   TQtWidget::SetBit(UInt_t f)         { fBits |= f & kBitMask;              }
inline void   TQtWidget::ResetBit(UInt_t f)       { fBits &= ~(f & kBitMask);           }
inline Bool_t TQtWidget::TestBit(UInt_t f) const  { return (Bool_t) ((fBits & f) != 0); }
inline Int_t  TQtWidget::TestBits(UInt_t f) const { return (Int_t) (fBits & f);         }
inline void   TQtWidget::InvertBit(UInt_t f)      { fBits ^= f & kBitMask;              }
   
inline void   TQtWidget::EnableSignalEvents  (UInt_t f){ SetBit  (f); }
inline void   TQtWidget::DisableSignalEvents (UInt_t f){ ResetBit(f); }
inline Bool_t TQtWidget::IsSignalEventEnabled(UInt_t f) const { return TestBit (f); }
inline void   TQtWidget::EmitSignal(UInt_t f)  {if (IsSignalEventEnabled(f)) EmitTestedSignal();};

#endif
