// @(#)root/qt:$Id$
// Author: Valeri Fine   21/01/2002

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
#  include <QtGui/QWidget>
#  include <QtGui/QMouseEvent>
#  include <QtGui/QShowEvent>
#  include <QtGui/QFocusEvent>
#  include <QtGui/QKeyEvent>
#  include <QtGui/QResizeEvent>
#  include <QtCore/QEvent>
#  include <QtGui/QPaintEvent>
#  include <QtGui/QPaintDevice>
#  include <QtCore/QSize>
#  include <QtCore/QPoint>
#  include <QtCore/QPointer>
#  include <QtGui/QPixmap>
#  include "TQtCanvasPainter.h"
#else
  // List of the fake classes to make RootCint happy.
  class QWidget;
  class QPixmap;
  class QMouseEvent;
  class QFocusEvent;
  class QCustomEvent;
  class QKeyEvent;
  class QShowEvent;
  class QPaintEvent;
  class QPaintDevice;
  class QResizeEvent;
  class QSize;
  class QString;
  class QEvent;
  class QSizePolicy;
  class QContextMenuEvent;
  class QSize;
  class QPoint;
  class TQtCanvasPainter;
#endif
  class QTimer;
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
class TQtWidgetBuffer
{
private:
   const QWidget *fWidget;
   QPaintDevice  *fBuffer;
   bool  fIsImage;
public:
   TQtWidgetBuffer(const QWidget *w, bool clear=false);
   TQtWidgetBuffer(const TQtWidgetBuffer &b);
   const QPaintDevice  *Buffer() const  { return fBuffer; }
   QPaintDevice  *Buffer()  { return fBuffer; }
   ~TQtWidgetBuffer();
   void Clear();
   bool PaintingActive(){ return fBuffer ? fBuffer->paintingActive() : false; }
   QRect Rect () const { return fWidget->rect();                }
   int Height () const { return fBuffer ? fBuffer->height() : 0;}
   int Width  () const { return fBuffer ? fBuffer->width() : 0; }
};

//___________________________________________________________________
class  TQtWidget : public QWidget {
#ifndef __CINT__
 Q_OBJECT
 friend class TQtSynchPainting;
#endif
private:

   TQtWidget(const TQtWidget&);
   void operator=(const TQtWidget&);
   //----- Private bits, clients can only test but not change them
   UInt_t         fBits;       //bit field status word
   enum {
      kBitMask       = 0x00ffffff
   };
   bool fNeedStretch;
#ifndef __CINT__
   QPointer<TQtCanvasPainter> fCanvasDecorator;  //< The object to paint onto the TQtWidget on the top of TCanvas image
#endif
protected:
   void Init();
   void ResetCanvas() { fCanvas = 0;}

public:
   enum {
      kEXITSIZEMOVE,
      kENTERSIZEMOVE,
      kFORCESIZE
   };
#ifndef __CINT__
  TQtWidget( QWidget* parent, const char* name, Qt::WFlags f=0, bool embedded=TRUE);
  TQtWidget( QWidget* parent=0, Qt::WFlags f=0, bool embedded=TRUE);
#else
  TQtWidget( QWidget* parent=0);
#endif
  virtual ~TQtWidget();
  void SetCanvas(TCanvas *c);
//  inline TCanvas  *GetCanvas() const         { return fCanvas;}
  TCanvas  *GetCanvas() const;
  TQtWidgetBuffer  &SetBuffer();
  const TQtWidgetBuffer  *GetBuffer()  const;
  QPixmap  *GetOffScreenBuffer()  const;

  // overloaded methods
  virtual void Erase ();
  bool    IsDoubleBuffered() const { return fDoubleBufferOn; }
  void    SetDoubleBuffer(bool on=TRUE);
  virtual void SetSaveFormat(const char *format);

protected:
   friend class TGQt;
   friend class TQtFeedBackWidget;
   TCanvas           *fCanvas;
   TQtWidgetBuffer   *fPixmapID;     // Double buffer of this widget
   TQtWidgetBuffer   *fPixmapScreen; // Double buffer for no-double buffer operation
   bool        fPaint;
   bool        fSizeChanged;
   bool        fDoubleBufferOn;
   bool        fEmbedded;
   QSize       fSizeHint;
   QWidget    *fWrapper;
   QString     fSaveFormat;
   bool        fInsidePaintEvent;
   QPoint      fOldMousePos;
   int         fIgnoreLeaveEnter;
   QTimer     *fRefreshTimer;


   void SetRootID(QWidget *wrapper);
   QWidget *GetRootID() const;
   virtual void EmitCanvasPainted() { emit CanvasPainted(); }
   TCanvas  *Canvas();
   bool paintFlag(bool mode=TRUE);
   void AdjustBufferSize();

   bool PaintingActive () const;
   void SetIgnoreLeaveEnter(int ignoreLE = 1);


   virtual void enterEvent       ( QEvent *      );
   virtual void customEvent      ( QEvent *      );
   virtual void contextMenuEvent ( QContextMenuEvent *);
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
   TQtCanvasPainter *CanvasDecorator();
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
   void SetCanvasDecorator( TQtCanvasPainter *decorator);

public:
   // Static method to immitate ROOT as needed
   static TApplication *InitRint(Bool_t prompt=kFALSE, const char *appClassName="QtRint", int *argc=0, char **argv=0,
          void *options = 0, int numOptions = 0, Bool_t noLogo = kTRUE);
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
protected slots:
   void RefreshCB();

signals:
   // emit the Qt signal when the double buffer of the TCamvas has been filled up
   void CanvasPainted();  // Signal the TCanvas has been painted onto the screen
   void Saved(bool ok);   // Signal the TCanvas has been saved into the file
   void RootEventProcessed(TObject *selected, unsigned int event, TCanvas *c);
#endif

#ifndef Q_MOC_RUN
   ClassDef(TQtWidget,0) // QWidget to back ROOT TCanvas (Can be used with Qt designer)
#endif
};

//______________________________________________________________________________
inline TCanvas  *TQtWidget::GetCanvas() const         { return fCanvas; }

//______________________________________________________________________________
inline const TQtWidgetBuffer  *TQtWidget::GetBuffer()  const {
   //  return the current widget buffer;
   return IsDoubleBuffered() ? fPixmapScreen : fPixmapID;
}

//______________________________________________________________________________
inline bool TQtWidget::PaintingActive () const {
  return QWidget::paintingActive() || (fPixmapID && fPixmapID->PaintingActive())
     || (fPixmapScreen && fPixmapScreen->PaintingActive());
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

inline TQtCanvasPainter *TQtWidget::CanvasDecorator() { return fCanvasDecorator;   }
inline void   TQtWidget::SetCanvasDecorator( TQtCanvasPainter *decorator) { fCanvasDecorator = decorator;}

inline void   TQtWidget::EnableSignalEvents  (UInt_t f){ SetBit  (f); }
inline void   TQtWidget::DisableSignalEvents (UInt_t f){ ResetBit(f); }
inline Bool_t TQtWidget::IsSignalEventEnabled(UInt_t f) const { return TestBit (f); }
inline void   TQtWidget::EmitSignal(UInt_t f)  {if (IsSignalEventEnabled(f)) EmitTestedSignal();}
inline void   TQtWidget::SetIgnoreLeaveEnter(int ignoreLE) { fIgnoreLeaveEnter = ignoreLE; }

#endif
