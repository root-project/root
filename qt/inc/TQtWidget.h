// @(#)root/qt:$Name:  $:$Id: TQtWidget.h,v 1.3 2004/07/28 00:12:40 rdm Exp $
// Author: Valeri Fine   21/01/2002

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * Copyright (C) 2002 by Valeri Fine.                                    *
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

#include <qwidget.h>
#include <qpixmap.h>

class TCanvas;

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
 Q_OBJECT
public:
   enum {
      kEXITSIZEMOVE,
      kENTERSIZEMOVE,
      kFORCESIZE
   };
  TQtWidget( QWidget* parent=0, const char* name=0, WFlags f=Qt::WStyle_NoBorder, bool embedded=TRUE);
  virtual ~TQtWidget();
  void SetCanvas(TCanvas *c){ fCanvas = c;}
  inline TCanvas  *GetCanvas() const   { return fCanvas;}
  inline QPixmap  &GetBuffer()         { return fPixmapID;}

  // overloaded methods
  virtual void adjustSize();
  virtual void resize (int w, int h);
  virtual void erase ();
  bool    IsDoubleBuffered() { return fDoubleBufferOn; }
  void    SetDoubleBuffer(bool on=TRUE){ fDoubleBufferOn = on;}

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
   virtual QSize       sizeHint () const;        //  returns the preferred size of the widget.
   virtual QSize       minimumSizeHint () const; // returns the smallest size the widget can have.
   virtual QSizePolicy sizePolicy () const;      //  returns a QSizePolicy; a value describing the space requirements of the
   // -- A special event handler
   virtual void exitSizeEvent ();
   virtual void stretchWidget(QResizeEvent *e);

public slots:
   virtual TVirtualPad *cd();
   virtual TVirtualPad *cd(int subpadnumber);
   void Disconnect();
   void Refresh();
signals:
   // emit the Qt signal when the double buffer of the TCamvas has been filled up
   void CanvasPainted();

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

#endif
