// Author: Valeri Fine   21/01/2002
/****************************************************************************
** $Id: TQtWidget.h,v 1.17 2004/07/02 00:22:49 fine Exp $
**
** Copyright (C) 2002 by Valeri Fine.  All rights reserved.
**
** This file may be distributed under the terms of the Q Public License
** as defined by Trolltech AS of Norway and appearing in the file
** LICENSE.QPL included in the packaging of this file.
*****************************************************************************/

#ifndef ROOT_QtWidget
#define ROOT_QtWidget

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

   TCanvas  *Canvas();
   bool paintFlag(bool mode=TRUE);
   void AdjustBufferSize();

   // overloaded methods
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
   virtual void cd();
   virtual void cd(int subpadnumber);
   void Disconnect();

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
