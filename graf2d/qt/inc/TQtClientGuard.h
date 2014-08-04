#ifndef ROOT_TQtClientGuard
#define ROOT_TQtClientGuard

// @(#)root/qt:$Id$
// Author: Valeri Fine   21/01/2002

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * Copyright (C) 2002 by Valeri Fine.                                    *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <qobject.h>
#include <qpixmap.h>
#if QT_VERSION < 0x40000
#  include <qptrlist.h>
#else /* QT_VERSION */
#  include <QList>
#endif /* QT_VERSION */
#include "TQtClientWidget.h"


class TQtClientGuard : public QObject {
   Q_OBJECT
private:
   TQtClientGuard& operator=(const TQtClientGuard&); // AXEL: intentionally not implementedprotected:
#if QT_VERSION < 0x40000
   mutable QPtrList<QWidget> fQClientGuard;
#else /* QT_VERSION */
   mutable QList<QWidget *> fQClientGuard;
#endif /* QT_VERSION */
   int  fDeadCounter;
   friend class TQtClientWidget;
public:
   TQtClientGuard(): QObject(), fDeadCounter(0){};
   virtual ~TQtClientGuard(){;}
   TQtClientWidget *Create(QWidget* parent=0, const char* name=0, Qt::WFlags f=0 );
   void    Delete(QWidget *w);
   QWidget *Find(Window_t id);
   void    Add(QWidget *w);

protected:
   void    Disconnect(QWidget *w, int found=-1);
   void    DisconnectChildren(TQtClientWidget *w);
protected slots:
   void    Disconnect();
};

class TQtPixmapGuard : public QObject {
   Q_OBJECT
private:
   TQtPixmapGuard& operator=(const TQtPixmapGuard&); // AXEL: intentionally not implementedprotected:
#if QT_VERSION < 0x40000
   mutable QPtrList<QPixmap> fQClientGuard;
#else /* QT_VERSION */
   mutable QList<QPixmap *> fQClientGuard;
#endif /* QT_VERSION */
   int  fDeadCounter;
   int  fLastFound;

public:
   TQtPixmapGuard(): QObject(),fDeadCounter(0),fLastFound(-1){};
   virtual ~TQtPixmapGuard(){;}
   QPixmap* Create(int w, int h, int depth = -1);
      //Optimization optimization=DefaultOptim);
   QPixmap* Create (const QString &fileName, const char *format = 0);
   QPixmap* Create(int w, int h, const uchar *bits, bool isXbitmap=TRUE);
   QPixmap* Create(const QPixmap &src);
   QBitmap* Create(const QBitmap &src);
   //, ColorMode mode = Auto);
   QPixmap* Create ( const char* xpm[]);
   void    Delete(QPixmap *w);
   QPixmap *Pixmap(Pixmap_t id,bool needBitmap=kFALSE);
   QPixmap *Find(Window_t id);
   void    Add(QPixmap *w);

protected:
   void    Disconnect(QPixmap *w, int found=-1);
   void    SetCurrent(int found) { fLastFound = found;}
protected slots:
   void    Disconnect();
};

#endif

