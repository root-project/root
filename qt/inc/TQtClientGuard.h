#ifndef ROOT_TQtClientGuard
#define ROOT_TQtClientGuard
 
/****************************************************************************
** $Id: TQtClientGuard.h,v 1.9 2004/05/12 18:27:58 fine Exp $
**
** Copyright (C) 2004 by Valeri Fine. Brookhaven National Laboratory.
**                                    All rights reserved.
**
** This file may be distributed under the terms of the Q Public License
** as defined by Trolltech AS of Norway and appearing in the file
** LICENSE.QPL included in the packaging of this file.
ls**
*****************************************************************************/

#include <qobject.h>
#include <qpixmap.h>
#include <qptrlist.h> 
#include "TQtClientWidget.h"


class TQtClientGuard : public QObject {
   Q_OBJECT
private:
   TQtClientGuard& operator=(const TQtClientGuard&); // AXEL: intentionally not implementedprotected:
   mutable QPtrList<QWidget> fQClientGuard;
   int  fDeadCounter;
   friend class TQtClientWidget;
public:
   TQtClientGuard(): QObject(), fDeadCounter(0){};
   virtual ~TQtClientGuard(){;}
   TQtClientWidget *Create(QWidget* parent=0, const char* name=0, WFlags f=0 );
   void    Delete(QWidget *w);
   QWidget *Find(Window_t id);
   void    Add(QWidget *w);

protected:
   void    Disconnect(QWidget *w);
   void    DisconnectChildren(TQtClientWidget *w);
protected slots:
   void    Disconnect();
};

class TQtPixmapGuard : public QObject {
   Q_OBJECT
private:
   TQtPixmapGuard& operator=(const TQtPixmapGuard&); // AXEL: intentionally not implementedprotected:
   mutable QPtrList<QPixmap> fQClientGuard;
   int  fDeadCounter;

public:
   TQtPixmapGuard(): QObject(),fDeadCounter(0){};
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
   void    Disconnect(QPixmap *w);
protected slots:
   void    Disconnect();
};

#endif

