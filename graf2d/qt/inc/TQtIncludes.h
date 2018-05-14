// @(#)root/qt:$Id$
// Author: Raphael Isemann  14/05/2018

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * Copyright (C) 2003 by Valeri Fine.                                    *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TQtIncludes
#define ROOT_TQtIncludes

// Single header that provides Qt headers to all classes.
// This prevents that the Qt headers are duplicated in multiple modules.

#ifndef __CINT__
#  include <qapplication.h>
#  include <qbrush.h>
#  include <qcolor.h>
#  include <qglobal.h>
#  include <qmutex.h>
#  include <qnamespace.h>
#  include <qobject.h>
#  include <qpixmap.h>
#  include <QApplication>
#  include <QByteArray>
#  include <QColor>
#  include <QCursor>
#  include <QEvent>
#  include <QFont>
#  include <QFrame>
#  include <QKeySequence>
#  include <QList>
#  include <QMap>
#  include <QMouseEvent>
#  include <QPainter>
#  include <QPolygon>
#  include <QQueue>
#  include <QRect>
#  include <QtCore/QEvent>
#  include <QtCore/QObject>
#  include <QtCore/QPoint>
#  include <QtCore/QPointer>
#  include <QtCore/QSize>
#  include <QtCore/QVector>
#  include <QTextCodec>
#  include <QtGui/QFocusEvent>
#  include <QtGui/QFontDatabase>
#  include <QtGui/QKeyEvent>
#  include <QtGui/QMouseEvent>
#  include <QtGui/QPaintDevice>
#  include <QtGui/QPaintEvent>
#  include <QtGui/QPen>
#  include <QtGui/QPixmap>
#  include <QtGui/QResizeEvent>
#  include <QtGui/QShowEvent>
#  include <QtGui/QWidget>
#  include <QTimer>

#if QT_VERSION < 0x40000
#  include <qevent.h>
#endif

#endif

#endif
