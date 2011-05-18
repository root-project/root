/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#define G__QT_DUMMY
#ifdef G__QT_DUMMY

#include <qt.h>

//////////////////////////////////////////////////////////////////////////
// dummy static object for satisfying linker
//////////////////////////////////////////////////////////////////////////
QStringData* QString::shared_null;
QStyle* QApplication::app_style;
QCursor* QApplication::app_cursor;
int QApplication::app_tracking;
QWidget* QApplication::main_widget;
QWidget* QApplication::focus_widget;
QWidget* QApplication::active_window;
QSize QApplication::app_strut;
QApplication* qApp;
QChar QChar::null;            // 0000
QChar QChar::replacement;     // FFFD
QChar QChar::byteOrderMark;     // FEFF
QChar QChar::byteOrderSwapped;     // FFFE
QChar QChar::nbsp;            // 00A0
bool QColor::lazy_alloc;
HPALETTE__* QColor::hpal;
QString QString::null;

const QFont& QFontInfo::font(void) const { return(QFont()); }

QColor dmy;
QColor& Qt::white = dmy;
QColor& Qt::blue = dmy;

const int QTextStream::floatfield=0;
const int QTextStream::adjustfield=1;
const int QTextStream::basefield=2;

QWidgetMapper* QWidget::mapper;
const QSize dmyqsize;
const QSize& QWidgetItem::widgetSizeHint() const { return(dmyqsize); }

// dummy static function for satisfying global variable linkage
bool qt_testCollision(class QCanvasSprite const*,class QCanvasSprite const*) { return true; }
void qInitPngIO(void) { }
bool qt_builtin_gif_reader(void) { return true; }


#endif

