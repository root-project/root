// @(#)root/qt:$Name:  $:$Id: TQtSymbolCodec.h,v 1.1 2005/02/08 07:36:08 brun Exp $
// Author: Valeri Fine   08/02/2005
/****************************************************************************
** $Id: TQtSymbolCodec.h,v 1.1 2005/02/08 07:36:08 brun Exp $
**
** Implementation of QTextCodec class
**
** Created : 20050125
**
**********************************************************************/

#ifndef ROOT_QSYMBOLCODEC_H
#define ROOT_QSYMBOLCODEC_H

#ifndef QT_H
#include "qtextcodec.h"
#endif

#ifndef QT_NO_CODEC_SYMBOL

class QSymbolCodec : public QTextCodec {
public:
    virtual int mibEnum() const;
    const char* name() const;
    const char* mimeName() const;

#if !defined(Q_NO_USING_KEYWORD)
    using QTextCodec::fromUnicode;
#endif
    QCString fromUnicode(const QString& uc, int& lenInOut) const;
    QString toUnicode(const char* chars, int len) const;

    int heuristicContentMatch(const char* chars, int len) const;
};

#endif

#endif
