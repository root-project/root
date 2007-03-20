/****************************************************************************
** $Id: TQtSymbolCodec.h,v 1.3 2006/12/29 00:32:55 fine Exp $
**
** Implementation of QTextCodec class
**
** Created : 20050125
**
**********************************************************************/

#ifndef ROOT_QSYMBOLCODEC_H
#define ROOT_QSYMBOLCODEC_H

#include "qglobal.h"

#if QT_VERSION < 0x40000
# ifndef QT_H
#   include "qtextcodec.h"
#  endif // QT_H
#else
//Added by qt3to4:
#  include <Q3CString>

#  include <QTextCodec>
#  include <QByteArray>
#endif /* QT_VERSION */

#ifndef QT_NO_CODEC_SYMBOL

class QSymbolCodec : public QTextCodec {
public:
    virtual int mibEnum() const;
#if QT_VERSION < 0x40000
    const char* name() const;
#else
    QByteArray  name() const;
#endif /* QT_VERSION */
    const char* mimeName() const;

#if !defined(Q_NO_USING_KEYWORD)
    using QTextCodec::fromUnicode;
#endif
#if QT_VERSION < 0x40000
    QCString fromUnicode(const QString& uc, int& lenInOut) const;
    QCString fromUnicode( const QString & uc ) const { return QTextCodec::fromUnicode(uc); }
#else
    Q3CString fromUnicode(const QString& uc, int& lenInOut) const;
    virtual QByteArray convertFromUnicode( const QChar * input, int number, ConverterState *state ) const;
    virtual QString    convertToUnicode(const char *chars, int len, ConverterState *state) const;
#endif /* QT_VERSION */
    
    QString toUnicode(const char* chars, int len) const;
    

    int heuristicContentMatch(const char* chars, int len) const;
};

#endif /* QT_NO_CODEC_SYMBOL */
#endif
