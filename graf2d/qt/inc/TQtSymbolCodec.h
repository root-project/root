/****************************************************************************
** $Id: TQtSymbolCodec.h,v 1.4 2008/04/14 02:14:18 fine Exp $
**
** Implementation of QTextCodec class
**
** Created : 20050125
**
**********************************************************************/

#ifndef ROOT_QSYMBOLCODEC_H
#define ROOT_QSYMBOLCODEC_H


#include <QByteArray>
#include <QTextCodec>
#include <QByteArray>

#ifndef QT_NO_CODEC_SYMBOL

class QSymbolCodec : public QTextCodec {
public:
    virtual int mibEnum() const;
    QByteArray  name() const;
    const char* mimeName() const;

#if !defined(Q_NO_USING_KEYWORD)
    using QTextCodec::fromUnicode;
#endif
    QByteArray fromUnicode(const QString& uc, int& lenInOut) const;
    virtual QByteArray convertFromUnicode( const QChar * input, int number, ConverterState *state ) const;
    virtual QString    convertToUnicode(const char *chars, int len, ConverterState *state) const;

    QString toUnicode(const char* chars, int len) const;
    int heuristicContentMatch(const char* chars, int len) const;
};

#endif /* QT_NO_CODEC_SYMBOL */
#endif
