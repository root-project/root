/****************************************************************************
** $Id: TQtSymbolCodec.cxx,v 1.6 2009/07/01 04:27:52 fine Exp $
**
** Implementation of QTextCodec class
**
** Created : 20050125
**
**
**********************************************************************/

#include "TQtSymbolCodec.h"
#include <QByteArray>

#ifndef QT_NO_CODEC_SYMBOL

// NOT REVISED

#if 0 // Not used!
static const uchar unkn = '?'; // BLACK SQUARE (94) would be better
#endif

static const ushort greek_symbol_to_unicode[64] = {
//
//  upper case letters:
//  ----------------
//   Alpha   Beta     Chi   Delta   Epsilon   Phi    Gamma    Eta
    0x0391, 0x0392, 0x03A7, 0x0394, 0x0395, 0x03A6, 0x0393, 0x0397,
//   Iota   Theta(S) Kappa  Lambda    Mu      Nu    Omicron   Pi
    0x0399, 0x03D1, 0x039A, 0x039B, 0x039C, 0x039D, 0x039F, 0x03A0,
//  Theta    Rho    Sigma    Tau    Upsilon Stigma   Omega     Xi
    0x0398, 0x03A1, 0x03A3, 0x03A4, 0x03A5, 0x03DB, 0x03A9, 0x039E,
//   Psi     Zeta   Sigma
    0x03A8, 0x0396, 0x03EA, 0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD,
//
//  lower case letters:
//  ----------------
//   Alpha   Beta    Chi    Delta   Epsilon   Phi    Gamma    Eta
    0x03B1, 0x03B2, 0x03C7, 0x03B4, 0x03B5, 0x03D5, 0x03B3, 0x03B7,
//   Iota   Phi (2) Kappa   Lambda    Mu      Nu    Omicron   Pi
    0x03B9, 0x03C6, 0x03BA, 0x03BB, 0x03BC, 0x03BD, 0x03BF, 0x03C0,
//  Theta    Rho    Sigma    Tau    Upsilon OmegaPi Omega     Xi
    0x03B8, 0x03C1, 0x03C3, 0x03C4, 0x03C5, 0x03D6, 0x03B9, 0x03BE,
//   Psi     Zeta   Sigma
    0x03C8, 0x03B6, 0x03EA, 0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD
};

static const ushort symbol_to_unicode[96] = {
//
//  upper case letters:
//  ----------------
// UPSILON  NUMBER  LessEq   Div   INFINITY ???????  CLUB   DIAMOND     
    0x03D2, 0x0374, 0x2264, 0x2215, 0x221E, 0x0395, 0x2663, 0x2666,
//   HEART  SPADE   <--->   <---             -->             RING
    0x2665, 0x2660, 0x2194, 0x2190, 0x2191, 0x2192, 0x2193, 0x2218,
//  Theta    Rho    Great    ????   Propor  diff    bullet   ????
    0x0398, 0x03A1, 0x2265, 0x03A4, 0x221D, 0x2202, 0x2219, 0x03A8,
//   NonEq  Ident    ????   ellips   div     minus  ?????   aleph
    0x2260, 0x2261, 0xFFFD, 0x22EF, 0x2223, 0x2212, 0xFFFD, 0xFFFD,
//
//  lower case letters:
//  ----------------
//   Alpha   Beta    Chi    Delta   Epsilon   Phi    Gamma    Eta
    0x03B1, 0x03B2, 0x03C7, 0x03B4, 0x03B5, 0x03D5, 0x03B3, 0x03B7,
//   Iota   Phi (2) Kappa   Lambda    Mu      Nu    Omicron   Pi
    0x03B9, 0x03C6, 0x03BA, 0x03BB, 0x03BC, 0x03BD, 0x03BF, 0x03C0,
//  Theta    Rho    Sigma    Tau    Upsilon OmegaPi Omega     Xi
    0x03B8, 0x03C1, 0x03C3, 0x03C4, 0x03C5, 0x03D6, 0x03B9, 0x03BE,
//   Psi     Zeta   arrow    Tau    Upsilon  ????   Omega     Xi
    0x03C8, 0x03B6, 0x21D4, 0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD,
//  lower case letters:
//  ----------------
//   Alpha   Beta    Chi    Delta   Sigma     Phi    Gamma    Eta
    0x03B1, 0x03B2, 0x03C7, 0x03B4, 0x2211, 0x03D5, 0x03B3, 0x03B7,
//   Iota   Phi (2) Kappa   Lambda    Mu      Nu    Omicron   Pi
    0x03B9, 0x03C6, 0x03BA, 0x03BB, 0x03BC, 0x03BD, 0x03BF, 0x03C0,
//  Theta   Integr  Sigma    Tau    Upsilon OmegaPi Omega     Xi
    0x03B8, 0x222b, 0x03C3, 0x03C4, 0x03C5, 0x03D6, 0x03B9, 0x03BE,
//   Psi     Zeta   Sigma    Tau    Upsilon  ????   Omega     Xi
    0x03C8, 0x03B6, 0x03EA, 0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD
};


#if 0  // Not used!
static const uchar unicode_to_symbol_00[32] = {
    0xA0, unkn, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7,
    0xA8, 0xA9, 0xD7, 0xAB, 0xAC, 0xAD, 0xAE, unkn,
    0xB0, 0xB1, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7,
    0xB8, 0xB9, 0xF7, 0xBB, 0xBC, 0xBD, 0xBE, unkn,
};

static const uchar unicode_to_symbol_05[32] = {
    0xE0, 0xE1, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7,
    0xE8, 0xE9, 0xEA, 0xEB, 0xEC, 0xED, 0xEE, 0xEF,
    0xF0, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7,
    0xF8, 0xF9, 0xFA, unkn, unkn, unkn, unkn, unkn
};
#endif

/*!
    \class QSymbolCodec qsymbolcodec.h
    \reentrant
    \ingroup i18n

    \brief The QSymbolCodec class provides conversion to and from
    visually ordered Microsoft Symbol.ttf.


    This codec has the name "symbol". 
*/

//_______________________________________________________________________
/*! \reimp */
int QSymbolCodec::mibEnum() const
{
    return 2001;
}

//_______________________________________________________________________
/*! \reimp */
QByteArray QSymbolCodec::name() const
{
    return mimeName(); // "symbol";
}

//_______________________________________________________________________
/*! \reimp */ 
const char* QSymbolCodec::mimeName() const
{
   //   Returns the codec's mime name.
  return "symbol";
}

//_______________________________________________________________________
/*! \reimp */ 
QString QSymbolCodec::toUnicode(const char* chars, int len ) const
{
   QString r;
   const unsigned char * c = (const unsigned char *)chars;

   if( len == 0 ) return QString::null;

   // Test, if the user gives us a directionality.
   // We use 0xFE and 0xFF in ISO8859-8 for that.
   // These chars are undefined in the charset, and are mapped to
   // RTL overwrite

   for( int i=0; i<len; i++ ) {
      if ( 64 < c[i] && c[i] <= 64+32 )
         r[i] = greek_symbol_to_unicode[c[i]-65];
      else if (64+32+1 <= c[i] && c[i] < 64+32+32+1)
         r[i] = greek_symbol_to_unicode[c[i]-65-32]+32;
      else if (161 <= c[i] )
         r[i] = symbol_to_unicode[c[i]-161];
      else
         r[i]= c[i];
   }
   return r;
}
//_______________________________________________________________________
QByteArray QSymbolCodec::fromUnicode(const QString& /*uc*/, int& /*lenInOut*/) const
{
   // process only len chars... - not implemented yet.
   qWarning( "Method <QSymbolCodec::fromUnicode> has not been implemented yet");
   /*
   //int l;
   //if( lenInOut > 0 )
   //   l = qMin((int)uc.length(),lenInOut);
   //else
   //   l = (int)uc.length();
   */
   QByteArray rstr;

   return rstr;
}

//_______________________________________________________________________
/*! \reimp */
QByteArray QSymbolCodec::convertFromUnicode( const QChar *input, int number, ConverterState *) const
{  return  fromUnicode(input, number) ;                         }

 //_______________________________________________________________________
/*! \reimp */
QString    QSymbolCodec::convertToUnicode(const char *chars, int len, ConverterState *) const
{  return toUnicode(chars,len);                                                    }

//_______________________________________________________________________
/*! \reimp */
int QSymbolCodec::heuristicContentMatch(const char* chars, int len) const
{
   const unsigned char * c = (const unsigned char *)chars;
   int score = 0;
   for (int i=0; i<len; i++) {
      if( c[i] > 64 && c[i] < 255 ) 
         //     if ( symbol_to_unicode[c[i] - 0x80] != 0xFFFD)
         score++;
      else
         return -1;
   }
   return score;
}

#endif
