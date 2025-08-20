// @(#)root/base:$Id$
// Author: Fons Rademakers   28/11/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TTime
#define ROOT_TTime


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TTime                                                                //
//                                                                      //
// Basic time type with millisecond precision.                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "Rtypes.h"


class TTime {

private:
   Long64_t   fMilliSec;   // time with millisecond precision

public:
   TTime(): fMilliSec(0) { }
   TTime(Long64_t msec): fMilliSec(msec) { }
   TTime(const TTime &t): fMilliSec(t.fMilliSec) { }
   virtual ~TTime() { }

   TTime& operator=(const TTime &t);

   TTime operator+=(const TTime &t);
   TTime operator-=(const TTime &t);
   TTime operator*=(const TTime &t);
   TTime operator/=(const TTime &t);

   friend TTime operator+(const TTime &t1, const TTime &t2);
   friend TTime operator-(const TTime &t1, const TTime &t2);
   friend TTime operator*(const TTime &t1, const TTime &t2);
   friend TTime operator/(const TTime &t1, const TTime &t2);

   friend Bool_t operator== (const TTime &t1, const TTime &t2);
   friend Bool_t operator!= (const TTime &t1, const TTime &t2);
   friend Bool_t operator<  (const TTime &t1, const TTime &t2);
   friend Bool_t operator<= (const TTime &t1, const TTime &t2);
   friend Bool_t operator>  (const TTime &t1, const TTime &t2);
   friend Bool_t operator>= (const TTime &t1, const TTime &t2);

   operator long() const;
   operator unsigned long() const;
   operator long long() const;
   operator unsigned long long() const;
   const char *AsString() const;

   ClassDef(TTime,2)  //Basic time type with milli second precision
};

inline TTime& TTime::operator= (const TTime &t)
   { fMilliSec = t.fMilliSec; return *this; }
inline TTime TTime::operator+=(const TTime &t)
   { fMilliSec += t.fMilliSec; return *this; }
inline TTime TTime::operator-=(const TTime &t)
   { fMilliSec -= t.fMilliSec; return *this; }
inline TTime TTime::operator*=(const TTime &t)
   { fMilliSec *= t.fMilliSec; return *this; }
inline TTime TTime::operator/=(const TTime &t)
   { fMilliSec /= t.fMilliSec; return *this; }
inline TTime::operator long long() const
   { return fMilliSec; }
inline TTime::operator unsigned long long() const
   { return (ULong64_t) fMilliSec; }

inline TTime operator+(const TTime &t1, const TTime &t2)
   { return TTime(t1.fMilliSec + t2.fMilliSec); }
inline TTime operator-(const TTime &t1, const TTime &t2)
   { return TTime(t1.fMilliSec - t2.fMilliSec); }
inline TTime operator*(const TTime &t1, const TTime &t2)
   { return TTime(t1.fMilliSec * t2.fMilliSec); }
inline TTime operator/(const TTime &t1, const TTime &t2)
   { return TTime(t1.fMilliSec / t2.fMilliSec); }

inline Bool_t operator== (const TTime &t1, const TTime &t2)
   { return t1.fMilliSec == t2.fMilliSec; }
inline Bool_t operator!= (const TTime &t1, const TTime &t2)
   { return t1.fMilliSec != t2.fMilliSec; }
inline Bool_t operator< (const TTime &t1, const TTime &t2)
   { return t1.fMilliSec < t2.fMilliSec; }
inline Bool_t operator<= (const TTime &t1, const TTime &t2)
   { return t1.fMilliSec <= t2.fMilliSec; }
inline Bool_t operator> (const TTime &t1, const TTime &t2)
   { return t1.fMilliSec > t2.fMilliSec; }
inline Bool_t operator>= (const TTime &t1, const TTime &t2)
   { return t1.fMilliSec >= t2.fMilliSec; }

#endif
