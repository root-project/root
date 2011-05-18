// @(#)root/base:$Id$
// Author: R. Hatcher   30/9/2001

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TTimeStamp
#define ROOT_TTimeStamp

//////////////////////////////////////////////////////////////////////////
//
// The TTimeStamp encapsulates seconds and ns since EPOCH
//
// This extends (and isolates) struct timespec
//    struct timespec
//       {
//          time_t   tv_sec;   /* seconds */
//          long     tv_nsec;  /* nanoseconds */
//       }
//    time_t seconds is relative to Jan 1, 1970 00:00:00 UTC
//
// No accounting of leap seconds is made.
//
// Due to ROOT/CINT limitations TTimeStamp does not explicitly
// hold a timespec struct; attempting to do so means the Streamer
// must be hand written.  Instead we have chosen to simply contain
// similar fields within the private area of this class.
//
// NOTE: the use of time_t (and its default implementation as a 32 int)
//       implies overflow conditions occurs somewhere around
//       Jan 18, 19:14:07, 2038.
//       If this experiment is still going when it becomes significant
//       someone will have to deal with it.
//
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif
#ifndef ROOT_Riosfwd
#include "Riosfwd.h"
#endif

#include <time.h>
#if !defined(__CINT__) && (defined(R__MACOSX) || defined(R__OBSD))
#include <sys/time.h>
#endif
#if defined(__CINT__) || defined(R__WIN32)
// Explicit definition of timespec 'cause "rootcint" won't look in
// appropriate <time.h>. time_t appears to be defined as "typedef long time_t;"
// in CINT version of <time.h>.  This isn't required by the standard:
// to be compatible w/ std functions it must be at least 32-bits long,
// but it might be longer to avoid the year-2037 cutoff.
struct timespec
{
   time_t   tv_sec;             // seconds
   long     tv_nsec;            // nanoseconds
};
#endif
#if defined(__CINT__)
struct tm
{
   int tm_sec;                   // Seconds.     [0-60] (1 leap second)
   int tm_min;                   // Minutes.     [0-59]
   int tm_hour;                  // Hours.       [0-23]
   int tm_mday;                  // Day.         [1-31]
   int tm_mon;                   // Month.       [0-11]
   int tm_year;                  // Year - 1900.
   int tm_wday;                  // Day of week. [0-6]
   int tm_yday;                  // Days in year.[0-365]
   int tm_isdst;                 // DST.         [-1/0/1]
};
#endif

// define this typedef so that CINT doesn't barf at using these
// for function return values
typedef struct timespec timespec_t;
typedef struct tm       tm_t;

class TVirtualMutex;
class TTimeStamp;
ostream &operator<<(ostream &os,  const TTimeStamp &ts);
TBuffer &operator<<(TBuffer &buf, const TTimeStamp &ts);
TBuffer &operator>>(TBuffer &buf, TTimeStamp &ts);
Bool_t operator==(const TTimeStamp &lhs, const TTimeStamp &rhs);
Bool_t operator!=(const TTimeStamp &lhs, const TTimeStamp &rhs);
Bool_t operator< (const TTimeStamp &lhs, const TTimeStamp &rhs);
Bool_t operator<=(const TTimeStamp &lhs, const TTimeStamp &rhs);
Bool_t operator> (const TTimeStamp &lhs, const TTimeStamp &rhs);
Bool_t operator>=(const TTimeStamp &lhs, const TTimeStamp &rhs);

R__EXTERN TVirtualMutex *gTimeMutex;

class TTimeStamp {

friend Bool_t operator==(const TTimeStamp &lhs, const TTimeStamp &rhs);
friend Bool_t operator!=(const TTimeStamp &lhs, const TTimeStamp &rhs);
friend Bool_t operator< (const TTimeStamp &lhs, const TTimeStamp &rhs);
friend Bool_t operator<=(const TTimeStamp &lhs, const TTimeStamp &rhs);
friend Bool_t operator> (const TTimeStamp &lhs, const TTimeStamp &rhs);
friend Bool_t operator>=(const TTimeStamp &lhs, const TTimeStamp &rhs);

private:
   Int_t  fSec;      // seconds
   Int_t  fNanoSec;  // nanoseconds

   void   NormalizeNanoSec();

public:
   // empty ctor (builds current time with nsec field incremented from static)
   TTimeStamp();

   // construction from timespec struct
   TTimeStamp(const timespec_t &ts) :
      fSec(Int_t(ts.tv_sec)), fNanoSec(ts.tv_nsec) { NormalizeNanoSec(); }

   // construction from time_t and separate nsec
   TTimeStamp(time_t t, Int_t nsec) :
      fSec(Int_t(t)), fNanoSec(nsec) { NormalizeNanoSec(); }

   // construction from bits and pieces
   TTimeStamp(UInt_t year, UInt_t month,
              UInt_t day,  UInt_t hour,
              UInt_t min,  UInt_t sec,
              UInt_t nsec = 0, Bool_t isUTC = kTRUE, Int_t secOffset = 0);

   // compatibility with TDatime
   TTimeStamp(UInt_t date, UInt_t time, UInt_t nsec,
              Bool_t isUTC = kTRUE, Int_t secOffset = 0);

   // compatability with time() and DOS date
   TTimeStamp(UInt_t tloc, Bool_t isUTC = kTRUE, Int_t secOffset = 0,
              Bool_t dosDate = kFALSE);

   virtual ~TTimeStamp() { }

   // initialize to current time with nsec field incremented from static
   void Set();

   // construction from bits and pieces
   void Set(Int_t year, Int_t month,  Int_t day,
            Int_t hour, Int_t min,    Int_t sec,
            Int_t nsec, Bool_t isUTC, Int_t secOffset);

   // compatibility with TDatime
   void Set(Int_t date,   Int_t time, Int_t nsec,
            Bool_t isUTC, Int_t secOffset);

   // compatability with time() and DOS date
   void Set(UInt_t tloc, Bool_t isUTC, Int_t secOffset, Bool_t dosDate);

   // direct setters
   void SetSec(Int_t sec) { fSec = sec; }
   void SetNanoSec(Int_t nsec) { fNanoSec = nsec; }

   timespec_t   GetTimeSpec() const
      { timespec_t value = {fSec,fNanoSec}; return value; }
   time_t       GetSec() const { return fSec; }
   Int_t        GetNanoSec() const { return fNanoSec; }

   Double_t     AsDouble() const { return fSec + 1e-9 * fNanoSec; }
   Double_t     AsJulianDate() const { return (AsDouble()/86400.0 + 2440587.5); }
   const char  *AsString(const Option_t *option="") const;

   void         Copy(TTimeStamp &ts) const;
   UInt_t       GetDate(Bool_t inUTC = kTRUE, Int_t secOffset = 0,
                        UInt_t *year = 0, UInt_t *month = 0,
                        UInt_t *day = 0) const;
   UInt_t       GetTime(Bool_t inUTC = kTRUE, Int_t secOffset = 0,
                        UInt_t *hour = 0, UInt_t *min = 0,
                        UInt_t *sec = 0) const;
   Int_t        GetDayOfYear(Bool_t inUTC = kTRUE, Int_t secOffset = 0) const;
   Int_t        GetDayOfWeek(Bool_t inUTC = kTRUE, Int_t secOffset = 0) const;
   Int_t        GetMonth(Bool_t inUTC = kTRUE, Int_t secOffset = 0) const;
   Int_t        GetWeek(Bool_t inUTC = kTRUE, Int_t secOffset = 0) const;
   Bool_t       IsLeapYear(Bool_t inUTC = kTRUE, Int_t secOffset = 0) const;

   void         Add(const TTimeStamp &offset);

   void         Print(const Option_t *option="") const;

   operator double() const { return AsDouble(); }

   // Utility functions
   static Int_t   GetZoneOffset();
   static time_t  MktimeFromUTC(tm_t *tmstruct);
   static void    DumpTMStruct(const tm_t &tmstruct);
   static Int_t   GetDayOfYear(Int_t day, Int_t month, Int_t year);
   static Int_t   GetDayOfWeek(Int_t day, Int_t month, Int_t year);
   static Int_t   GetWeek(Int_t day, Int_t month, Int_t year);
   static Bool_t  IsLeapYear(Int_t year);

   ClassDef(TTimeStamp,1)  //Encapsulates seconds and ns since EPOCH
};


inline Bool_t operator==(const TTimeStamp &lhs, const TTimeStamp &rhs)
   { return lhs.fSec == rhs.fSec &&
            lhs.fNanoSec == rhs.fNanoSec; }

inline Bool_t operator!=(const TTimeStamp &lhs, const TTimeStamp &rhs)
   { return lhs.fSec != rhs.fSec ||
            lhs.fNanoSec != rhs.fNanoSec; }

inline Bool_t operator<(const TTimeStamp &lhs, const TTimeStamp &rhs)
   { return lhs.fSec < rhs.fSec ||
             (lhs.fSec == rhs.fSec &&
              lhs.fNanoSec < rhs.fNanoSec); }

inline Bool_t operator<=(const TTimeStamp &lhs, const TTimeStamp &rhs)
   { return lhs.fSec < rhs.fSec ||
             (lhs.fSec == rhs.fSec &&
              lhs.fNanoSec <= rhs.fNanoSec); }

inline Bool_t operator>(const TTimeStamp &lhs, const TTimeStamp &rhs)
   { return lhs.fSec > rhs.fSec ||
             (lhs.fSec == rhs.fSec &&
              lhs.fNanoSec > rhs.fNanoSec); }

inline Bool_t operator>=(const TTimeStamp &lhs, const TTimeStamp &rhs)
   { return lhs.fSec > rhs.fSec ||
             (lhs.fSec == rhs.fSec &&
              lhs.fNanoSec >= rhs.fNanoSec); }

#endif
