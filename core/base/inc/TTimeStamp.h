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

#include "Rtypes.h"

#include <ctime>

#if defined (_MSC_VER) && (_MSC_VER < 1900)
struct timespec {
   time_t   tv_sec;  // seconds
   long     tv_nsec; // nanoseconds
};
#endif

// For backward compatibility
typedef struct timespec timespec_t;
typedef struct tm       tm_t;

class TVirtualMutex;
class TTimeStamp;

std::ostream &operator<<(std::ostream &os,  const TTimeStamp &ts);
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

   // compatibility with time() and DOS date
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

   // compatibility with time() and DOS date
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

   // return stored time values converted to sidereal time
   Double_t     AsGMST(Double_t UT1Offset = 0 /*milliseconds*/) const; //rval in hours
   Double_t     AsGAST(Double_t UT1Offset = 0 /*milliseconds*/) const; //rval in hours
   Double_t     AsLMST(Double_t Longitude /*degrees*/, Double_t UT1Offset = 0 /*milliseconds*/) const; //rval in hours
   Double_t     AsLAST(Double_t Longitude /*degrees*/, Double_t UT1Offset = 0 /*milliseconds*/) const; //rval in hours

   const char  *AsString(const Option_t *option="") const;

   void         Copy(TTimeStamp &ts) const;
   UInt_t       GetDate(Bool_t inUTC = kTRUE, Int_t secOffset = 0,
                        UInt_t *year = nullptr, UInt_t *month = nullptr,
                        UInt_t *day = nullptr) const;
   UInt_t       GetTime(Bool_t inUTC = kTRUE, Int_t secOffset = 0,
                        UInt_t *hour = nullptr, UInt_t *min = nullptr,
                        UInt_t *sec = nullptr) const;
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
