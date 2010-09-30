// @(#)root/base:$Id$
// Author: Rene Brun   05/01/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TDatime                                                              //
//                                                                      //
// This class stores the date and time with a precision of one second   //
// in an unsigned 32 bit word (e.g. 950130 124559). The date is stored  //
// with the origin being the 1st january 1995.                          //
//                                                                      //
// This class has no support for time zones. The time is assumed        //
// to be in the local time of the machine where the object was created. //
// As a result, TDatime objects are not portable between machines       //
// operating in different time zones and unsuitable for storing the     //
// date/time of data taking events and the like. If absolute time is    //
// required, use TTimeStamp.                                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "RConfig.h"

#include <time.h>

#ifdef WIN32
#include "Windows4Root.h"
#include <string.h>
#endif

#include "TBuffer.h"
#include "Strlen.h"
#include "TDatime.h"
#include "TError.h"
#include "Bytes.h"
#include "TString.h"


ClassImp(TDatime)

//______________________________________________________________________________
TDatime::TDatime()
{
   // Create a TDatime and set it to the current time.

   Set();
}

//______________________________________________________________________________
TDatime::TDatime(Int_t date, Int_t time)
{
   // Create a TDatime and set it to the specified date and time.
   // See Set(Int_t, Int_t) about the date, time format.

   Set(date, time);
}

//______________________________________________________________________________
TDatime::TDatime(Int_t year, Int_t month, Int_t day,
                 Int_t hour, Int_t min, Int_t sec)
{
   // Create a TDatime and set it to the specified year, month,
   // day, time, hour, minute and second. See Set() about the format.

   Set(year, month, day, hour, min, sec);
}

//______________________________________________________________________________
TDatime::TDatime(const char *sqlDateTime)
{
   // Expects as input a string in SQL date/time compatible format, like:
   // yyyy-mm-dd hh:mm:ss.

   Set(sqlDateTime);
}

//______________________________________________________________________________
Int_t TDatime::GetDayOfWeek() const
{
   // Returns day of week, with Monday being day 1 and Sunday day 7.

   static TString weekDays[7] = { "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun" };
   TString wd = AsString();
   int  day;
   for (day = 0; day < 7; day++) {
      if (wd(0, 3) == weekDays[day])
         break;
   }
   return (day < 7) ? day+1: -1;
}

//______________________________________________________________________________
const char *TDatime::AsString() const
{
   // Return the date & time as a string (ctime() format).
   // Copy result because it points to a statically allocated string.

   time_t t = Convert();
   char *retStr = ctime(&t);
   if (retStr) {
      *(retStr + 24) = 0;
      return retStr;
   } else {
      static const char *defaulttime = "15/06/96";
      Error("TDatime::AsString", "could not get time string");
      return defaulttime;
   }
}

//______________________________________________________________________________
const char *TDatime::AsString(char *out) const
{
   // Return the date & time as a string (ctime() format).
   // Result is copied into out (and out is returned). Make sure
   // out can at least contain 26 characters. Thread safe.

   time_t t = Convert();
#ifdef _REENTRANT
#if defined(R__SOLARIS) && (_POSIX_C_SOURCE - 0 < 199506L)
   char *retStr = ctime_r(&t, out, 26);
#else
   char *retStr = ctime_r(&t, out);
#endif
#else
   char *retStr = ctime(&t);
#endif
   if (retStr) {
      *(retStr + 24) = 0;
#ifndef _REENTRANT
      strcpy(out, retStr);
#endif
      return retStr;
   } else {
      static const char *defaulttime = "15/06/96";
      strcpy(out, defaulttime);
      Error("TDatime::AsString", "could not get time string");
      return defaulttime;
   }
}

//______________________________________________________________________________
const char *TDatime::AsSQLString() const
{
   // Return the date & time in SQL compatible string format, like:
   // 1997-01-15 20:16:28. The returned string buffer is static and
   // will be reused.

   static char sqldate[20];

   UInt_t year  = fDatime>>26;
   UInt_t month = (fDatime<<6)>>28;
   UInt_t day   = (fDatime<<10)>>27;
   UInt_t hour  = (fDatime<<15)>>27;
   UInt_t min   = (fDatime<<20)>>26;
   UInt_t sec   = (fDatime<<26)>>26;

   snprintf(sqldate,20, "%04d-%02d-%02d %02d:%02d:%02d", (year+1995), month, day,
           hour, min, sec);

   return sqldate;
}

//______________________________________________________________________________
UInt_t TDatime::Convert(Bool_t toGMT) const
{
   // Convert fDatime from TDatime format to the standard time_t format.
   // If toGMT is true, the time offset of the current local time zone is
   // subtracted from the returned time_t. One use of such a non-standard time_t
   // value is to convert a TDatime object that contains local time to GMT,
   // as in this example:
   //
   // TDatime now;
   // now.Set(now.Convert(kTRUE));
   //
   // Caution: the time_t returned from Convert(kTRUE) is incompatible with
   // regular Unix time - it contains an artificial, locale-dependent offset.

   UInt_t year  = fDatime>>26;
   UInt_t month = (fDatime<<6)>>28;
   UInt_t day   = (fDatime<<10)>>27;
   UInt_t hour  = (fDatime<<15)>>27;
   UInt_t min   = (fDatime<<20)>>26;
   UInt_t sec   = (fDatime<<26)>>26;

   struct tm tp;
   tp.tm_year  = year+95;
   tp.tm_mon   = month-1;
   tp.tm_mday  = day;
   tp.tm_hour  = hour;
   tp.tm_min   = min;
   tp.tm_sec   = sec;
   tp.tm_isdst = -1;

   time_t t = mktime(&tp);
   if ((int)t == -1) {
      Error("TDatime::Convert", "error converting fDatime to time_t");
      return 0;
   }
   if (toGMT) {
#ifdef _REENTRANT
      struct tm tg;
      struct tm *tgp = gmtime_r(&t, &tg);
#else
      struct tm *tgp = gmtime(&t);
#endif
      tgp->tm_isdst = -1;
      t  = mktime(tgp);
   }
   return (UInt_t)t;
}

//______________________________________________________________________________
void TDatime::Copy(TDatime &datime) const
{
   // Copy this to datime.

   datime.fDatime = fDatime;
}

//______________________________________________________________________________
void TDatime::FillBuffer(char *&buffer)
{
   // Encode Date/Time into buffer, used by I/O system.

   tobuf(buffer, fDatime);
}

//______________________________________________________________________________
UInt_t TDatime::Get() const
{
   // Return raw date/time as encoded by TDatime. Note, this value cannot
   // be used to e.g. calculate time differences, as it is an encoded value.
   // To calculate time differences use the Convert() method to get a time
   // in seconds and then subtract the values.

   return fDatime;
}

//______________________________________________________________________________
Int_t TDatime::GetDate() const
{
   // Return date in form of 19971224 (i.e. 24/12/1997)

   UInt_t year  = fDatime>>26;
   UInt_t month = (fDatime<<6)>>28;
   UInt_t day   = (fDatime<<10)>>27;
   return 10000*(year+1995) + 100*month + day;
}

//______________________________________________________________________________
Int_t TDatime::GetTime() const
{
   // Return time in form of 123623 (i.e. 12:36:23)

   UInt_t hour  = (fDatime<<15)>>27;
   UInt_t min   = (fDatime<<20)>>26;
   UInt_t sec   = (fDatime<<26)>>26;
   return 10000*hour + 100*min + sec;
}

//______________________________________________________________________________
void TDatime::Print(Option_t *) const
{
   // Print date and time.

   printf("Date/Time = %s\n", AsString());
}

//______________________________________________________________________________
void TDatime::ReadBuffer(char *&buffer)
{
   // Decode Date/Time from output buffer, used by I/O system.

   frombuf(buffer, &fDatime);
}

//______________________________________________________________________________
void TDatime::Set()
{
   // Set Date/Time to current time as reported by the system.
   // Date and Time are encoded into one single unsigned 32 bit word.
   // Date is stored with the origin being the 1st january 1995.
   // Time has 1 second precision.

#ifndef WIN32
   time_t tloc   = time(0);
#ifdef _REENTRANT
   struct tm tpa;
   struct tm *tp = localtime_r(&tloc, &tpa);
#else
   struct tm *tp = localtime(&tloc);
#endif
   UInt_t year   = tp->tm_year;
   UInt_t month  = tp->tm_mon + 1;
   UInt_t day    = tp->tm_mday;
   UInt_t hour   = tp->tm_hour;
   UInt_t min    = tp->tm_min;
   UInt_t sec    = tp->tm_sec;
#else
   SYSTEMTIME tp;
   GetLocalTime(&tp);
   UInt_t year   = tp.wYear-1900;
   UInt_t month  = tp.wMonth;
   UInt_t day    = tp.wDay;
   UInt_t hour   = tp.wHour;
   UInt_t min    = tp.wMinute;
   UInt_t sec    = tp.wSecond;
#endif

   fDatime = (year-95)<<26 | month<<22 | day<<17 | hour<<12 | min<<6 | sec;
}

//______________________________________________________________________________
void TDatime::Set(UInt_t tloc, Bool_t dosDate)
{
   // The input arg is a time_t value returned by time() or a value
   // returned by Convert(). This value is the number of seconds since
   // the EPOCH (i.e. 00:00:00 on Jan 1m 1970). If dosDate is true then
   // the input is a dosDate value.

   UInt_t year, month, day, hour, min, sec;

   if (dosDate) {
      year  = ((tloc >> 25) & 0x7f) + 80;
      month = ((tloc >> 21) & 0xf);
      day   = (tloc >> 16) & 0x1f;
      hour  = (tloc >> 11) & 0x1f;
      min   = (tloc >> 5) & 0x3f;
      sec   = (tloc & 0x1f) * 2;
   } else {
      time_t t = (time_t) tloc;
#ifdef _REENTRANT
      struct tm tpa;
      struct tm *tp = localtime_r(&t, &tpa);
#else
      struct tm *tp = localtime(&t);
#endif
      year   = tp->tm_year;
      month  = tp->tm_mon + 1;
      day    = tp->tm_mday;
      hour   = tp->tm_hour;
      min    = tp->tm_min;
      sec    = tp->tm_sec;
   }

   fDatime = (year-95)<<26 | month<<22 | day<<17 | hour<<12 | min<<6 | sec;
}

//______________________________________________________________________________
void TDatime::Set(Int_t date, Int_t time)
{
   // Set date and time. Data must be in format 980418 or 19980418 and time in
   // 224512 (second precision). The date must
   // be >= 950101.
   // For years >= 2000, date can be given in the form 20001127 or 1001127
   // internally the date will be converted to 1001127

   if (date > 19000000) date -= 19000000;
   if (date < 950101) {
      Error("TDatime::Set", "year smaller than 1995");
      return;
   }

   Int_t year  = date/10000;
   Int_t month = (date-year*10000)/100;
   Int_t day   = date%100;

   Int_t hour, min, sec;

   hour = time/10000;
   min  = (time-hour*10000)/100;
   sec  = time%100;

   fDatime = (year-95)<<26 | month<<22 | day<<17 | hour<<12 | min<<6 | sec;
}

//______________________________________________________________________________
void TDatime::Set(Int_t year, Int_t month, Int_t day,
                  Int_t hour, Int_t min, Int_t sec)
{
   // Set date and time. Year may be xx where 95 <= xx <= 158 (158 being 2058).
   // The year must be >= 1995.

   if (year < 159) year += 1900;
   if (year < 1995) {
      Error("TDatime::Set", "year must be >= 1995");
      return;
   }

   fDatime = (year-1995)<<26 | month<<22 | day<<17 | hour<<12 | min<<6 | sec;
}

//______________________________________________________________________________
void TDatime::Set(const char* sqlDateTime)
{
   // Expects as input a string in SQL date/time compatible format, like:
   // yyyy-mm-dd hh:mm:ss.

   Int_t yy, mm, dd, hh, mi, ss;

   if (sscanf(sqlDateTime, "%d-%d-%d %d:%d:%d", &yy, &mm, &dd, &hh, &mi, &ss) == 6)
      Set(yy, mm, dd, hh, mi, ss);
   else {
      Error("TDatime(sqlDatTime)", "input string not in right format, set"
            " to current date/time");
      Set();
   }
}

//______________________________________________________________________________
void TDatime::Streamer(TBuffer &b)
{
   // Stream a object of type TDatime.

   if (b.IsReading()) {
      b >> fDatime;
   } else {
      b << fDatime;
   }
}

//______________________________________________________________________________
void TDatime::GetDateTime(UInt_t datetime, Int_t &date, Int_t &time)
{
   // Static function that returns the date and time. The input is
   // in TDatime format (as obtained via TDatime::Get()).
   // Date is returned in the format 950223  February 23 1995.
   // Time is returned in the format 102459 10h 24m 59s.

   UInt_t year  = datetime>>26;
   UInt_t month = (datetime<<6)>>28;
   UInt_t day   = (datetime<<10)>>27;
   UInt_t hour  = (datetime<<15)>>27;
   UInt_t min   = (datetime<<20)>>26;
   UInt_t sec   = (datetime<<26)>>26;
   date         =  10000*(year+1995) + 100*month + day;
   time         =  10000*hour + 100*min + sec;
}
