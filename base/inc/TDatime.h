// @(#)root/base:$Name:  $:$Id: TDatime.h,v 1.4 2001/12/10 14:46:06 rdm Exp $
// Author: Rene Brun   05/01/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TDatime
#define ROOT_TDatime


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TDatime                                                              //
//                                                                      //
// Data and time 950130 124559.                                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_Htypes
#include "Htypes.h"
#endif


class TDatime {

protected:
   UInt_t     fDatime;            //Date (relative to 1995) + time

public:
   TDatime();
   TDatime(const TDatime &d) { fDatime = d.fDatime; }
   TDatime(UInt_t time) { fDatime = time; }
   TDatime(Int_t date, Int_t time);
   TDatime(Int_t year, Int_t month, Int_t day,
           Int_t hour, Int_t min, Int_t sec);
   TDatime(const char *sqlDateTime);

   TDatime operator=(const TDatime &d);

   const char  *AsString() const;
   const char  *AsSQLString() const;
   UInt_t       Convert(Bool_t toGMT = kFALSE) const;
   void         Copy(TDatime &datime);
   UInt_t       Get() const { return fDatime; }
   Int_t        GetDate() const;
   Int_t        GetTime() const;
   Int_t        GetYear() const { return (fDatime>>26) + 1995; }
   Int_t        GetMonth() const { return (fDatime<<6)>>28; }
   Int_t        GetDay() const { return (fDatime<<10)>>27; }
   Int_t        GetHour() const { return (fDatime<<15)>>27; }
   Int_t        GetMinute() const { return (fDatime<<20)>>26; }
   Int_t        GetSecond() const { return (fDatime<<26)>>26; }
   void         FillBuffer(char *&buffer);
   void         Print(Option_t *option="") const;
   void         ReadBuffer(char *&buffer);
   void         Set();
   void         Set(UInt_t tloc);
   void         Set(Int_t date, Int_t time);
   void         Set(Int_t year, Int_t month, Int_t day,
                    Int_t hour, Int_t min, Int_t sec);
   Int_t        Sizeof() const {return sizeof(UInt_t);}

   friend Bool_t operator==(const TDatime &d1, const TDatime &d2);
   friend Bool_t operator!=(const TDatime &d1, const TDatime &d2);
   friend Bool_t operator< (const TDatime &d1, const TDatime &d2);
   friend Bool_t operator<=(const TDatime &d1, const TDatime &d2);
   friend Bool_t operator> (const TDatime &d1, const TDatime &d2);
   friend Bool_t operator>=(const TDatime &d1, const TDatime &d2);

   static void GetDateTime(UInt_t datetime, Int_t &date, Int_t &time);

   ClassDef(TDatime,1)  //Date and time 950130 124559
};


inline TDatime TDatime::operator=(const TDatime &d)
   { fDatime = d.fDatime; return *this; }

inline Bool_t operator==(const TDatime &d1, const TDatime &d2)
   { return d1.fDatime == d2.fDatime; }
inline Bool_t operator!=(const TDatime &d1, const TDatime &d2)
   { return d1.fDatime != d2.fDatime; }
inline Bool_t operator< (const TDatime &d1, const TDatime &d2)
   { return d1.fDatime < d2.fDatime; }
inline Bool_t operator<=(const TDatime &d1, const TDatime &d2)
   { return d1.fDatime <= d2.fDatime; }
inline Bool_t operator> (const TDatime &d1, const TDatime &d2)
   { return d1.fDatime > d2.fDatime; }
inline Bool_t operator>=(const TDatime &d1, const TDatime &d2)
   { return d1.fDatime >= d2.fDatime; }

#endif
