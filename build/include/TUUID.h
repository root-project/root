// @(#)root/base:$Id$
// Author: Fons Rademakers   30/9/2001

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TUUID
#define ROOT_TUUID

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TUUID                                                                //
//                                                                      //
// This class defines a UUID (Universally Unique IDentifier), also      //
// known as GUIDs (Globally Unique IDentifier). A UUID is 128 bits      //
// long, and if generated according to this algorithm, is either        //
// guaranteed to be different from all other UUIDs/GUIDs generated      //
// until 3400 A.D. or extremely likely to be different. UUIDs were      //
// originally used in the Network Computing System (NCS) and            //
// later in the Open Software Foundation's (OSF) Distributed Computing  //
// Environment (DCE).                                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifdef WIN32
#undef GetCurrentTime
#endif
#include "Rtypes.h"

// forward declaration
class TBuffer;
class TFile;
class TDirectory;
class TInetAddress;
class TDatime;

class TUUID {

protected:
   UInt_t    fUUIDIndex;             //!index in the list of UUIDs in TProcessUUID
   UInt_t    fTimeLow;               // 60 bit time, lower 32 bits
   UShort_t  fTimeMid;               // middle 16 time bits
   UShort_t  fTimeHiAndVersion;      // high 12 time bits + 4 UUID version bits
   UChar_t   fClockSeqHiAndReserved; // high 6 clock bits + 2 bits reserved
   UChar_t   fClockSeqLow;           // low 8 clock bits
   UChar_t   fNode[6];               // 6 node id bytes

   struct uuid_time_t {
      UInt_t high;
      UInt_t low;
   };

   Int_t CmpTime(uuid_time_t *t1, uuid_time_t *t2);
   void  Format(UShort_t clockseq, uuid_time_t ts);
   void  GetNodeIdentifier();
   void  GetCurrentTime(uuid_time_t *timestamp);
   void  GetSystemTime(uuid_time_t *timestamp);
   void  GetRandomInfo(UChar_t seed[16]);
   void  SetFromString(const char *uuid_str);

public:
   TUUID();
   TUUID(const char *uuid_str);
   virtual ~TUUID();

   const char  *AsString() const;
   Int_t        Compare(const TUUID &u) const;
   UShort_t     Hash() const;
   void         Print() const;
   TInetAddress GetHostAddress() const;
   TDatime      GetTime() const;
   void         GetUUID(UChar_t uuid[16]) const;
   void         SetUUID(const char *uuid_str);
   UInt_t       GetUUIDNumber() const { return fUUIDIndex; }
   void         SetUUIDNumber(UInt_t index) { fUUIDIndex = index; }

   void         StreamerV1(TBuffer &b);
   void         FillBuffer(char *&buffer);
   void         ReadBuffer(char *&buffer);
   Int_t        Sizeof() const { return 18; }

   ClassDef(TUUID,1)  // Universally Unique IDentifier
};


inline TBuffer &operator>>(TBuffer &buf, TUUID &uuid)
{ uuid.Streamer(buf); return buf; }

// Not inlined in order to avoid const casted away warning in user code.
TBuffer &operator<<(TBuffer &buf, const TUUID &uuid);

inline Bool_t operator==(const TUUID &u1, const TUUID &u2)
{ return (!u1.Compare(u2)) ? kTRUE : kFALSE; }

inline Bool_t operator!=(const TUUID &u1, const TUUID &u2)
{ return !(u1 == u2); }


#endif
