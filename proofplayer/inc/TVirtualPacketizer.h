// @(#)root/proofplayer:$Name:  $:$Id: TVirtualPacketizer.h,v 1.8 2007/05/29 16:06:55 ganis Exp $
// Author: Maarten Ballintijn    9/7/2002

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TVirtualPacketizer
#define ROOT_TVirtualPacketizer

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TVirtualPacketizer                                                   //
//                                                                      //
// XXX update Comment XXX                                               //
// Packetizer generates packets to be processed on PROOF worker servers.//
// A packet is an event range (begin entry and number of entries) or    //
// object range (first object and number of objects) in a TTree         //
// (entries) or a directory (objects) in a file.                        //
// Packets are generated taking into account the performance of the     //
// remote machine, the time it took to process a previous packet on     //
// the remote machine, the locality of the database files, etc.         //
//                                                                      //
// TVirtualPacketizer includes common parts of PROOF packetizers.       //
// Look in subclasses for details.                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif


class TDSet;
class TDSetElement;
class TSlave;
class TMessage;
class TNtupleD;



class TVirtualPacketizer : public TObject {

friend class TPacketizer;
friend class TAdaptivePacketizer;
friend class TPacketizerProgressive;

private:
   Long64_t  fProcessed;    // number of entries processed
   Long64_t  fBytesRead;    // number of bytes processed
   TTimer   *fProgress;     // progress updates timer

   Long64_t  fTotalEntries; // total number of entries to be distributed;
                            // not used in the progressive packetizer

   // Members for progress info
   Long_t    fStartTime;    // time offset
   Float_t   fInitTime;     // time before processing
   Float_t   fProcTime;     // time since start of processing
   Float_t   fTimeUpdt;     // time between updates
   TNtupleD *fCircProg;     // Keeps circular info for "instantenous"
                            // rate calculations
   Long_t     fCircN;       // Circularity

   TVirtualPacketizer(const TVirtualPacketizer &);  // no implementation, will generate
   void operator=(const TVirtualPacketizer &);      // error on accidental usage

   virtual Bool_t HandleTimer(TTimer *timer);

   void           SplitEventList(TDSet *dset);
   TDSetElement  *CreateNewPacket(TDSetElement* base, Long64_t first, Long64_t num);

protected:
   Bool_t   fValid;           // Constructed properly?
   Bool_t   fStop;            // Termination of Process() requested?

   TVirtualPacketizer();
   Long64_t GetEntries(Bool_t tree, TDSetElement *e); // Num of entries or objects

public:
   enum EStatusBits { kIsInitializing = BIT(16) };
   virtual ~TVirtualPacketizer() { }

   Bool_t                  IsValid() const { return fValid; }
   Long64_t                GetEntriesProcessed() const { return fProcessed; }
   virtual Long64_t        GetEntriesProcessed(TSlave *sl) const;
   virtual TDSetElement   *GetNextPacket(TSlave *sl, TMessage *r);
   virtual void            SetInitTime();
   virtual void            StopProcess(Bool_t abort);

   Long64_t      GetBytesRead() const { return fBytesRead; }
   Float_t       GetInitTime() const { return fInitTime; }
   Float_t       GetProcTime() const { return fProcTime; }

   ClassDef(TVirtualPacketizer,0)  //Generate work packets for parallel processing
};

#endif
