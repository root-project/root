// @(#)root/proofplayer:$Name:  $:$Id: TPacketizerProgressive.h,v 1.4 2007/02/12 13:05:31 rdm Exp $
// Author: Zev Benjamin  13/09/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TPacketizerProgressive                                               //
//                                                                      //
// This class generates packets to be processed on PROOF slave servers. //
// A packet is an event range (begin entry and number of entries) or    //
// object range (first object and number of objects) in a TTree         //
// (entries) or a directory (objects) in a file.                        //
// Packets are generated taking into account the performance of the     //
// remote machine, the time it took to process a previous packet on     //
// the remote machine, the locality of the database files, etc.         //
// This packetizer does not pre-open the files to calculate the total   //
// number of events, it just walks sequentially through the list of     //
// files.                                                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TPacketizerProgressive
#define ROOT_TPacketizerProgressive

#ifndef ROOT_TVirtualPacketizer
#include "TVirtualPacketizer.h"
#endif

class TDSet;
class TDSetElement;
class THashTable;
class TList;
class TMap;
class TMessage;
class TSlave;
class TTimer;

class TPacketizerProgressive : public TVirtualPacketizer {

public:
   class TFileStat;

   class TFileNode : public TObject {
   private:
      TString        fNodeName;        // FQDN of the node
      TList         *fFiles;           // TDSetElements (files) stored on this node
      TObject       *fUnAllocFileNext; // cursor in fFiles
      TList         *fActFiles;        // files with work remaining
      TObject       *fActFileNext;     // cursor in fActFiles
      Int_t          fMySlaveCnt;      // number of slaves running on this node
      Int_t          fSlaveCnt;        // number of external slaves processing files on this node
   public:
      TFileNode(const char *name);
      ~TFileNode();

      void        IncMySlaveCnt() { fMySlaveCnt++; }
      void        IncSlaveCnt(const char *slave) { if (fNodeName != slave) fSlaveCnt++; }
      void        DecSlaveCnt(const char *slave);
      Int_t       GetSlaveCnt() const { return fMySlaveCnt + fSlaveCnt; }
      Int_t       GetNumberOfActiveFiles() const;
      Bool_t      IsSortable() const { return kTRUE; }
      const char *GetName() const { return fNodeName; }
      void        Add(TDSetElement *elem);
      TFileStat  *GetNextUnAlloc();
      TFileStat  *GetNextActive();
      void        RemoveActive(TFileStat *file);
      Bool_t      HasActiveFiles();
      Bool_t      HasUnAllocFiles() {if (fUnAllocFileNext) return kTRUE; return kFALSE; }
      Int_t       Compare(const TObject *other) const;
      void        Print(Option_t *opt ="") const;
      void        Reset();
   };

   class TFileStat : public TObject {
   private:
      Bool_t        fIsDone;       // is this element processed
      TFileNode    *fNode;         // my FileNode
      TDSetElement *fElement;      // location of the file and its range
      Long64_t      fNextEntry;    // cursor in the range, -1 when done
   public:
      TFileStat(TFileNode *node, TDSetElement *elem);

      Bool_t        IsDone() const { return fIsDone; }
      void          SetDone() { fIsDone = kTRUE; }
      TFileNode    *GetNode() const { return fNode; }
      TDSetElement *GetElement() const { return fElement; }
      Long64_t      GetNextEntry() const { return fNextEntry; }
      void          MoveNextEntry(Long64_t step) { fNextEntry += step; }
   };

   class TSlaveStat : public TObject {
   private:
      TSlave       *fSlave;        // corresponding TSlave record
      TFileNode    *fFileNode;     // corresponding node or 0
      TFileStat    *fCurFile;      // file currently being processed
      TDSetElement *fCurElem;      // TDSetElement currently being processed
      Long64_t      fProcessed;    // number of entries processed
   public:
      TSlaveStat(TSlave *slave);

      TFileNode      *GetFileNode() const { return fFileNode; }
      TFileStat      *GetCurrentFile() const { return fCurFile; }
      TDSetElement   *GetCurrentElement() const { return fCurElem; }
      const char     *GetName() const;
      Long64_t        GetEntriesProcessed() const { return fProcessed; }
      void            SetFileNode(TFileNode *node) { fFileNode = node; }
      void            SetCurrentFile(TFileStat *file) { fCurFile = file; }
      void            SetCurrentElement(TDSetElement* elem) { fCurElem = elem; }
      void            IncEntriesProcessed(Long64_t n) { fProcessed += n; }
   };

private:
   enum {
      kSlaveHostConnLim    = 2,
      kNonSlaveHostConnLim = 2,
      kEntryListSize       = 5
   };

   TDSet      *fDset;
   TList      *fSlaves;
   TList      *fSlavesRemaining;  // slaves stilll working
   Long64_t    fFirstEvent;
   Long64_t    fTotalEvents;

   Long64_t    fEntriesSeen;      // number of entries found so far
   Long64_t    fFilesOpened;      // total number of files with their entries recorded
   Long64_t    fEstTotalEntries;  // estimated total number of entries
   Long64_t    fEntriesProcessed; // total number of entries processed
   TMap       *fSlaveStats;       // map of slave addresses to its TSlaveStat object
   THashTable *fNewFileSlaves;    // slaves that have just opened a new file and need to
                                  // record the number of entries in them (keyed by TSlaveStat)

   TList      *fUnAllocSlaves;    // slave hosts that have unallocated files
   TList      *fUnAllocNonSlaves; // non-slave hosts that have unallocated files
   TList      *fActiveSlaves;     // slave hosts that have active files
   TList      *fActiveNonSlaves;  // non-slave hosts that have active files

   TList      *fLastEntrySizes;   // list of the last kEntryListSize TDSetElement sizes (in entries)
   Long64_t    fPacketSize;       // current packet size based on estimate of total number of entries

   TTimer     *fProgress;         // progress bar timer

   TPacketizerProgressive();
   TPacketizerProgressive(const TPacketizerProgressive&);

   void          RecalculatePacketSize(Long64_t newCount);
   TFileStat    *GetNextActive(TSlaveStat *stat);
   TFileStat    *GetNextUnAlloc(TSlaveStat *stat);
   TDSetElement *BuildPacket(TSlaveStat *stat, Long64_t num);

   void Init();

   virtual Bool_t HandleTimer(TTimer *timer);

public:
   TPacketizerProgressive(TDSet *dset, TList *slaves,
                          Long64_t first, Long64_t num,
                          TList *input);
   virtual ~TPacketizerProgressive();

   Long64_t      GetEntriesProcessed() const { return fEntriesProcessed; }
   Long64_t      GetEntriesProcessed(TSlave *s) const;
   TDSetElement *GetNextPacket(TSlave *s, TMessage *r);

   ClassDef(TPacketizerProgressive, 0);  // Packetizer that does not pre-open any files
};

#endif
