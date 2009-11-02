// @(#)root/treeplayer:$Id$
// Author: Rene Brun 29/10/09

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TTreePerfStats
#define ROOT_TTreePerfStats

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TTreePerfStats                                                       //
//                                                                      //
// TTree I/O performance measurement                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TVirtualPerfStats
#include "TVirtualPerfStats.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif


class TFile;
class TTree;
class TStopwatch;
class TPaveText;
class TGraphErrors;

class TTreePerfStats : public TVirtualPerfStats {

protected:
   Int_t         fTreeCacheSize; //TTreeCache buffer size
   Int_t         fNleaves;       //Number of leaves in the tree
   Int_t         fReadCalls;     //Number of read calls
   Int_t         fReadaheadSize; //Readahead cache size
   Long64_t      fBytesRead;     //Number of bytes read
   Long64_t      fBytesReadExtra;//Number of bytes (overhead) of the readahead cache
   Double_t      fRealTime;      //Real time 
   Double_t      fCpuTime;       //Cpu time
   
   TString       fName;          //name of this TTreePerfStats
   TFile        *fFile;          //!pointer to the file containing the Tree
   TTree        *fTree;          //!pointer to the Tree being monitored
   TGraphErrors *fGraph ;        //pointer to the graph
   TPaveText    *fPave;          //pointer to annotation pavetext
   TStopwatch   *fWatch;         //TStopwatch pointer
   
public:
   TTreePerfStats();
   TTreePerfStats(const char *name, TTree *T);
   virtual ~TTreePerfStats();
   virtual Int_t    DistancetoPrimitive(Int_t px, Int_t py);
   virtual void     Draw(Option_t *option=""); 
   virtual void     ExecuteEvent(Int_t event, Int_t px, Int_t py);
   virtual void     Finish();
   virtual Long64_t GetBytesRead() const {return fBytesRead;}
   virtual Long64_t GetBytesReadExtra() const {return fBytesReadExtra;}
   virtual Double_t GetCpuTime()   const {return fCpuTime;}
   TGraphErrors    *GetGraph()     {return fGraph;}
   const char      *GetName() const{return fName.Data();}
   virtual Int_t    GetNleaves() const {return fNleaves;}
   TPaveText       *GetPave()      {return fPave;}
   virtual Int_t    GetReadaheadSize() const {return fReadaheadSize;}
   virtual Int_t    GetReadCalls() const {return fReadCalls;}
   virtual Double_t GetRealTime()  const {return fRealTime;}
   TStopwatch      *GetStopwatch() const {return fWatch;}
   virtual Int_t    GetTreeCacheSize() const {return fTreeCacheSize;}
   virtual void     Paint(Option_t *chopt="");
   virtual void     Print(Option_t *option="") const;

   virtual void     SimpleEvent(EEventType) {}
   virtual void     PacketEvent(const char *, const char *, const char *,
                            Long64_t , Double_t ,Double_t , Double_t ,Long64_t ) {}
   virtual void     FileEvent(const char *, const char *, const char *, const char *, Bool_t) {}
   virtual void     FileOpenEvent(TFile *, const char *, Double_t) {}
   virtual void     FileReadEvent(TFile *file, Int_t len, Double_t proctime);
   virtual void     RateEvent(Double_t , Double_t , Long64_t , Long64_t) {}

   virtual void     SaveAs(const char *filename="",Option_t *option="") const; 
   virtual void     SavePrimitive(ostream &out, Option_t *option = "");
   virtual void     SetBytesRead(Long64_t nbytes) {fBytesRead = nbytes;}
   virtual void     SetBytesReadExtra(Long64_t nbytes) {fBytesReadExtra = nbytes;}
   virtual void     SetCpuTime(Double_t cptime) {fCpuTime = cptime;}
   virtual void     SetGraph(TGraphErrors *gr) {fGraph = gr;}
   virtual void     SetName(const char *name) {fName = name;}
   virtual void     SetNleaves(Int_t nleaves) {fNleaves = nleaves;}
   virtual void     SetReadaheadSize(Int_t nbytes) {fReadaheadSize = nbytes;}
   virtual void     SetReadCalls(Int_t ncalls) {fReadCalls = ncalls;}
   virtual void     SetRealTime(Double_t rtime) {fRealTime = rtime;}
   virtual void     SetTreeCacheSize(Int_t nbytes) {fTreeCacheSize = nbytes;}
   
   ClassDef(TTreePerfStats,1)  // TTree I/O performance measurement
};

#endif
