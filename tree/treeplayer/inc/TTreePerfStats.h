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


class TBrowser;
class TFile;
class TTree;
class TStopwatch;
class TPaveText;
class TGraphErrors;
class TGaxis;
class TText;
class TTreePerfStats : public TVirtualPerfStats {

protected:
   Int_t         fTreeCacheSize; //TTreeCache buffer size
   Int_t         fNleaves;       //Number of leaves in the tree
   Int_t         fReadCalls;     //Number of read calls
   Int_t         fReadaheadSize; //Readahead cache size
   Long64_t      fBytesRead;     //Number of bytes read
   Long64_t      fBytesReadExtra;//Number of bytes (overhead) of the readahead cache
   Double_t      fRealNorm;      //Real time scale factor for fGraphTime
   Double_t      fRealTime;      //Real time
   Double_t      fCpuTime;       //Cpu time
   Double_t      fDiskTime;      //Time spent in pure raw disk IO
   Double_t      fUnzipTime;     //Time spent uncompressing the data.
   Double_t      fCompress;      //Tree compression factor      
   TString       fName;          //name of this TTreePerfStats
   TString       fHostInfo;      //name of the host system, ROOT version and date
   TFile        *fFile;          //!pointer to the file containing the Tree
   TTree        *fTree;          //!pointer to the Tree being monitored
   TGraphErrors *fGraphIO ;      //pointer to the graph with IO data
   TGraphErrors *fGraphTime ;    //pointer to the graph with timestamp info
   TPaveText    *fPave;          //pointer to annotation pavetext
   TStopwatch   *fWatch;         //TStopwatch pointer
   TGaxis       *fRealTimeAxis;  //pointer to TGaxis object showing real-time
   TText        *fHostInfoText;  //Graphics Text object with the fHostInfo data
      
public:
   TTreePerfStats();
   TTreePerfStats(const char *name, TTree *T);
   virtual ~TTreePerfStats();
   virtual void     Browse(TBrowser *b);
   virtual Int_t    DistancetoPrimitive(Int_t px, Int_t py);
   virtual void     Draw(Option_t *option="");
   virtual void     ExecuteEvent(Int_t event, Int_t px, Int_t py);
   virtual void     Finish();
   virtual Long64_t GetBytesRead() const {return fBytesRead;}
   virtual Long64_t GetBytesReadExtra() const {return fBytesReadExtra;}
   virtual Double_t GetCpuTime()   const {return fCpuTime;}
   virtual Double_t GetDiskTime()  const {return fDiskTime;}
   TGraphErrors    *GetGraphIO()     {return fGraphIO;}
   TGraphErrors    *GetGraphTime()   {return fGraphTime;}
   const char      *GetHostInfo() const{return fHostInfo.Data();}
   const char      *GetName()    const{return fName.Data();}
   virtual Int_t    GetNleaves() const {return fNleaves;}
   virtual Long64_t GetNumEvents() const {return 0;}
   TPaveText       *GetPave()      {return fPave;}
   virtual Int_t    GetReadaheadSize() const {return fReadaheadSize;}
   virtual Int_t    GetReadCalls() const {return fReadCalls;}
   virtual Double_t GetRealTime()  const {return fRealTime;}
   TStopwatch      *GetStopwatch() const {return fWatch;}
   virtual Int_t    GetTreeCacheSize() const {return fTreeCacheSize;}
   virtual Double_t GetUnzipTime() const {return fUnzipTime; }
   virtual void     Paint(Option_t *chopt="");
   virtual void     Print(Option_t *option="") const;

   virtual void     SimpleEvent(EEventType) {}
   virtual void     PacketEvent(const char *, const char *, const char *,
                            Long64_t , Double_t ,Double_t , Double_t ,Long64_t ) {}
   virtual void     FileEvent(const char *, const char *, const char *, const char *, Bool_t) {}
   virtual void     FileOpenEvent(TFile *, const char *, Double_t) {}
   virtual void     FileReadEvent(TFile *file, Int_t len, Double_t start);
   virtual void     UnzipEvent(TObject *tree, Long64_t pos, Double_t start, Int_t complen, Int_t objlen);
   virtual void     RateEvent(Double_t , Double_t , Long64_t , Long64_t) {}

   virtual void     SaveAs(const char *filename="",Option_t *option="") const;
   virtual void     SavePrimitive(ostream &out, Option_t *option = "");
   virtual void     SetBytesRead(Long64_t nbytes) {fBytesRead = nbytes;}
   virtual void     SetBytesReadExtra(Long64_t nbytes) {fBytesReadExtra = nbytes;}
   virtual void     SetCompress(Double_t cx) {fCompress = cx;}
   virtual void     SetDiskTime(Double_t t) {fDiskTime = t;}
   virtual void     SetNumEvents(Long64_t) {}
   virtual void     SetCpuTime(Double_t cptime) {fCpuTime = cptime;}
   virtual void     SetGraphIO(TGraphErrors *gr) {fGraphIO = gr;}
   virtual void     SetGraphTime(TGraphErrors *gr) {fGraphTime = gr;}
   virtual void     SetHostInfo(const char *info) {fHostInfo = info;}
   virtual void     SetName(const char *name) {fName = name;}
   virtual void     SetNleaves(Int_t nleaves) {fNleaves = nleaves;}
   virtual void     SetReadaheadSize(Int_t nbytes) {fReadaheadSize = nbytes;}
   virtual void     SetReadCalls(Int_t ncalls) {fReadCalls = ncalls;}
   virtual void     SetRealNorm(Double_t rnorm) {fRealNorm = rnorm;}
   virtual void     SetRealTime(Double_t rtime) {fRealTime = rtime;}
   virtual void     SetTreeCacheSize(Int_t nbytes) {fTreeCacheSize = nbytes;}
   virtual void     SetUnzipTime(Double_t uztime) {fUnzipTime = uztime;}

   ClassDef(TTreePerfStats,5)  // TTree I/O performance measurement
};

#endif
