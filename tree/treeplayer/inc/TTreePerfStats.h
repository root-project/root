// @(#)root/base:$Id$
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


#ifndef ROOT_VirtualPerfStats
#include "TVirtualPerfStats.h"
#endif


class TFile;
class TTree;
class TStopwatch;
class TGraphErrors;

class TTreePerfStats : public TVirtualPerfStats {

protected:
   TFile        *fFile;   //pointer to the file containing the Tree
   TTree        *fTree;   //pointer to the Tree being monitored
   TGraphErrors *fGraph ; //pointer to the graph
   TStopwatch   *fWatch;  //TStopwatch pointer
      
public:
   TTreePerfStats(TTree *T);
   virtual ~TTreePerfStats() {}
   virtual void Draw(Option_t *option="");
   TGraphErrors *GetGraph()     {return fGraph;}
   TStopwatch   *GetStopwatch() {return fWatch;}
   virtual void Print(Option_t *option="") const;
   virtual void SimpleEvent(EEventType) {}

   virtual void PacketEvent(const char *, const char *, const char *,
                            Long64_t , Double_t ,
                            Double_t , Double_t ,
                            Long64_t ) {}

   virtual void FileEvent(const char *, const char *, const char *,
                          const char *, Bool_t) {}

   virtual void FileOpenEvent(TFile *, const char *, Double_t) {}

   virtual void FileReadEvent(TFile *file, Int_t len, Double_t proctime);

   virtual void RateEvent(Double_t , Double_t ,
                          Long64_t , Long64_t) {}

   virtual void SetBytesRead(Long64_t ) {}
   virtual Long64_t GetBytesRead() const {return 0;}

   ClassDef(TTreePerfStats,0)  // TTree I/O performance measurement
};

#endif
