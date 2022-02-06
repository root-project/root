// @(#)root/hist:$Id$
// Author: Christophe.Delaere@cern.ch   21/08/2002

#ifndef ROOT_TLimitDataSource
#define ROOT_TLimitDataSource

#include "TObjArray.h"

#include "TVectorDfwd.h"

class TH1;

//_______________________________________________________________________
//
// TLimitDataSource
//
// This class serves as input for the TLimit::ComputeLimit method.
// It takes the signal, background and data histograms to form a channel.
// More channels can be added using AddChannel(), as well as different
// systematics sources.
//_______________________________________________________________________


class TLimitDataSource : public TObject{
public:
   TLimitDataSource();
   ~TLimitDataSource() override {}
   TLimitDataSource(TH1* s,TH1* b,TH1* d);
   TLimitDataSource(TH1* s,TH1* b,TH1* d, TVectorD* es,TVectorD* eb,TObjArray* names);
   virtual void AddChannel(TH1*,TH1*,TH1*);
   virtual void AddChannel(TH1*,TH1*,TH1*,TVectorD*, TVectorD*, TObjArray*);
   inline virtual TObjArray* GetSignal() { return &fSignal;}
   inline virtual TObjArray* GetBackground() { return &fBackground;}
   inline virtual TObjArray* GetCandidates() { return &fCandidates;}
   inline virtual TObjArray* GetErrorOnSignal() { return &fErrorOnSignal;}
   inline virtual TObjArray* GetErrorOnBackground() { return &fErrorOnBackground;}
   inline virtual TObjArray* GetErrorNames() { return &fIds;}
   virtual void SetOwner(bool swtch=kTRUE);
private:
   /// @name Arrays used to store the packed inputs
   ///@{
   TObjArray fSignal;            ///< Packed input signal
   TObjArray fBackground;        ///< Packed input background
   TObjArray fCandidates;        ///< Packed input candidates (data)
   TObjArray fErrorOnSignal;     ///< Packed error sources for signal
   TObjArray fErrorOnBackground; ///< Packed error sources for background
   TObjArray fIds;               ///< Packed IDs for the different error sources
   ///@}

   /// Dummy objects the class will use and delete
   ///@{
   TObjArray fDummyTA;           ///< Array of dummy object (used for bookeeping)
   TObjArray fDummyIds;          ///< Array of dummy object (used for bookeeping)
   ///@}

   ClassDefOverride(TLimitDataSource, 2 ) // Input for TLimit routines
};

#endif
