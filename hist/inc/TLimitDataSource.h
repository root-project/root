// @(#)root/hist:$Name:  $:$Id: TLimitDataSource.h,v 1.34 2002/08/16 21:16:00 brun Exp $
// Author: Christophe.Delaere@cern.ch   21/08/2002

#ifndef ROOT_TLimitDataSource
#define ROOT_TLimitDataSource

#ifndef ROOT_TObjArray
#include "TObjArray.h"
#endif

class TH1F;

//_______________________________________________________________________
//
// TLimitDataSource
//
// This class serves as input for the TLimit::ComputeLimit method.
// It takes the signal, background and data histograms to form a channel. 
// More channels can be added using AddChannel(), as well as different
// systematics sources. 
//_______________________________________________________________________


class TLimitDataSource {
public:
	TLimitDataSource();
	virtual ~TLimitDataSource() {}
	TLimitDataSource(TH1F* s,TH1F* b,TH1F* d);
	virtual void AddChannel(TH1F*,TH1F*,TH1F*);
	virtual void AddChannel(TH1F*,TH1F*,TH1F*,TH1F*, TH1F*, TObjArray*);
	inline virtual TObjArray* GetSignal() { return &fSignal;}
	inline virtual TObjArray* GetBackground() { return &fBackground;}
	inline virtual TObjArray* GetCandidates() { return &fCandidates;}
	inline virtual TObjArray* GetErrorOnSignal() { return &fErrorOnSignal;}
	inline virtual TObjArray* GetErrorOnBackground() { return &fErrorOnBackground;}
	inline virtual TObjArray* GetErrorNames() { return &fIds;}
	virtual void SetOwner(bool swtch=kTRUE);
private:
	// The arrays used to store the packed inputs
	TObjArray fSignal;            //packed input signal
	TObjArray fBackground;        //packed input background
	TObjArray fCandidates;        //packed input candidates (data)
	TObjArray fErrorOnSignal;     //packed error sources for signal
	TObjArray fErrorOnBackground; //packed error sources for background
	TObjArray fIds;               //packed IDs for the different error sources
	// some dummy objects that the class will use and delete
	TObjArray fDummyTH1F;         //array of dummy object (used for bookeeping)
	TObjArray fDummyIds;          //array of dummy object (used for bookeeping)

  ClassDef(TLimitDataSource, 1 ) // input for TLimit routines
};

#endif
