// @(#)root/hist:$Name:  $:$Id: TLimitDataSource,v 1.34 2002/08/16 21:16:00 brun Exp $
// Author: Christophe.Delaere@cern.ch   21/08/2002

///////////////////////////////////////////////////////////////////////////
//
// TLimitDataSource
//
// This class serves as interface to feed data into the TLimit routines
//
///////////////////////////////////////////////////////////////////////////

#include "TLimitDataSource.h"
#include "TH1.h"
#include "TObjString.h"
#include "TRandom3.h"
#include <iostream>

ClassImp(TLimitDataSource)

TLimitDataSource::TLimitDataSource() 
{
   // Default constructor
   fDummyTH1F.SetOwner();
   fDummyIds.SetOwner();
}

TLimitDataSource::TLimitDataSource(TH1F * s, TH1F * b, TH1F * d) 
{
   // Another constructor, directly adds one channel
   // with signal, background and data given as input.
   fDummyTH1F.SetOwner();
   fDummyIds.SetOwner();
   AddChannel(s, b, d);
}

void TLimitDataSource::AddChannel(TH1F * s, TH1F * b, TH1F * d)
{
   // Adds a channel with signal, background and data given as input.
   
   TH1F *empty;
   TRandom3 generator;
   fSignal.AddLast(s);
   fBackground.AddLast(b);
   fCandidates.AddLast(d);
   char rndname[20];
   sprintf(rndname, "rndname%f", generator.Rndm());
   empty = new TH1F(rndname, "", s->GetSize(), 0, 1);
   empty->SetDirectory(0);
   fErrorOnSignal.AddLast(empty);
   fDummyTH1F.AddLast(empty);
   sprintf(rndname, "rndname%f", generator.Rndm());
   empty = new TH1F(rndname, "", s->GetSize(), 0, 1);
   empty->SetDirectory(0);
   fErrorOnBackground.AddLast(empty);
   fDummyTH1F.AddLast(empty);
   TObjArray *dummy = new TObjArray(0);
   fIds.AddLast(dummy);
   fDummyIds.AddLast(dummy);
}

void TLimitDataSource::AddChannel(TH1F * s, TH1F * b, TH1F * d, TH1F * es,
                                  TH1F * eb, TObjArray * names)
{
   // Adds a channel with signal, background and data given as input.
   // In addition, error sources are defined.
   // TH1F are here used for convenience: each bin has to be seen as 
   // an error source (relative).
   // names is an array of strings containing the names of the sources.
   // Sources with the same name are correlated.
   
   fSignal.AddLast(s);
   fBackground.AddLast(b);
   fCandidates.AddLast(d);
   fErrorOnSignal.AddLast(es);
   fErrorOnBackground.AddLast(eb);
   fIds.AddLast(names);
}

void TLimitDataSource::SetOwner(bool swtch)
{
   // Gives to the TLimitDataSource the ownership of the various objects
   // given as input.
   // Objects are then deleted by the TLimitDataSource destructor.
   
   fSignal.SetOwner(swtch);
   fBackground.SetOwner(swtch);
   fCandidates.SetOwner(swtch);
   fErrorOnSignal.SetOwner(swtch);
   fErrorOnBackground.SetOwner(swtch);
   fIds.SetOwner(swtch);
   fDummyTH1F.SetOwner(!swtch);
   fDummyIds.SetOwner(!swtch);
}

