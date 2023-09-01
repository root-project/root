// @(#)root/hist:$Id$
// Author: Christophe.Delaere@cern.ch   21/08/2002


/** \class TLimitDataSource
 This class serves as input for the TLimit::ComputeLimit method.
 It takes the signal, background and data histograms to form a channel.
 More channels can be added using AddChannel(), as well as different
 systematics sources.
*/


#include "TLimitDataSource.h"
#include "TH1.h"
#include "TVectorD.h"
#include "TRandom3.h"
#include "snprintf.h"

ClassImp(TLimitDataSource);

////////////////////////////////////////////////////////////////////////////////
/// Default constructor

TLimitDataSource::TLimitDataSource()
{
   fDummyTA.SetOwner();
   fDummyIds.SetOwner();
}

////////////////////////////////////////////////////////////////////////////////
/// Another constructor, directly adds one channel
/// with signal, background and data given as input.

TLimitDataSource::TLimitDataSource(TH1 * s, TH1 * b, TH1 * d)
{
   fDummyTA.SetOwner();
   fDummyIds.SetOwner();
   AddChannel(s, b, d);
}

////////////////////////////////////////////////////////////////////////////////
/// Another constructor, directly adds one channel
/// with signal, background and data given as input.

TLimitDataSource::TLimitDataSource(TH1 * s, TH1 * b, TH1 * d,
                                   TVectorD * es, TVectorD * eb, TObjArray * names)
{
   fDummyTA.SetOwner();
   fDummyIds.SetOwner();
   AddChannel(s, b, d, es, eb, names);
}

////////////////////////////////////////////////////////////////////////////////
/// Adds a channel with signal, background and data given as input.

void TLimitDataSource::AddChannel(TH1 * s, TH1 * b, TH1 * d)
{
   TVectorD *empty;
   TRandom3 generator;
   fSignal.AddLast(s);
   fBackground.AddLast(b);
   fCandidates.AddLast(d);
   char rndname[20];
   snprintf(rndname,20, "rndname%f", generator.Rndm());
   empty = new TVectorD(1);
   fErrorOnSignal.AddLast(empty);
   fDummyTA.AddLast(empty);
   snprintf(rndname,20, "rndname%f", generator.Rndm());
   empty = new TVectorD(1);
   fErrorOnBackground.AddLast(empty);
   fDummyTA.AddLast(empty);
   TObjArray *dummy = new TObjArray(0);
   fIds.AddLast(dummy);
   fDummyIds.AddLast(dummy);
}

////////////////////////////////////////////////////////////////////////////////
/// Adds a channel with signal, background and data given as input.
/// In addition, error sources are defined.
/// TH1 are here used for convenience: each bin has to be seen as
/// an error source (relative).
/// names is an array of strings containing the names of the sources.
/// Sources with the same name are correlated.

void TLimitDataSource::AddChannel(TH1 * s, TH1 * b, TH1 * d, TVectorD * es,
                                  TVectorD * eb, TObjArray * names)
{
   fSignal.AddLast(s);
   fBackground.AddLast(b);
   fCandidates.AddLast(d);
   fErrorOnSignal.AddLast(es);
   fErrorOnBackground.AddLast(eb);
   fIds.AddLast(names);
}

////////////////////////////////////////////////////////////////////////////////
/// Gives to the TLimitDataSource the ownership of the various objects
/// given as input.
/// Objects are then deleted by the TLimitDataSource destructor.

void TLimitDataSource::SetOwner(bool swtch)
{
   fSignal.SetOwner(swtch);
   fBackground.SetOwner(swtch);
   fCandidates.SetOwner(swtch);
   fErrorOnSignal.SetOwner(swtch);
   fErrorOnBackground.SetOwner(swtch);
   fIds.SetOwner(swtch);
   fDummyTA.SetOwner(!swtch);
   fDummyIds.SetOwner(!swtch);
}

