// @(#)root/hist:$Name:  $:$Id: TLimit.cxx,v 1.5 2003/03/21 14:53:49 brun Exp $
// Author: Christophe.Delaere@cern.ch   21/08/2002

///////////////////////////////////////////////////////////////////////////
//
// TLimit
//
// Class to compute 95% CL limits
//
///////////////////////////////////////////////////////////////////////////

/*************************************************************************
 * C.Delaere                                                             *
 * adapted from the mclimit code from Tom Junk                           *
 * see http://cern.ch/thomasj/searchlimits/ecl.html                      *
 *************************************************************************/

#include "TLimit.h"
#include "TArrayD.h"
#include "TOrdCollection.h"
#include "TConfidenceLevel.h"
#include "TLimitDataSource.h"
#include "TRandom3.h"
#include "TH1.h"
#include "TObjArray.h"
#include "TMath.h"
#include "TIterator.h"
#include "TObjString.h"
#include "TClassTable.h"
#include "Riostream.h"

ClassImp(TLimit)

TArrayD *TLimit::fgTable = new TArrayD(0);
TOrdCollection *TLimit::fgSystNames = new TOrdCollection();

TConfidenceLevel *TLimit::ComputeLimit(TLimitDataSource * data,
                                       Int_t nmc, bool stat,
				       TRandom * generator,
                                       Double_t(*statistic) (Double_t,
                                                             Double_t,
                                                             Double_t))
{
   // class TLimit
   // ------------
   //
   // Algorithm to compute 95% C.L. limits using the Likelihood ratio
   // semi-bayesian method.
   // It takes signal, background and data histograms wrapped in a
   // TLimitDataSource as input and runs a set of Monte Carlo experiments in
   // order to compute the limits. If needed, inputs are fluctuated according
   // to systematics. The output is a TConfidenceLevel.
   //
   // class TLimitDataSource
   // ----------------------
   //
   // Takes the signal, background and data histograms as well as different
   // systematics sources to form the TLimit input.
   //
   //  class TConfidenceLevel
   //  ----------------------
   //
   // Final result of the TLimit algorithm. It is created just after the
   // time-consuming part and can be stored in a TFile for further processing.
   // It contains light methods to return CLs, CLb and other interesting
   // quantities.
   //
   // The actual algorithm...
   // From an input (TLimitDataSource) it produces an output TConfidenceLevel.
   // For this, nmc Monte Carlo experiments are performed.
   // As usual, the larger this number, the longer the compute time,
   // but the better the result.
   //Begin_Html
   /*
   <FONT SIZE=+0>
   <p>Supposing that there is a plotfile.root file containing 3 histograms
           (signal, background and data), you can imagine doing things like:</p>
   <p>
   <BLOCKQUOTE><PRE>
    TFile* infile=new TFile("plotfile.root","READ");
    infile->cd();
    TH1D* sh=(TH1D*)infile->Get("signal");
    TH1D* bh=(TH1D*)infile->Get("background");
    TH1D* dh=(TH1D*)infile->Get("data");
    TLimitDataSource* mydatasource = new TLimitDataSource(sh,bh,dh);
    TConfidenceLevel *myconfidence = TLimit::ComputeLimit(mydatasource,50000);
    cout &lt&lt "  CLs    : " &lt&lt myconfidence->CLs()  &lt&lt endl;
    cout &lt&lt "  CLsb   : " &lt&lt myconfidence->CLsb() &lt&lt endl;
    cout &lt&lt "  CLb    : " &lt&lt myconfidence->CLb()  &lt&lt endl;
    cout &lt&lt "&lt CLs &gt  : " &lt&lt myconfidence->GetExpectedCLs_b()  &lt&lt endl;
    cout &lt&lt "&lt CLsb &gt : " &lt&lt myconfidence->GetExpectedCLsb_b() &lt&lt endl;
    cout &lt&lt "&lt CLb &gt  : " &lt&lt myconfidence->GetExpectedCLb_b()  &lt&lt endl;
    delete myconfidence;
    delete mydatasource;
    infile->Close();
   </PRE></BLOCKQUOTE></p>
   <p></p>
   <p>More informations can still be found on
   <a HREF="http://cern.ch/aleph-proj-alphapp/doc/tlimit.html">this</a> page.</p>
   </FONT>
   */
   //End_Html

   // The final object returned...
   TConfidenceLevel *result = new TConfidenceLevel(nmc);
   // The random generator used...
   TRandom *myrandom = generator ? generator : new TRandom3;
   // Compute some total quantities on all the channels
   Int_t nbins   = 0;
   Int_t maxbins = 0;
   Double_t nsig = 0;
   Double_t nbg  = 0;
   Int_t ncand   = 0;
   Int_t i;
   for (i = 0; i <= data->GetSignal()->GetLast(); i++) {
      nbins  += ((TH1D *) (data->GetSignal()->At(i)))->GetNbinsX();
      maxbins = ((TH1D *) (data->GetSignal()->At(i)))->GetNbinsX() > maxbins ?
	        ((TH1D *) (data->GetSignal()->At(i)))->GetNbinsX() + 1 : maxbins;
      nsig   += ((TH1D *) (data->GetSignal()->At(i)))->Integral();
      nbg    += ((TH1D *) (data->GetBackground()->At(i)))->Integral();
      ncand  += (Int_t) ((TH1D *) (data->GetCandidates()->At(i)))->Integral();
   }
   result->SetBtot(nbg);
   result->SetStot(nsig);
   result->SetDtot(ncand);
   Double_t buffer = 0;
   fgTable->Set(maxbins * (data->GetSignal()->GetLast() + 1));
   for (Int_t channel = 0; channel <= data->GetSignal()->GetLast(); channel++)
      for (Int_t bin = 0;
           bin <= ((TH1D *) (data->GetSignal()->At(channel)))->GetNbinsX();
           bin++) {
         Double_t s = (Double_t) ((TH1D *) (data->GetSignal()->At(channel)))->GetBinContent(bin);
         Double_t b = (Double_t) ((TH1D *) (data->GetBackground()->At(channel)))->GetBinContent(bin);
         Double_t d = (Double_t) ((TH1D *) (data->GetCandidates()->At(channel)))->GetBinContent(bin);
         // Compute the value of the "-2lnQ" for the actual data
         if ((b == 0) && (s > 0)) {
            cout << "WARNING: Ignoring bin " << bin << " of channel "
                 << channel << " which has s=" << s << " but b=" << b << endl;
            cout << "         Maybe the MC statistic has to be improved..." << endl;
         }
         if ((s > 0) && (b > 0))
            buffer += statistic(s, b, d);
         // precompute the log(1+s/b)'s in an array to speed up computation
         // background-free bins are set to have a maximum t.s. value
         // for protection (corresponding to s/b of about 5E8)
         if ((s > 0) && (b > 0))
            fgTable->AddAt(statistic(s, b, 1), (channel * maxbins) + bin);
         else if ((s > 0) && (b == 0))
            fgTable->AddAt(20, (channel * maxbins) + bin);
      }
   result->SetTSD(buffer);
   // accumulate MC experiments.  Hold the test statistic function fixed, but
   // fluctuate s and b within syst. errors for computing probabilities of
   // having that outcome.  (Alex Read's prescription -- errors are on the ensemble,
   // not on the observed test statistic.  This technique does not split outcomes.)
   // keep the tstats as sum log(1+s/b). convert to -2lnQ when preparing the results
   // (reason -- like to keep the < signs right)
   Double_t *tss = new Double_t[nmc];
   Double_t *tsb = new Double_t[nmc];
   Double_t *lrs = new Double_t[nmc];
   Double_t *lrb = new Double_t[nmc];
   for (i = 0; i < nmc; i++) {
      tss[i] = 0;
      tsb[i] = 0;
      lrs[i] = 0;
      lrb[i] = 0;
      // fluctuate signal and background
      TLimitDataSource *fluctuated = Fluctuate(data, !i, myrandom, stat);
      for (Int_t channel = 0;
           channel <= fluctuated->GetSignal()->GetLast(); channel++) {
         for (Int_t bin = 0;
              bin <=((TH1D *) (fluctuated->GetSignal()->At(channel)))->GetNbinsX();
              bin++) {
            if ((Double_t) ((TH1D *) (fluctuated->GetSignal()->At(channel)))->GetBinContent(bin) != 0) {
               // s+b hypothesis
               Double_t rate = (Double_t) ((TH1D *) (fluctuated->GetSignal()->At(channel)))->GetBinContent(bin) +
                               (Double_t) ((TH1D *) (fluctuated->GetBackground()->At(channel)))->GetBinContent(bin);
               Double_t rand = myrandom->Poisson(rate);
               tss[i] += rand * fgTable->At((channel * maxbins) + bin);
               Double_t s = (Double_t) ((TH1D *) (fluctuated->GetSignal()->At(channel)))->GetBinContent(bin);
               Double_t b = (Double_t) ((TH1D *) (fluctuated->GetBackground()->At(channel)))->GetBinContent(bin);
               if ((s > 0) && (b > 0))
                  lrs[i] += statistic(s, b, rand) - s;
               else if ((s > 0) && (b == 0))
                  lrs[i] += 20 * rand - s;
               // b hypothesis
               rate = (Double_t) ((TH1D *) (fluctuated->GetBackground()->At(channel)))->GetBinContent(bin);
               rand = myrandom->Poisson(rate);
               tsb[i] += rand * fgTable->At((channel * maxbins) + bin);
               if ((s > 0) && (b > 0))
                  lrb[i] += statistic(s, b, rand) - s;
               else if ((s > 0) && (b == 0))
                  lrb[i] += 20 * rand - s;
            }
         }
      }
      lrs[i] = TMath::Exp(lrs[i]);
      lrb[i] = TMath::Exp(lrb[i]);
      if (data != fluctuated)
         delete fluctuated;
   }
   // lrs and lrb are the LR's (no logs) = prob(s+b)/prob(b) for
   // that choice of s and b within syst. errors in the ensemble.  These are
   // the MC experiment weights for relating the s+b and b PDF's of the unsmeared
   // test statistic (in which cas one can use another test statistic if one likes).

   // Now produce the output object.
   // The final quantities are computed on-demand form the arrays tss, tsb, lrs and lrb.
   result->SetTSS(tss);
   result->SetTSB(tsb);
   result->SetLRS(lrs);
   result->SetLRB(lrb);
   if (!generator)
      delete myrandom;
   return result;
}

TLimitDataSource *TLimit::Fluctuate(TLimitDataSource * input, bool init,
                                    TRandom * generator, bool stat)
{
   // initialisation: create a sorted list of all the names of systematics
   if (init) {
      // create a "map" with the systematics names
      TIterator *errornames = input->GetErrorNames()->MakeIterator();
      TObjArray *listofnames = 0;
      while ((listofnames = ((TObjArray *) errornames->Next()))) {
         TObjString *name = NULL;
         TIterator *loniter = listofnames->MakeIterator();
         while ((name = (TObjString *) (loniter->Next())))
            if ((fgSystNames->IndexOf(name)) < 0)
               fgSystNames->AddLast(name);
      }
      fgSystNames->Sort();
   }
   // if there are no systematics, just returns the input as "fluctuated" output
   if ((fgSystNames->GetSize() <= 0)&&(!stat))
      return input;
   // if there are only stat, just fluctuate stats.
   if (fgSystNames->GetSize() <= 0) {
     TLimitDataSource *result = new TLimitDataSource();
     result->SetOwner();
     for (Int_t channel = 0; channel <= input->GetSignal()->GetLast(); channel++) {
        TH1D *newsignal = new TH1D(*(TH1D *) (input->GetSignal()->At(channel)));
        if(stat)
           for(int i=1; i<=newsignal->GetNbinsX(); i++) {
              newsignal->SetBinContent(i,newsignal->GetBinContent(i)+generator->Gaus(0,newsignal->GetBinError(i)));
           }
        newsignal->SetDirectory(0);
        TH1D *newbackground = new TH1D(*(TH1D *) (input->GetBackground()->At(channel)));
        if(stat)
           for(int i=1; i<=newbackground->GetNbinsX(); i++)
              newbackground->SetBinContent(i,newbackground->GetBinContent(i)+generator->Gaus(0,newbackground->GetBinError(i)));
        newbackground->SetDirectory(0);
        TH1D *newcandidates = new TH1D(*(TH1D *) (input->GetCandidates()));
        newcandidates->SetDirectory(0);
        result->AddChannel(newsignal, newbackground, newcandidates);
     }
        return result;
   }
   // Find a choice for the random variation and
   // re-toss all random numbers if any background or signal
   // goes negative.  (background = 0 is bad too, so put a little protection
   // around it -- must have at least 10% of the bg estimate).
   bool retoss = kTRUE;
   Double_t *serrf = NULL;
   Double_t *berrf = NULL;
   do {
      Double_t *toss = new Double_t[fgSystNames->GetSize()];
      for (Int_t i = 0; i < fgSystNames->GetSize(); i++)
         toss[i] = generator->Gaus(0, 1);
      retoss = kFALSE;
      serrf = new Double_t[(input->GetSignal()->GetLast()) + 1];
      berrf = new Double_t[(input->GetSignal()->GetLast()) + 1];
      for (Int_t channel = 0;
           channel <= input->GetSignal()->GetLast();
           channel++) {
         serrf[channel] = 0;
         berrf[channel] = 0;
         for (Int_t bin = 0;
              bin <=((TH1D *) (input->GetErrorOnSignal()->At(channel)))->GetNbinsX();
	      bin++) {
            serrf[channel] += ((TH1D *) (input->GetErrorOnSignal()->At(channel)))->GetBinContent(bin) *
                toss[fgSystNames->BinarySearch((TObjString*) (((TObjArray *) (input->GetErrorNames()->At(channel)))->At(bin)))];
            berrf[channel] += ((TH1D *) (input->GetErrorOnBackground()->At(channel)))->GetBinContent(bin) *
                toss[fgSystNames->BinarySearch((TObjString*) (((TObjArray *) (input->GetErrorNames()->At(channel)))->At(bin)))];
         }
         if ((serrf[channel] < -1.0) || (berrf[channel] < -0.9)) {
            retoss = kTRUE;
            continue;
         }
      }
      delete[]toss;
   } while (retoss);
   // adjust the fluctuated signal and background counts with a legal set
   // of random fluctuations above.
   TLimitDataSource *result = new TLimitDataSource();
   result->SetOwner();
   for (Int_t channel = 0; channel <= input->GetSignal()->GetLast();
        channel++) {
      TH1D *newsignal = new TH1D(*(TH1D *) (input->GetSignal()->At(channel)));
      if(stat)
         for(int i=1; i<=newsignal->GetNbinsX(); i++) {
            newsignal->SetBinContent(i,newsignal->GetBinContent(i)+generator->Gaus(0,newsignal->GetBinError(i)));
	 }
      newsignal->Scale(1 + serrf[channel]);
      newsignal->SetDirectory(0);
      TH1D *newbackground = new TH1D(*(TH1D *) (input->GetBackground()->At(channel)));
      if(stat)
         for(int i=1; i<=newbackground->GetNbinsX(); i++)
            newbackground->SetBinContent(i,newbackground->GetBinContent(i)+generator->Gaus(0,newbackground->GetBinError(i)));
      newbackground->Scale(1 + berrf[channel]);
      newbackground->SetDirectory(0);
      TH1D *newcandidates = new TH1D(*(TH1D *) (input->GetCandidates()));
      newcandidates->SetDirectory(0);
      result->AddChannel(newsignal, newbackground, newcandidates);
   }
   delete[] serrf;
   delete[] berrf;
   return result;
}

