Bool_t h1analysisCut()
{
// entry is the entry number in the current Tree
// Selection function to select D* and D0.
         
   //in case one event list is given in input, the selection has already been done.
   if (useList) return kTRUE;

   // Read only the necessary branches to select entries.
   // return as soon as a bad entry is detected
   // to read complete event, call fChain->GetTree()->GetEntry(entry)

   float f1 = md0_d;
   float f2 = md0_d-1.8646;
   bool test = TMath::Abs(md0_d-1.8646) >= 0.04;
   if (gDebug>0) fprintf(stderr,"entry #%d f1=%f f2=%f test=%d\n",
           fChain->GetReadEntry(),f1,f2,test);

   if (TMath::Abs(md0_d-1.8646) >= 0.04) return kFALSE;
   if (ptds_d <= 2.5) return kFALSE;
   if (TMath::Abs(etads_d) >= 1.5) return kFALSE;

   int cik = ik-1;    //original ik used f77 convention starting at 1
   int cipi = ipi-1;  //original ipi used f77 convention starting at 1

   f1 = nhitrp[cik];
   f2 = nhitrp[cipi];
   test = nhitrp[cik]*nhitrp[cipi] <= 1;
   if (gDebug>0) fprintf(stderr,"entry #%d f1=%f f2=%f test=%d\n",
                         fChain->GetReadEntry(),f1,f2,test);
   
   if (nhitrp[cik]*nhitrp[cipi] <= 1) return kFALSE;
   if (rend[cik] -rstart[cik]  <= 22) return kFALSE;
   if (rend[cipi]-rstart[cipi] <= 22) return kFALSE;
   if (nlhk[cik] <= 0.1)    return kFALSE;
   if (nlhpi[cipi] <= 0.1)  return kFALSE;
   // fix because read-only 
   if (nlhpi[ipis-1] <= 0.1) return kFALSE;
   if (njets < 1)          return kFALSE;
   
   // if option fillList, fill the event list
   if (fillList) elist->Enter(fChain->GetChainEntryNumber(fChain->GetReadEntry()));
   if (gDebug>0) fprintf(stderr,"accepted entry #%d\n",fChain->GetReadEntry());
   return kTRUE;
}

