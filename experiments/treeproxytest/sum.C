double sum() {

Float_t trk_reduced_chi2=0;

for ( Int_t itrk = 0; itrk < NtpSR.evthdr.ntrack; itrk++ ) 

    {
        if(NtpSR.trk.fit.ndof[itrk] ==0) trk_reduced_chi2=0;
        else {trk_reduced_chi2 = NtpSR.trk.time.chi2[itrk] / NtpSR.trk.fit.ndof[itrk] ;}
}


return trk_reduced_chi2;

}
