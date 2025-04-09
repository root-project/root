void runPEvent() {

#ifndef ClingWorkAroundMissingDynamicScope
   gROOT->ProcessLine(".L PEvent.cc+");
#endif

    const Int_t n = 100;
    UInt_t *array = new UInt_t[n];

#ifdef ClingWorkAroundJITandInline
    for(Int_t i(0); i<n; i++)
        array[i] = 300*TMath::Abs(sin(6.28318530717959*i/n));
#else
   for(Int_t i(0); i<n; i++)
      array[i] = 300*TMath::Abs(TMath::Sin(TMath::TwoPi()*i/n));
#endif
    QRawTriggerPulse* tp = new QRawTriggerPulse(n,array);

    TFile *f = new TFile("myTest.root","recreate");
    TTree *t = new TTree("t","mytree");

    PEvent *q = new PEvent(*tp);
    t->Branch("event.","PEvent",&q);
    //t->Branch("thepulse",&tp);


    t->Fill();
    t->AutoSave();

    for(Int_t i(0); i<n; i++)
        cout << i << "   " << tp->GetSample()[i] << endl;

    f->Close();
    delete f;
    delete q;
    delete tp;

#ifdef ClingReinstateRedeclarationAllowed
   TFile *f = new TFile("myTest.root");
   TTree *t = (TTree*)f->Get("t");   
#else
   f = new TFile("myTest.root");
   t = (TTree*)f->Get("t");
#endif
    t->StartViewer();
    t->Draw("event.fRawTriggerPulse.fSample");

}

