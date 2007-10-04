// @(#)root/test:$Id$
// Author: Rene Brun   10/01/97

{
//  This macro is a variant of the macro eventa.
//
//  NOTE: Before executing this macro, you must have executed the macro eventload.
//
//  This small program loop on all events:
//    - It reads the small branch containing the number of tracks per event
//    - It reads the full event only for events having less than 587 tracks
//    - It dumps the selected events.

   gROOT->Reset();

//   Connect file generated in $ROOTSYS/test
   TFile f("Event.root");

//   Read Tree named "T" in memory. Tree pointer is assigned the same name
   TTree *T = (TTree*)f.Get("T");

//   Create a timer object to benchmark this loop
   TStopwatch timer;
   timer.Start();

//   Start main loop on all events
   Event *event = new Event();   //we create the event object once outside the loop

   TBranch *bntrack = T->GetBranch("fNtrack");
   TBranch *branch  = T->GetBranch("event");
   branch->SetAddress(&event);
   Int_t nevent = T->GetEntries();
   Int_t nselected = 0;
   Int_t nb = 0;
   for (Int_t i=0;i<nevent;i++) {
      if(i%50 == 0) printf("Event:%d\n",i);
      bntrack->GetEntry(i);                  //read branch "fNtrack" only
      if (event->GetNtrack() > 587)continue; //reject events with more than 587 tracks
      nb += T->GetEntry(i);                  //read complete accepted event in memory
      nselected++;
      if (nselected == 1) event->Dump();     //dump the first accepted event
      event->Clear();                        //clear tracks array
   }

//  Stop timer and print results
   timer.Stop();
   Float_t mbytes = T->GetTotBytes()*1.e-6;
   Double_t rtime = timer.RealTime();
   Double_t ctime = timer.CpuTime();
   printf("You have selected %d events out of %d\n",nselected,nevent);
   printf("RealTime=%f seconds, CpuTime=%f seconds\n",rtime,ctime);
   printf("You have scanned %f Mbytes/Realtime seconds\n",mbytes/rtime);
   printf("You have scanned %f Mbytes/Cputime seconds\n",mbytes/ctime);

   f.Close();
}
