{
#ifndef ClingWorkAroundMissingDynamicScope
  gROOT->ProcessLine(".L inherit.C+g"); // .x STreeEvent.so
#endif
  STreeEvent* pEvent = new STreeEvent; pEvent->Init();
  TTree tree; 
  tree.Branch("Event.", "STreeEvent", &pEvent);
  TBranch *b = tree.GetBranch("Event.Fit");
  if (b==0) {
     cerr << "There are no reasons to not have the branch Event.Fit" << endl;
  }

  pEvent->Fit.val = 3;
  new ((*pEvent->Fit.pPhotons)[0]) BottomTrack(11);
  new ((*pEvent->Fit.pPhotons)[1]) BottomTrack(12);
  tree.Fill();

  delete pEvent; pEvent = 0;

  cerr << "Resetting the branch addresses\n";
  tree.SetBranchAddress("Event.",&pEvent);
  tree.GetEntry(0);
  cerr << "For pEvent val is    : " << pEvent->Fit.val << endl;
  if (pEvent->Fit.val != 3) {
     cerr << "Abnormal value of pEvent.Fit.val! It should have been 3." << endl;
     gApplication->Terminate(1);
  }
  BottomTrack * t = dynamic_cast<BottomTrack*>(pEvent->Fit.pPhotons->At(0));
  if (t==0) {
     cerr << "Missing Track in pEvent->Fit.pPhotons" << endl;
     gApplication->Terminate(1);
  }
  if (t->topval != -11) {
     cerr << "Bad value of Track in pEvent->Fit.pPhotons" << endl;
     cerr << t->topval << " instead of " << -11 << endl;
     gApplication->Terminate(1);
  }
  if (t->bottomval != 11) {
     cerr << "Bad value of Track bottomval in pEvent->Fit.pPhotons" << endl;
     cerr << t->bottomval << " instead of " << 11 << endl;
     gApplication->Terminate(1);
  }
  t = dynamic_cast<BottomTrack*>(pEvent->Fit.pPhotons->At(1));
  if (t==0) {
     cerr << "Missing Track in pEvent->Fit.pPhotons" << endl;
     gApplication->Terminate(1);
  }
  if (t->topval != -12) {
     cerr << "Bad value of Track in pEvent->Fit.pPhotons" << endl;
     cerr << t->topval << " instead of " << -12 << endl;
     gApplication->Terminate(1);
  }
  if (t->bottomval != 12) {
     cerr << "Bad value of Track in pEvent->Fit.pPhotons" << endl;
     cerr << t->bottomval << " instead of " << 12 << endl;
     gApplication->Terminate(1);
  }
}
