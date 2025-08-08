{
#ifndef ClingWorkAroundMissingDynamicScope
   gROOT->ProcessLine(".L header.h+g"); // .x STreeEvent.so
#endif
   STreeEvent* pSubEvent = new SubSTreeEvent; pSubEvent->Init();
   TTree tree; 
   TBranchElement *t1 = (TBranchElement*)tree.Branch("EventBad.", "SubSTreeEvent", &pSubEvent);
   TBranch *b=tree.GetBranch("EventBad.STreeEvent.Clusters");
   if (b==0) {
      cerr << "There are no reasons to not have the branch EventBad.STreeEvent.Clusters" << endl;
      cerr << "The branch is missing only because it is located inside a base class!!!" << endl;
   }
   STreeEvent* pEvent = new STreeEvent; pEvent->Init();  
   TBranchElement *t2 = (TBranchElement*)tree.Branch("Event.", "STreeEvent", &pEvent);
   b=tree.GetBranch("Event.Clusters");
   if (b==0) {
      cerr << "Now I don't understand the branch Event.Clusters has disappeared!!" << endl;
   }
   pSubEvent->header.number = 11;
   pSubEvent->Clusters.val = 33;
   pEvent->header.number = 1;
   pEvent->Clusters.val = 3;
   new ((*pEvent->Clusters.pPhotons)[0]) Track(11);
   new ((*pEvent->Clusters.pPhotons)[1]) Track(12);
   tree.Fill();
   
   delete pEvent; pEvent = 0;
   delete pSubEvent; pSubEvent = 0;
   cerr << "Resetting the branch addresses\n";
   tree.SetBranchAddress("Event.",&pEvent);
   tree.SetBranchAddress("EventBad.",&pSubEvent);
   tree.GetEntry(0);
   cerr << "For pEvent val is    : " << pEvent->Clusters.val << endl;
   cerr << "For pSubEvent val is : " << pSubEvent->Clusters.val << endl;
   cerr << "For pEvent number is    : " << pEvent->header.number << endl;
   cerr << "For pSubEvent number is : " << pSubEvent->header.number << endl;
   if (pEvent->Clusters.val != 3) {
      cerr << "Abnormal value of pEvent.Clusters.val! It should have been 3." << endl;
      gApplication->Terminate(1);
   }
   if (pSubEvent->Clusters.val != 33) {
      cerr << "Abnormal value of pSubEvent.Clusters.val! It should have been 33." << endl;
      gApplication->Terminate(1);
   }
   Track * t = dynamic_cast<Track*>(pEvent->Clusters.pPhotons->At(0));
   if (t==0) {
      cerr << "Missing Track in pEvent->Cluster.pPhotons" << endl;
      gApplication->Terminate(1);
   }
   if (t->a != 11) {
      cerr << "Bad value of Track in pEvent->Cluster.pPhotons" << endl;
      cerr << t->a << " instead of " << 11 << endl;
      gApplication->Terminate(1);
   }
   t = dynamic_cast<Track*>(pEvent->Clusters.pPhotons->At(1));
   if (t==0) {
      cerr << "Missing Track in pEvent->Cluster.pPhotons" << endl;
      gApplication->Terminate(1);
   }
   if (t->a != 12) {
      cerr << "Bad value of Track in pEvent->Cluster.pPhotons" << endl;
      cerr << t->a << " instead of " << 12 << endl;
      gApplication->Terminate(1);
   }
}
