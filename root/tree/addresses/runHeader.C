{
  gROOT->ProcessLine(".L header.h+g"); // .x STreeEvent.so
  STreeEvent* pSubEvent = new SubSTreeEvent; pSubEvent->Init();
  TTree tree; 
  tree.Branch("EventBad.", "SubSTreeEvent", &pSubEvent);
  TBranch *b=tree.GetBranch("EventBad.STreeEvent.Clusters");
  if (b==0) {
     cerr << "There are no reasons to not have the branch EventBad.STreeEvent.Clusters" << endl;
     cerr << "The branch is missing only because it is located inside a base class!!!" << endl;
  }
  STreeEvent* pEvent = new STreeEvent; pEvent->Init();  
  tree.Branch("Event.", "STreeEvent", &pEvent);
  b=tree.GetBranch("Event.Clusters");
  if (b==0) {
     cerr << "Now I don't understand the branch Event.Clusters has disappeared!!" << end;
  }
  pSubEvent->Cluster.val = 33;
  pEvent->Cluster.val = 3;
  tree.Fill();

  delete pEvent; pEvent = 0;
  delete pSubEvent; pSubEvent = 0;
  cerr << "Resetting the branch addresses\n";
  tree.SetBranchAddress("Event.",&pEvent);
  tree.SetBranchAddress("EventBad.",&pSubEvent);
  tree.GetEntry(0);
  cerr << "At pEvent   : " << pEvent << ", val is " << pEvent.Clusters.val << endl;
  cerr << "At pSubEvent: " << pSubEvent << ", val is " << pSubEvent.Clusters.val << endl;
}
