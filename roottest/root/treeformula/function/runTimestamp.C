{

  TTimeStamp* ts = new TTimeStamp(2007,11,06,10,21,00);
  TTree* tree = new TTree("EventShape","Event shape global variables");
  tree->Branch("time", "TTimeStamp", &ts, 32000, 0);
  tree->Branch("timesplit", "TTimeStamp", &ts, 32000, 9);
  tree->Fill();
#ifdef ClingWorkAroundMissingDynamicScope
   TTree *EventShape = tree;
#endif
  EventShape->Scan("time.AsDouble()","","colsize=20");
  EventShape->Scan("time","","colsize=20");
  EventShape->Scan("timesplit.AsDouble()","","colsize=20");
  EventShape->Scan("timesplit","","colsize=20");
#ifdef ClingWorkAroundBrokenUnnamedReturn
  gApplication->Terminate(0);
#else
  return 0;
#endif
}
