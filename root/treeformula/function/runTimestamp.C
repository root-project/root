{

  TTimeStamp* ts = new TTimeStamp;
  TTree* tree = new TTree("EventShape","Event shape global variables");
  tree->Branch("time", "TTimeStamp", &ts, 32000, 0);
  tree->Branch("timesplit", "TTimeStamp", &ts, 32000, 9);
  tree->Fill();
  EventShape->Scan("time.AsDouble()");
  EventShape->Scan("time");
  EventShape->Scan("timesplit.AsDouble()");
  EventShape->Scan("timesplit");
  return 0;
}
