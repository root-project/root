{

  TTimeStamp* ts = new TTimeStamp;
  TTree* tree = new TTree("EventShape","Event shape global variables");
  tree->Branch("time", "TTimeStamp", &ts, 32000, 0);
  tree->Fill();
  EventShape->Scan("time.AsDouble()");
  EventShape->Scan("time");
  return 0;
}
