{
new TFile("digi.root");
int n = Digi->BuildIndex("m_runId", "m_eventId");
return ! (n>0); // to signal succesfull we need a zero!
}
