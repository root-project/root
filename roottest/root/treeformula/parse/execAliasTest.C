TChain* t=0;
TChain* u=0;
TTree* pUser=0;

TChain* makeChain(TString KEY="") {

  t=new TChain("MeritTuple");
  u=new TChain("user");
  
  TString files="McGen-E5-T45-P0-MonteCarloMcHitsC";
  files+=KEY;
  files+="*";
  
  files+=".root";
  
  t->Add(files);
  u->Add(files);
  pUser=t;
  u->SetAlias("TracksFilter","TkrNumTracks > 0");
  //TPython::ExecScript("yamlScript.py");

  u->AddFriend(t,"M");

  cout << "u:" << u->GetEntries() << endl;

  return u;

}

int execAliasTest()
{
   TChain *u = makeChain();
   return 0 != u->Scan("TracksFilter:TkrNumTracks:FT1Energy","radlenAtHit>0.0&&TracksFilter&&TkrNumTracks==0","",2,26);
}
