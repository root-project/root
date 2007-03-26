bool dt_RefProxyTest() {
  gSystem->Load("libTreePlayer");
  gSystem->Load("libEvent");

  const char* fname;

  fname = "Event.new.split0.root";
  printf("\n\n\n=====================  %s  =================================\n\n\n",fname);
  TFile::Open(fname);
  T->SetScanField(0);
  T->Scan("fLastTrack.fUniqueID:fLastTrack.fUniqueID&0xFFFF:fLastTrack@.fUniqueID:fLastTrack.fPx:fLastTrack.GetPx():fTracks[fTracks@.size()-1].fPx:fTracks[fTracks@.size()-1].GetPx():fTracks[0].fPx","","precision=3");
  T->Scan("fLastTrack.fUniqueID:fLastTrack@.fUniqueID:fLastTrack.fUniqueID&0xFFFF:fTracks[fTracks@.size()-1].fUniqueID&0xFFFF:fTracks[0].fUniqueID&0xFFFF","","precision=3");
  T->Scan("fLastTrack.GetUniqueID():fLastTrack@.GetUniqueID():fLastTrack.GetUniqueID()&0xFFFF:fTracks[fTracks@.size()-1].fUniqueID&0xFFFF:fTracks[0].GetUniqueID()&0xFFFF","","precision=3");
  T->Scan("fHighPt.fUniqueID:fHighPt.fUniqueID&0xFFFF:fHighPt.fPx:fHighPt.fPy:fHighPt.fPz:fTracks[fHighPt.fUniqueID&0xFFFF-1].fUniqueID&0xFFFF:fTracks[fHighPt.fUniqueID&0xFFFF-1].fPx:fTracks[fHighPt.fUniqueID&0xFFFF-1].fPy:fTracks[fHighPt.fUniqueID&0xFFFF-1].fPz","","precision=3",1);

  fname = "Event.new.split1.root";
  printf("\n\n\n=====================  %s  =================================\n\n\n",fname);
  TFile::Open(fname);
  T->SetScanField(0);
  T->Scan("fLastTrack.fUniqueID:fLastTrack.fUniqueID&0xFFFF:fLastTrack@.fUniqueID:fLastTrack.fPx:fLastTrack.GetPx():fTracks[fTracks@.size()-1].fPx:fTracks[fTracks@.size()-1].GetPx():fTracks[0].fPx","","precision=3");
  T->Scan("fLastTrack.fUniqueID:fLastTrack@.fUniqueID:fLastTrack.fUniqueID&0xFFFF:fTracks[fTracks@.size()-1].fUniqueID&0xFFFF:fTracks[0].fUniqueID&0xFFFF","","precision=3");
  T->Scan("fLastTrack.GetUniqueID():fLastTrack@.GetUniqueID():fLastTrack.GetUniqueID()&0xFFFF:fTracks[fTracks@.size()-1].fUniqueID&0xFFFF:fTracks[0].GetUniqueID()&0xFFFF","","precision=3");
  T->Scan("fHighPt.fUniqueID:fHighPt.fUniqueID&0xFFFF:fHighPt.fPx:fHighPt.fPy:fHighPt.fPz:fTracks[fHighPt.fUniqueID&0xFFFF-1].fUniqueID&0xFFFF:fTracks[fHighPt.fUniqueID&0xFFFF-1].fPx:fTracks[fHighPt.fUniqueID&0xFFFF-1].fPy:fTracks[fHighPt.fUniqueID&0xFFFF-1].fPz","","precision=3",1);

  fname = "Event.new.split2.root";
  printf("\n\n\n=====================  %s  =================================\n\n\n",fname);
  TFile::Open(fname);
  T->SetScanField(0);
  T->Scan("fLastTrack.fUniqueID:fLastTrack.fUniqueID&0xFFFF:fLastTrack@.fUniqueID:fLastTrack.fPx:fLastTrack.GetPx():fTracks[fTracks@.size()-1].fPx:fTracks[fTracks@.size()-1].GetPx():fTracks[0].fPx","","precision=3");
  T->Scan("fLastTrack.fUniqueID:fLastTrack@.fUniqueID:fLastTrack.fUniqueID&0xFFFF:fTracks[fTracks@.size()-1].fUniqueID&0xFFFF:fTracks[0].fUniqueID&0xFFFF","","precision=3");
  T->Scan("fLastTrack.GetUniqueID():fLastTrack@.GetUniqueID():fLastTrack.GetUniqueID()&0xFFFF:fTracks[fTracks@.size()-1].GetUniqueID()&0xFFFF:fTracks[0].GetUniqueID()&0xFFFF","","precision=3");
  T->Scan("fHighPt.fUniqueID:fHighPt.fUniqueID&0xFFFF:fHighPt.fPx:fHighPt.fPy:fHighPt.fPz:fTracks[fHighPt.fUniqueID&0xFFFF-1].fUniqueID&0xFFFF:fTracks[fHighPt.fUniqueID&0xFFFF-1].fPx:fTracks[fHighPt.fUniqueID&0xFFFF-1].fPy:fTracks[fHighPt.fUniqueID&0xFFFF-1].fPz","","precision=3",1);

  fname = "Event.new.split9.root";
  printf("\n\n\n=====================  %s  =================================\n\n\n",fname);
  TFile::Open(fname);
  T->SetScanField(0);
  T->Scan("fLastTrack.fUniqueID:fLastTrack.fUniqueID&0xFFFF:fLastTrack@.fUniqueID:fLastTrack.fPx:fLastTrack.GetPx():fTracks[fTracks@.size()-1].fPx:fTracks[fTracks@.size()-1].GetPx():fTracks[0].fPx","","precision=3");
  T->Scan("fLastTrack.fUniqueID:fLastTrack@.fUniqueID:fLastTrack.fUniqueID&0xFFFF:fLastTrack.GetUniqueID()&0xFFFF:fTracks[fTracks@.size()-1].fUniqueID&0xFFFF:fTracks[0].fUniqueID&0xFFFF","","precision=3");
  T->Scan("fLastTrack.GetUniqueID():fLastTrack@.GetUniqueID():fLastTrack.GetUniqueID()&0xFFFF:fTracks[fTracks@.size()-1].GetUniqueID()&0xFFFF:fTracks[0].GetUniqueID()&0xFFFF","","precision=3");
  T->Scan("fHighPt.fUniqueID:fHighPt.fUniqueID&0xFFFF:fHighPt.fPx:fHighPt.fPy:fHighPt.fPz:fTracks[fHighPt.fUniqueID&0xFFFF-1].fUniqueID&0xFFFF:fTracks[fHighPt.fUniqueID&0xFFFF-1].fPx:fTracks[fHighPt.fUniqueID&0xFFFF-1].fPy:fTracks[fHighPt.fUniqueID&0xFFFF-1].fPz","","precision=3",1);

  return true;
}   
