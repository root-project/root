double script() {
   int ntracks = fNtrack;
   fprintf(stderr,"did run script and found %d tracks\n",ntracks);

   // Let's test reading a little bit;
   float temp = fTemperature;
   int meas0 = fMeasures[0];
   int meas1 = fMeasures[3];
   
   if (fChain->GetReadEntry()==1) {
      cout << ntracks << "==" << event.fNtrack << endl;
      cout << fNtrack << endl;
      cout << temp << endl;
      cout << meas0 << endl;
      cout << meas1 << endl;
      cout << fClosestDistance[2] << endl;
      cout << fType[0] << endl;
// there are problems with std::string in interpreted mode :(
#ifdef __CINT__
      // The aut-conversion to std::string does not work in CINT.
      const char *ctype = fType.c_str();
      string type = ctype;
      // the operator<<(std::string) is not available in CINT.
      //cout << type.c_str() << endl;
      cout << ctype << endl;
#else
      string type = fType;
      cout << type << endl;
#endif
      cout << "fMatrix[2][1]: " << fMatrix[2][1] << endl;
      cout << "fH->GetMean() " << fH->GetMean() << endl;

#ifdef seold_cxx
      cout << fEvtHdr_fEvtNum << endl;
      cout << fTracks_ << endl;
      cout << fTracks_fPx[0] << endl;
      cout << fTracks_fPx[1] << endl;
      cout << "fTracks.fNsp[2]: " << fTracks_fNsp[2] << endl;
      cout << "fTracks.fPointValue[2][1]: " << fTracks_fPointValue[2][1] << endl;
      cout << "fLastTrack: " << fLastTrack.GetUniqueID() << endl;
#else
      cout << fEvtHdr.fEvtNum << endl;
      cout << fTracks->GetLast()+1 << endl;
      cout << fTracks.fPx[0] << endl;
      cout << fTracks.fPx[1] << endl;
      cout << "fTracks.fNsp[0]: " << fTracks.fNsp[0] << endl;
      cout << "fTracks.fPointValue[0][0]: " << fTracks.fPointValue[0][0] << endl;
      cout << "fLastTrack: " << fLastTrack->GetUniqueID() << endl;
#endif
      // new!
      cout << "fTriggerBits nbits " << fTriggerBits->GetNbits() << "==" << fTriggerBits.fNbits<< endl;
      cout << "fTracks[2].fTriggerBits.fNbits " << fTracks.fTriggerBits[2]->GetNbits()  << "==" << fTracks.fTriggerBits.fNbits[2]  << endl;
#ifdef WITH_EVENT
#ifdef seold_cxx
#else
      cout << "event->GetHeader()->GetEvtNum(): " << event->GetHeader()->GetEvtNum() << endl;
      // cout << "event->GetHeader()->GetEvtNum(): " << const_cast<Event*>(event)->GetHeader()->GetEvtNum() << endl;
      cout << "event->GetHeader()->GetEvtNum(): " << ((Event&)event).GetHeader()->GetEvtNum() << endl;
#endif      
#endif

   }
#ifdef __CINT__
   // bug in conversions
   return ntracks;
#else 
   return fNtrack;
#endif

}
