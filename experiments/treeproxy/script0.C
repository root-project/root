double script0() {
   int ntracks = event.fNtrack;
   fprintf(stderr,"did run script and found %d tracks\n",ntracks);

   // Let's test reading a little bit;
   float temp = event.fTemperature;
   int meas0 = event.fMeasures[0];
   int meas1 = event.fMeasures[3];
   
   if (fChain->GetReadEntry()==1) {
      cout << ntracks << "==" << event.fNtrack << endl;
      cout << event.fNtrack << endl;
      cout << temp << endl;
      cout << meas0 << endl;
      cout << meas1 << endl;
      cout << event.fClosestDistance[2] << endl;
      cout << event.fType[0] << endl;
// there are problems with std::string in interpreted mode :(
#ifdef __CINT__
      // The aut-conversion to std::string does not work in CINT.
      const char *ctype = event.fType.c_str();
      string type = ctype;
      // the operator<<(std::string) is not available in CINT.
      //cout << type.c_str() << endl;
      cout << ctype << endl;
#else
      string type = event.fType;
      cout << type << endl;
#endif
      cout << "fMatrix[2][1]: " << event.fMatrix[2][1] << endl;
      cout << "fH->GetMean() " << event.fH->GetMean() << endl;

#ifdef seold_cxx
      cout << event.fEvtHdr_fEvtNum << endl;
      cout << event.fTracks_ << endl;
      cout << event.fTracks_fPx[0] << endl;
      cout << event.fTracks_fPx[1] << endl;
      cout << "fTracks.fNsp[2]: " << event.fTracks_fNsp[2] << endl;
      cout << "fTracks.fPointValue[2][1]: " << event.fTracks_fPointValue[2][1] << endl;
      cout << "fLastTrack: " << event.fLastTrack.GetUniqueID() << endl;
#else
      cout << event.fEvtHdr.fEvtNum << endl;
      cout << event.fTracks->GetLast()+1 << endl;
      cout << event.fTracks.fPx[0] << endl;
      cout << event.fTracks.fPx[1] << endl;
      cout << "fTracks.fNsp[0]: " << event.fTracks.fNsp[0] << endl;
      cout << "fTracks.fPointValue[0][0]: " << event.fTracks.fPointValue[0][0] << endl;
      cout << "fLastTrack: " << event.fLastTrack->GetUniqueID() << endl;
#endif
      // new!
      cout << "fTriggerBits nbits " << event.fTriggerBits->GetNbits() << "==" << event.fTriggerBits.fNbits<< endl;
      cout << "fTracks[2].fTriggerBits.fNbits " << event.fTracks.fTriggerBits[2]->GetNbits()  << "==" << event.fTracks.fTriggerBits.fNbits[2]  << endl;
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
   return event.fNtrack;
#endif

}
