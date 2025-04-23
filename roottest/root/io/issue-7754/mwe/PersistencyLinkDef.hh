#ifdef __ROOTCLING__

#pragma link C++ class TVHit+;
#pragma link C++ class TVDigi+;
#pragma link C++ class TVEvent+;
#pragma link C++ class TDetectorVEvent+;
#pragma link C++ class TDetectorVHit+;
#pragma link C++ class SpectrometerChannelID+;
#pragma link C++ class TSpectrometerHit+;
#pragma link C++ class TSpectrometerEvent+;
#pragma read sourceClass="TVHit" version="[1]" source="Int_t fMCTrackID" targetClass="TVHit" target="fKinePartIndex" code="{ fKinePartIndex = onfile.fMCTrackID;}"

#endif
