#ifndef VERSION
#error You need to explicitly specify a version
#endif

#include "TClonesArray.h"
#include <vector>
#include <list>

namespace std {} using namespace std;

class Track : public TObject {
   double fEnergy;
public:
   Track(double energy=-99.99) : fEnergy(energy) {};
   double GetEnergy() { return fEnergy; }
   ClassDef(Track,1);
};

#if VERSION==1
class TopLevel {
   TClonesArray  fTracks;
   TClonesArray *fTracksPtr;
public:
   TopLevel() : fTracks("Track"),fTracksPtr(0) {};
   ~TopLevel() { delete fTracksPtr; }
   void AddTrack(int seed) { 
      if (fTracksPtr==0) fTracksPtr = new TClonesArray("Track");
      new (fTracks[fTracks.GetEntries()]) Track(seed); 
      new ((*fTracksPtr)[fTracksPtr->GetEntries()]) Track(seed); 
   }
   const Track &GetTrack(int which) { return *(Track*)fTracks.At(which); }

   ClassDef(TopLevel,VERSION);
};

#elif VERSION==2

class TopLevel {
   vector<Track>  fTracks;
   vector<Track> *fTracksPtr;
public:
   TopLevel() : fTracksPtr(0) {};
   ~TopLevel() { delete fTracksPtr; }
   void AddTrack(int seed) {
      if (fTracksPtr==0) fTracksPtr = new vector<Track>;
      Track t(seed); fTracks.push_back(t); 
      fTracksPtr->push_back(t);
   }
   const Track &GetTrack(int which) { return fTracks[which]; }

   ClassDef(TopLevel,VERSION);
};

#elif VERSION==3

class TopLevel {
   list<Track>  fTracks;
   list<Track> *fTracksPtr;
public:
   TopLevel() : fTracksPtr(0) {};
   ~TopLevel() { delete fTracksPtr; }
   void AddTrack(int seed) { 
      if (fTracksPtr==0) fTracksPtr = new list<Track>;
      Track t(seed); fTracks.push_back(t); 
      fTracksPtr->push_back(t);
   }
   const Track &GetTrack(int which) { 
      list<Track>::iterator iter = fTracks.begin();
      for(int i=0;i<which && iter!=fTracks.end();++i,++iter) {}
      return *iter; 
   }

   ClassDef(TopLevel,VERSION);
};

#endif

#ifdef __MAKECINT__
#pragma link C++ class Track+;
#pragma link C++ class TopLevel+;
#endif
