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
   TClonesArray fTracks;
public:
   TopLevel() : fTracks("Track") {};
   void AddTrack(int seed) { new (fTracks[fTracks.GetEntries()]) Track(seed); }
   const Track &GetTrack(int which) { return *(Track*)fTracks.At(which); }

   ClassDef(TopLevel,VERSION);
};

#elif VERSION==2

class TopLevel {
   vector<Track> fTracks;
public:
   TopLevel() {};
   void AddTrack(int seed) { Track t(seed); fTracks.push_back(t); }
   const Track &GetTrack(int which) { return fTracks[which]; }

   ClassDef(TopLevel,VERSION);
};

#elif VERSION==3

class TopLevel {
   list<Track> fTracks;
public:
   TopLevel() {};
   void AddTrack(int seed) { Track t(seed); fTracks.push_back(t); }
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
