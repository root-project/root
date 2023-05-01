#ifndef VERSION
#error You need to explicitly specify a version
#endif

#include "TClonesArray.h"
#include <vector>
#include <list>

// #define WITHCLASSDEF

namespace std {} using namespace std;

#ifndef TRACK
#define TRACK
class Track : public TObject {
   double fEnergy;
public:
   Track(double energy=-99.99) : fEnergy(energy) {}
   double GetEnergy() { return fEnergy; }
   ClassDefOverride(Track,1);
};
#endif

#if VERSION==1
class TopLevel {
   TClonesArray  fTracks;
   TClonesArray *fTracksPtr;
public:
   TopLevel() : fTracks("Track"),fTracksPtr(0) {};
   virtual ~TopLevel() { delete fTracksPtr; }
   void AddTrack(int seed) { 
      if (fTracksPtr==0) fTracksPtr = new TClonesArray("Track");
      new (fTracks[fTracks.GetEntries()]) Track(seed); 
      new ((*fTracksPtr)[fTracksPtr->GetEntries()]) Track(seed); 
   }
   const Track &GetTrack(int which) { return *(Track*)fTracks.At(which); }

#ifdef WITHCLASSDEF
   ClassDef(TopLevel,VERSION);
#endif
};

#elif VERSION==2

class TopLevel {
   vector<Track>  fTracks;
   vector<Track> *fTracksPtr;
public:
   TopLevel() : fTracksPtr(0) {};
#ifdef ClingWorkAroundJITandInline
   virtual ~TopLevel();
   void AddTrack(int seed);
#else
   virtual ~TopLevel() { delete fTracksPtr; }
   void AddTrack(int seed) {
      if (fTracksPtr==0) fTracksPtr = new vector<Track>;
      Track t(seed); fTracks.push_back(t); 
      fTracksPtr->push_back(t);
   }
#endif
   const Track &GetTrack(int which) { return fTracks[which]; }

#ifdef WITHCLASSDEF
   ClassDef(TopLevel,VERSION);
#endif
};

#ifdef ClingWorkAroundJITandInline
inline TopLevel::~TopLevel() { delete fTracksPtr; }
void TopLevel::AddTrack(int seed) {
   if (fTracksPtr==0) fTracksPtr = new vector<Track>;
   Track t(seed); fTracks.push_back(t); 
   fTracksPtr->push_back(t);
}
#endif

#elif VERSION==3

class TopLevel {
   list<Track>  fTracks;
   list<Track> *fTracksPtr;
public:
   TopLevel() : fTracksPtr(0) {};
   virtual ~TopLevel() { delete fTracksPtr; }
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

#ifdef WITHCLASSDEF
   ClassDef(TopLevel,VERSION);
#endif
};

#endif

#undef TopLevel
#ifdef __MAKECINT__
#ifdef WITHCLASSDEF
#pragma link C++ class TopLevelCl+;
#else
#pragma link C++ class Track+;
#pragma link C++ class TopLevel+;
#endif
#endif
