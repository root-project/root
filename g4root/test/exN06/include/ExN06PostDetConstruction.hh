#ifndef ExN06PostDetConstruction_h
#define ExN06PostDetConstruction_h 1

// Example of post detector construction user class

#include "TG4RootDetectorConstruction.h"

class TPolyLine3D;

class ExN06PostDetConstruction : public TVirtualUserPostDetConstruction
{
private:
   TObjArray            *fTracks;  // Array of tracks
   TPolyLine3D          *fCurrent; // Current track
   
   ExN06PostDetConstruction();
   static ExN06PostDetConstruction *fgInstance; // Self pointer
public:
   virtual ~ExN06PostDetConstruction();

   static ExN06PostDetConstruction *GetInstance();

   void                  NewTrack(Double_t x, Double_t y, Double_t z);
   void                  AddPoint(Double_t x, Double_t y, Double_t z);
   void                  WriteTracks(const char *filename);
   
   virtual void          Initialize(TG4RootDetectorConstruction *dc);
};
#endif
   
   

