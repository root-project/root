#define var(x) int i##x; float f##x
#define udef(x) i##x(0),f##x(0.0)
#define def(x) i##x(x),f##x(x/3.0)

#include "TNamed.h"
#include "TVirtualStreamerInfo.h"
 
class THit {
protected:
  float     fX;         //x position at center
  float     fY;         //y position at center
  float     fZ;         //z position at center
  int       fNpulses;   //Number of pulses
  int      *fPulses;    //[fNpulses]
  int       fTime[10];  //time at the 10 layers
public:

  THit();
  THit(const THit &);
  THit(int time);
  virtual ~THit();

  void  Set (int time);
  inline int Get(int i) { return fTime[i]; }
  bool operator==(const THit& c) const { return this==&c;}
  bool operator<(const THit& c) const { return this<&c;}
  THit& operator=(const THit& c);
  friend TBuffer &operator<<(TBuffer &b, const THit *hit);

  ClassDef(THit,1) // the hit class
};

THit::THit() {
   fPulses = 0;
   fNpulses = 0;
}
THit::THit(const THit &hit) {
   fX = hit.fX;
   fY = hit.fY;
   fZ = hit.fZ;
   for (Int_t i=0;i<10;i++) fTime[i] = hit.fTime[i];
   fPulses = 0;
   fNpulses = hit.fNpulses;
   if (fNpulses == 0) return;
   if (hit.fPulses == 0) return;
   fPulses = new int[fNpulses];
   for (int j=0;j<fNpulses;j++) fPulses[j] = hit.fPulses[j];
}

THit& THit::operator=(const THit& hit)  {
   fX = hit.fX;
   fY = hit.fY;
   fZ = hit.fZ;
   for (Int_t i=0;i<10;i++) fTime[i] = hit.fTime[i];
   fPulses = 0;
   fNpulses = hit.fNpulses;
   if (fNpulses == 0) return *this;
   if (hit.fPulses == 0) return *this;
   if ( fPulses ) delete [] fPulses;
   fPulses = new int[fNpulses];
   for (int j=0;j<fNpulses;j++) fPulses[j] = hit.fPulses[j];
   return *this;
}

THit::THit(int t) {
   fPulses = 0;
   Set(t);
}

THit::~THit() {
   if (fPulses) delete [] fPulses;
   fPulses = 0;
}

void THit::Set(int t) {
   fX = 0.12;
   fY = 0.21;
   fZ = 0.33;
   if (fPulses && fNpulses > 0) delete [] fPulses;
   fNpulses = t%20 + 1;
   fPulses = new int[fNpulses];
   for (int j=0;j<fNpulses;j++) fPulses[j] = j+1;
   for (int i=0; i<10; i++) fTime[i] = t+i;
}

TBuffer &operator<<(TBuffer &buf, const THit *obj)
{
   ((THit*)obj)->Streamer(buf);
   return buf;
}

class Holder {
public:
   vector<THit> fVector;
   ClassDef(Holder,2);
};

#include "TBufferFile.h"
#include "TClass.h"
#include <vector> 
#ifdef __MAKECINT__
#pragma link C++ class vector<THit>+;
#endif

void write(TBuffer &buf,int ntimes, int nelems) {
   Holder holder;
   THit hit;
   for(int e=0; e<nelems; ++e) {
     hit.Set(e);
     holder.fVector.push_back(hit);
   }
   buf.SetWriteMode();
   for(int i=0; i<ntimes; ++i) {
      buf.Reset();
      holder.Streamer(buf);
   }
}

void read(TBuffer &buf,int ntimes) {
   Holder holder;
   buf.SetReadMode();
   for(int i=0; i<ntimes; ++i) {
      buf.Reset();
      holder.Streamer(buf);
      // obj->IsA()->Dump(obj);
      // delete obj;
   }
}


void vectorMWthit(int nread = 2, int nelems = 600) {
   TVirtualStreamerInfo::SetStreamMemberWise(kTRUE);
   TBufferFile buf(TBuffer::kWrite);
   write(buf,1, nelems);
   read(buf,nread);
}
