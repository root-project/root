#ifndef classes_h
#define classes_h

#include "TObject.h"
#include "marker.h"
#include <vector>

class TopLevel
{
public:
   Int_t fTopLevel;
   Marker fMarker;
   TopLevel() : fMarker(Class_Name()) {}
   virtual ~TopLevel() {
      // if (Marker::fgDebug==2) fprintf(stdout, "TopLevel destructor for 0x%lx\n", (long)this);
   }

   ClassDef(TopLevel,2);
};

class MidLevel : public TopLevel
{
public:
   Int_t fMidLevel;
   Marker fMarker;
   MidLevel() : fMarker(Class_Name()) {}
   ClassDef(MidLevel,2);
};

class TObjTopLevel : public TObject
{
public:
   Double_t fTObjTopLevel;
   Marker fMarker;
   TObjTopLevel() : fMarker(Class_Name()) {}
   ClassDef(TObjTopLevel,2);
};

class Bottom : public MidLevel
{
public:
   TObjTopLevel fObject;
   Marker fMarker;
   Bottom() : fMarker(Class_Name()) {}
   ClassDef(Bottom,2);
};

class TObjBottom : public TObject, public MidLevel
{
public:
   Marker fMarker;
   TObjBottom() : fMarker(Class_Name()) {}
   ClassDef(TObjBottom,2);
};

class Side
{
public:
   Marker fMarker;
   Side() : fMarker(Class_Name()) {}
   virtual ~Side() {
      // if (Marker::fgDebug==2) fprintf(stdout, "Side destructor for 0x%lx\n",(long)this);
   }
   ClassDef(Side,2);
};

class BottomDouble : public MidLevel, public Side
{
public:
   Marker fMarker;
   BottomDouble() : fMarker(Class_Name()) {}
   ClassDef(BottomDouble,2);
};

class TObjFirst : public TObject, public TopLevel
{
public:
   Marker fMarker;
   TObjFirst() : fMarker(Class_Name()) {}
   ClassDef(TObjFirst,2);
};

class TObjSecond : public TopLevel, public TObject
{
public:
   Marker fMarker;
   TObjSecond() : fMarker(Class_Name()) {}
   ClassDef(TObjSecond,2);
};

class Holder {
public:
   TopLevel *fMidLevel;
   TObject  *fTObjTopLevel;
   TopLevel *fBottom;
   TopLevel *fBottomDoubleTop;
   Side     *fBottomDoubleSide;
   TopLevel *fTObjFirst;
   TopLevel *fTObjSecond;
   vector<TopLevel*> fVec;
   Marker    *fMarker;

   void Init() {
      fMidLevel = new MidLevel();
      fTObjTopLevel = new TObjTopLevel();
      fBottomDoubleTop = new BottomDouble();
      fBottom = new Bottom();
      fBottomDoubleSide = new BottomDouble();
      fTObjFirst = new TObjFirst();
      fTObjSecond = new TObjSecond();
      // Fill(fVec);
   }
   void Fill(vector<TopLevel*> &vec) {
      vec.clear();
      vec.push_back(new MidLevel());
      // vec.push_back(new TObjTopLevel());
      vec.push_back(new Bottom());
      vec.push_back(new BottomDouble());
      vec.push_back(new BottomDouble());
      vec.push_back(new TObjFirst());
      vec.push_back(new TObjSecond());
   }
   void Clear(vector<TopLevel*> &vec) {
      vector<TopLevel*>::iterator iter = vec.begin();
      vector<TopLevel*>::iterator end = vec.end();
      for( ; iter != end; ++iter) {
         delete *iter;
      }
   }
   Holder() : fMidLevel(0),fTObjTopLevel(0),fBottom(0),fBottomDoubleTop(0),
              fBottomDoubleSide(0),fTObjFirst(0),fTObjSecond(0),fMarker(new Marker(Class_Name())) {}
   virtual ~Holder() {
      delete fMarker;
      // Clear(fVec);
      delete fTObjSecond;
      delete fTObjFirst;
      delete fBottomDoubleSide;
      delete fBottomDoubleTop;
      delete fBottom;
      delete fTObjTopLevel;
      delete fMidLevel;
   }
   ClassDef(Holder,2);
};

#ifdef __MAKECINT__
#pragma link C++ class vector<TopLevel*>;
#endif

#endif
