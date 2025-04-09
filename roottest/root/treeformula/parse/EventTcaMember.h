
#ifndef _EVENT_TCA_MEMBER_H
#define _EVENT_TCA_MEMBER_H

#include <TObject.h>
#include <TClonesArray.h>

class Track : public TObject
{
public:
   Double_t p;

   Track(Double_t _p = 0.) : p(_p) {}
   ~Track(void) override {}
   ClassDefOverride(Track, 1)
};

class EventTcaMember : public TObject
{
public:
   TClonesArray tca;
   EventTcaMember() : tca() {}
   EventTcaMember(Int_t n) : tca("Track", n) {}
   ~EventTcaMember (void) override {}
   ClassDefOverride(EventTcaMember, 1)
};

#ifdef __MAKECINT__
#pragma link C++ class Track+;
#pragma link C++ class EventTcaMember+;
#endif // __MAKECINT__
   
#endif // ! _EVENT_TCA_MEMBER_H

