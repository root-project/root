#include <list>

class Track {
public:
   Track() : fX(0) {}
   float fX;
};

class Event {
public:
  Event() : fPointers(0) {}

  list<Track>   fObjects;
  list<Track*> *fPointers;
};

void execMissing()
{
   TClass *c = TClass::GetClass("Event");
   c->GetStreamerInfo()->ls();
};

