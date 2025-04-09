#include <list>
#include <vector>

template <typename T>
class CustomAlloc : public std::allocator<T>
{
public:
   template <typename U>
   struct rebind {
      using other = CustomAlloc<U>;
   };
};

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
  std::vector<int, CustomAlloc<int>> fCustomVec;
};

void execMissing()
{
   TClass *c = TClass::GetClass("Event");
   c->GetStreamerInfo()->ls();
};

