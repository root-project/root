#ifndef EVENT_V3_H
#define EVENT_V3_H

#include <RtypesCore.h>

#include <memory>

struct StreamerBase {
   int fBase = 0;
   virtual ~StreamerBase() = default;

   ClassDef(StreamerBase, 2)
};

struct StreamerDerived : public StreamerBase {
   int fSecond = 2;
   int fFirst = 1;
   virtual ~StreamerDerived() = default;

   ClassDefOverride(StreamerDerived, 3)
};

struct StreamerContainer {
   std::unique_ptr<StreamerBase> fPtr;

   ClassDefNV(StreamerContainer, 2)
};

struct Event {
   StreamerContainer fField;
   int fY = 42;

   ClassDefNV(Event, 3)
};

#endif // EVENT_V3_H
