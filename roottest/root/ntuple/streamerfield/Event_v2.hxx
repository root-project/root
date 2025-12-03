#ifndef EVENT_V2_H
#define EVENT_V2_H

#include <RtypesCore.h>

#include <memory>

struct StreamerBase {
   int fBase = 0;
   virtual ~StreamerBase() = default;

   ClassDef(StreamerBase, 2)
};

struct StreamerDerived : public StreamerBase {
   int fFirst = 1;
   int fSecond = 2;
   virtual ~StreamerDerived() = default;

   ClassDefOverride(StreamerDerived, 2)
};

struct StreamerContainer {
   std::unique_ptr<StreamerBase> fPtr;

   ClassDefNV(StreamerContainer, 2)
};

struct Event {
   int fX = 137;
   StreamerContainer fField;

   ClassDefNV(Event, 2)
};

#endif // EVENT_V2_H
