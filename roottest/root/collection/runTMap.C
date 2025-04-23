// Test object (key,value) ownership behaviour of TMap.

#if !defined(__CINT__) || defined(__MAKECINT__)
#include "TObjString.h"
#include "TMap.h"
#include "TObjectSpy.h"
#endif


TMap* tmap_mkmap(TObjString*& key, TObjString*& value)
{
   TMap* map = new TMap;
   key   = new TObjString("Key");
   value = new TObjString("Value");
   map->Add(key, value);
   return map;
}

void tmap_test(Bool_t ownk, Bool_t ownv)
{
   TMap       *map;
   TObjString *key, *value;

   map = tmap_mkmap(key, value);
   map->SetOwnerKeyValue(ownk, ownv);

   TObjectSpy kspy(key), vspy(value);

   delete map;

   printf("Own key=%-5s - key object %-8s || Own value=%-5s - value object %-8s\n",
          ownk ? "true" : "false", kspy.GetObject() ? "survived" : "deleted",
          ownv ? "true" : "false", vspy.GetObject() ? "survived" : "deleted");
}

void runTMap()
{
   tmap_test(kFALSE, kFALSE);
   tmap_test(kFALSE, kTRUE);
   tmap_test(kTRUE,  kFALSE);
   tmap_test(kTRUE,  kTRUE);
}
