#ifndef PART1_H
#define PART1_H

#include <string>

struct Marker {
   const char *fMsg;
   Marker(const char *msg) : fMsg(msg) { fprintf(stderr,"Creating Marker %s\n",msg); }
   ~Marker() { fprintf(stderr,"Destroying Marker %s\n",fMsg); }
};

namespace IncidentType
{
   const Marker m1 = "from first.1 part";
   const Marker m2 = "from first.2 part";
   const Marker m3 = "from first.3 part";
}

#endif //GAUDI_INCIDENT_H

