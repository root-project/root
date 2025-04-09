#ifndef RTCALIB_HH
#define RTCALIB_HH

#include "TNamed.h"
#include <map>
#include <string>
using namespace std;

class T0Result;
class AthIndex;

class RTCalib : public TNamed {
public:
   RTCalib() : TNamed("calib","title") {};
   virtual ~RTCalib() {};

private:

   map<string,T0Result*> _t0res;

  ClassDef(RTCalib,1) // container and calculator of the RT calibrations
};
#ifdef __MAKECINT__
#pragma link C++ class pair<string,T0Result*>+;
#pragma link C++ class map<string,T0Result*>+;
#endif
#endif
