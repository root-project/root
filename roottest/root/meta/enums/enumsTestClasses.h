#ifndef __EcalSeverityLevel_h_
#define __EcalSeverityLevel_h_

class testClass{
public:
   enum testEnum1 {
      kLow=0,
      kMedium,
      kHigh
   };
   enum testEnum3 {
      kLow3=0,
      kMedium3,
      kHigh3
   };
};

enum testEnum2 {
   kLow=0,
   kMedium,
   kHigh
};


namespace EcalSeverityLevel {

  enum SeverityLevel {
    kGood=0,             // good channel 
    kProblematic,        // problematic (e.g. noisy)
    kRecovered,          // recovered (e.g. an originally dead or saturated)
    kTime,               // the channel is out of time (e.g. spike)
    kWeird,              // weird (e.g. spike)
    kBad                 // bad, not suitable to be used for reconstruction
  };
      
}

#endif // __EcalSeverityLevel_h_
