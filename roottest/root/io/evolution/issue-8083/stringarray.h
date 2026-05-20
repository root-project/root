#pragma once
#include <string>
class mystrarray {
public:
  string         fDataTag;        ///< offset=  0 type=300 ,stl=365, ctype=365, Tag of the reco data products (art::InputTag format)
  string         fOutputInstance; ///< offset= 32 type=300 ,stl=365, ctype=365, Instance name of the feature vector collection
  string         fOutputNames[4]; ///<   offset= 64 type=320 ,stl=365, ctype=365, Feature vector entries names/meaning
  // MUST UPDATE WHEN CLASS IS CHANGED!
  static short Class_Version() { return 10; }
};

class container {
public:
  std::vector<mystrarray> obj;
};

#ifdef __ROOTCLING__
#pragma link C++ class std::vector<mystrarray>+;
#endif
