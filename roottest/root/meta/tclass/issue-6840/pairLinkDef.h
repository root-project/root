#include <vector>
#include <list>
#include <map>
class CaloTowerTest;

#ifdef __ROOTCLING__
#pragma link C++ class vector<CaloTowerTest*>;
#pragma link C++ class list<CaloTowerTest*>;

#pragma link C++ class map<short, SameAsShort>+;
#pragma link C++ class map<SameAsShort, SameAsShort>+;
#pragma link C++ class map<short, short>;

#pragma link C++ class SameAsShort+;
#pragma link C++ class Contains+;

// #pragma link C++ class pair<short, SameAsShort>+;
#pragma link C++ class pair<SameAsShort, SameAsShort>+;


// Remove later
// #pragma link C++ class pair<unsigned char, SameAsShort>+;
#endif
