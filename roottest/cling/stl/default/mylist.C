#include <list>
using namespace std;
class BaseStar {
//...
};

template<class Star> class StarList : public list <Star*> {
//...
};

typedef list<BaseStar*> blist;

#ifdef __CINT__
#pragma link C++ nestedclass;
#pragma link C++ nestedtypedef;
#pragma link C++ class BaseStar;
#pragma link C++ class StarList<BaseStar>;
#pragma link C++ class list<BaseStar*>;
typedef list<BaseStar*> blist;
#pragma link C++ class blist::iterator;
#pragma link C++ class list<BaseStar*>::iterator;
#endif


