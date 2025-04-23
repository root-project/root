#include <map>

using namespace std;
template <class ElementType> 
class StoragePolicy 
{
public:
  typedef map<char,double> map_type;

  map_type map_;
};

#ifdef __CINT__

//#pragma link C++ class StoragePolicy+;
#pragma link C++ class StoragePolicy<int>+;

#endif


