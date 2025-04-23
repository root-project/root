#include <map>

#include <vector>


template <typename Anchor>
struct Simple {};

template <class Anchor, typename ... T>
struct Packing {};

typedef std::map<unsigned int, std::vector<const void*> > AttrListCVMap;

 

void makeIndex()
{
  std::pair<AttrListCVMap::iterator,bool> res;
}

