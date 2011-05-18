class entry;
class concrete {
  const entry* operator()(int x);
  const entry& operator()(short x);
  const entry*& operator()(char x);
};


namespace Foam
{

class word;
class entry;

template<class T, class Key=word, class Hash=int>
class HashTable
{
public:

    T& operator()(const Key& key);
    const T& operator()(const Key& key) const;
};

typedef HashTable<entry*> HashTableEntry;

}


#ifdef __MAKECINT__
#pragma link off all class;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;
#pragma link C++ class Foam::HashTableEntry;
#endif

int main() {
  Foam::HashTableEntry e;
  concrete c;
  return 0;
}
