#include <vector>
#include <list>

namespace std 
{
   template <class T> class helper {};
   template <class T, class alloc = helper<T> > class myvector {};
   //template <class T> class list {};
}
using namespace std;

class Top {
public:
   class Bottom {};

   void topfunction1(std::list<Bottom*>&) { }
   std::vector<Bottom*> topfunction(std::list<Bottom*>&) { return std::vector<Bottom*>(); }

   class Second {
   public:
      void topfunction1(std::list<Bottom*>&) { }
      std::vector<Bottom*> function(std::list<Bottom*>&) { return std::vector<Bottom*>(); }
   };
};

std::vector<Top* > extfunc1(std::list<Top*>) {};
std::myvector<Top*, helper<Top*> > extfunc2(std::list<Top*>) {};


#ifdef __MAKECINT__
//pragma link off all classes;
//pragma link off all functions;
//pragma link C++ class vector<Top::Bottom*>;
//pragma link C++ class list<Top::Bottom*>;
//pragma link C++ class Top;
//pragma link C++ class Top::Bottom;
//pragma link C++ class Top::Second;
//pragma link C++ function extfunc1;
//pragma link C++ function extfunc2;
#endif
