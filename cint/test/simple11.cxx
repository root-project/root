#define G__NOCINTDLL

#include <vector>
#include <string>

namespace std {}; using namespace std;

vector<void*> z;
vector<string> x;
allocator<int> y;

#ifdef __MAKECINT__
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ function vector<void*,allocator<void*> >::iterator::iterator;
#pragma link C++ function allocator<int>::address;
#pragma link C++ function vector<string,allocator<string> >::const_iterator::operator*;
#endif

int main() 
{
   return 0;
}
