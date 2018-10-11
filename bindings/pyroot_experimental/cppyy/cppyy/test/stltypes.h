#include <list>
#include <map>
#include <string>
#include <utility>
#include <vector>

//- basic example class
class just_a_class {
public:
    int m_i;
};

//- class with lots of std::string handling
class stringy_class {
public:
   stringy_class(const char* s);

   std::string get_string1();
   void get_string2(std::string& s);

   void set_string1(const std::string& s);
   void set_string2(std::string s);

   std::string m_string;
};

//- class that has an STL-like interface
class no_dict_available;
    
template<class T>
class stl_like_class {
public: 
   no_dict_available* begin() { return 0; }
   no_dict_available* end() { return 0; }
   int size() { return 4; }
   int operator[](int i) { return i; }
   std::string operator[](double) { return "double"; }
   std::string operator[](const std::string&) { return "string"; }
};      


//- instantiations of used STL types
namespace {

    stl_like_class<int> stlc_1;

} // unnamed namespace


// comps for int only to allow testing: normal use of vector is looping over a
// range-checked version of __getitem__
#if defined(__clang__) && defined(__APPLE__)
namespace std {
#define ns_prefix std::
#elif defined(__GNUC__) || defined(__GNUG__)
namespace __gnu_cxx {
#define ns_prefix
#endif
extern template bool ns_prefix operator==(const std::vector<int>::iterator&,
                         const std::vector<int>::iterator&);
extern template bool ns_prefix operator!=(const std::vector<int>::iterator&,
                         const std::vector<int>::iterator&);
}


//- helpers for testing array
namespace ArrayTest {

struct Point {
    int px, py;
};

int get_pp_px(Point** p, int idx);
int get_pp_py(Point** p, int idx);
int get_pa_px(Point* p[], int idx);
int get_pa_py(Point* p[], int idx);

} // namespace ArrayTest
