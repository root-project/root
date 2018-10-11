#include "stltypes.h"


//- explicit instantiations of used comparisons
#if defined __clang__
namespace std {
#define ns_prefix std::
#elif defined(__GNUC__) || defined(__GNUG__)
namespace __gnu_cxx {
#define ns_prefix
#endif
template bool ns_prefix operator==(const std::vector<int>::iterator&,
                         const std::vector<int>::iterator&);
template bool ns_prefix operator!=(const std::vector<int>::iterator&,
                         const std::vector<int>::iterator&);
}

//- class with lots of std::string handling
stringy_class::stringy_class(const char* s) : m_string(s) {}

std::string stringy_class::get_string1() { return m_string; }
void stringy_class::get_string2(std::string& s) { s = m_string; }

void stringy_class::set_string1(const std::string& s) { m_string = s; }
void stringy_class::set_string2(std::string s) { m_string = s; }


//- helpers for testing array
int ArrayTest::get_pp_px(Point** p, int idx) {
    return p[idx]->px;
}

int ArrayTest::get_pp_py(Point** p, int idx) {
    return p[idx]->py;
}

int ArrayTest::get_pa_px(Point* p[], int idx) {
    return p[idx]->px;
}

int ArrayTest::get_pa_py(Point* p[], int idx) {
    return p[idx]->py;
}
