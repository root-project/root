#include "stltypes.h"


//- explicit instantiations of used comparisons
#if defined __clang__ || defined(__GNUC__) || defined(__GNUG__)
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
#endif


//- adverse effect of implicit conversion on vector<string>
int vectest_ol1(const std::vector<std::string>&) { return 1; }
int vectest_ol1(std::string) { return 2; }
int vectest_ol2(std::string) { return 2; }
int vectest_ol2(const std::vector<std::string>&) { return 1; }


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


// helpers for string testing
std::string str_array_1[3] = {"a", "b", "c"};
std::string str_array_2[]  = {"d", "e", "f", "g"};
std::string str_array_3[3][2] = {{"a", "b"}, {"c", "d"}, {"e", "f"}};
std::string str_array_4[4][2][2] = {
     {{"a", "b"}, {"c", "d"}},
     {{"e", "f"}, {"g", "h"}},
     {{"i", "j"}, {"k", "l"}},
     {{"m", "n"}, {"o", "p"}},
};


// helpers for string_view testing
#if __cplusplus > 201402L
std::string_view::size_type StringViewTest::count(const std::string_view arg) {
    return arg.size();
}

std::string_view::size_type StringViewTest::count_cr(const std::string_view& arg) {
    return arg.size();
}
#endif
