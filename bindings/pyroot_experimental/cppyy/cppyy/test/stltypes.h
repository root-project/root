#include <list>
#include <map>
#include <string>
#include <sstream>
#if __cplusplus > 201402L
#include <string_view>
#endif
#include <utility>
#include <vector>


//- basic example class
class just_a_class {
public:
    int m_i;
};

// enum for vector of enums setitem tests
enum VecTestEnum {
    EVal1 = 1, EVal2 = 3
};

namespace VecTestEnumNS {
    enum VecTestEnum { EVal1 = 5, EVal2 = 42 };
}


//- adverse effect of implicit conversion on vector<string>
int vectest_ol1(const std::vector<std::string>&);
int vectest_ol1(std::string);
int vectest_ol2(std::string);
int vectest_ol2(const std::vector<std::string>&);


//- class with lots of std::[w]string handling
template<typename S>
class stringy_class {
public:
    stringy_class(const typename S::value_type* s) : m_string(s) {}

    S get_string1() { return m_string; }
    void get_string2(S& s) { s = m_string; }

    void set_string1(const S& s) { m_string = s; }
    void set_string2(S s) { m_string = s; }

    S m_string;
};

typedef stringy_class<std::string> stringy_class_t;
typedef stringy_class<std::wstring> wstringy_class_t;


//- class that has an STL-like interface
class no_dict_available;
    
template<class T>
class stl_like_class {
public: 
    no_dict_available* begin() { return 0; }
    no_dict_available* end() { return (no_dict_available*)1; }
    int size() { return 4; }
    int operator[](int i) { return i; }
    std::string operator[](double) { return "double"; }
    std::string operator[](const std::string&) { return "string"; }
};      

namespace {
    stl_like_class<int> stlc_1;
}


//- similar, but now the iterators don't work b/c they don't compile
template<class value_type, size_t sz>
class stl_like_class2 {
protected:
    value_type fData[sz];

public:
    static const size_t size() { return sz; }
    value_type& operator[](ptrdiff_t i) { return fData[i]; }
};

template<class value_type, size_t sz>
class stl_like_class3 : public stl_like_class2<value_type, sz> {
    using stl_like_class2<value_type, sz>::fData;
public:
    size_t size() { return sz; }
    value_type& begin() { return fData; }
    value_type& end() { return fData + sz; }
};


//- helpers for testing array
namespace ArrayTest {

struct Point {
    Point() : px(0), py(0) {}
    Point(int x, int y) : px(x), py(y) {}
    int px, py;
};

int get_pp_px(Point** p, int idx);
int get_pp_py(Point** p, int idx);
int get_pa_px(Point* p[], int idx);
int get_pa_py(Point* p[], int idx);

} // namespace ArrayTest


// helpers for string testing
extern std::string str_array_1[3];
extern std::string str_array_2[];
extern std::string str_array_3[3][2];
extern std::string str_array_4[4][2][2];


// helpers for string_view testing
#if __cplusplus > 201402L
namespace StringViewTest {
    std::string_view::size_type count(const std::string_view arg);
    std::string_view::size_type count_cr(const std::string_view& arg);
}
#endif
