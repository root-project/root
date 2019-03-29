#include <string>
#include <sstream>
#include <vector>


//===========================================================================
class MyTemplatedMethodClass {         // template methods
public:
    long get_size();      // to get around bug in genreflex
    template<class B> long get_size();

    long get_char_size();
    long get_int_size();
    long get_long_size();
    long get_float_size();
    long get_double_size();

    long get_self_size();

private:
    double m_data[3];
};

template<class B>
inline long MyTemplatedMethodClass::get_size() {
    return sizeof(B);
}

// 
typedef MyTemplatedMethodClass MyTMCTypedef_t;

// explicit instantiation
template long MyTemplatedMethodClass::get_size<char>();
template long MyTemplatedMethodClass::get_size<int>();

// "lying" specialization
template<>
inline long MyTemplatedMethodClass::get_size<long>() {
    return 42;
}


//===========================================================================
// global templated functions
template<typename T>
long global_get_size() {
    return sizeof(T);
}

template <typename T>
int global_some_foo(T) {
    return 42;
}

template <typename T>
int global_some_bar(T) {
    return 13;
}

template <typename F>
struct SomeResult {
    F m_retval;
};

template <class I, typename O = float>
SomeResult<O> global_get_some_result(const std::vector<I>& carrier) {
    SomeResult<O> r{};
    r.m_retval = O(carrier[0]);
    return r;
}


//===========================================================================
// variadic functions
inline bool isSomeInt(int) { return true; }
inline bool isSomeInt(double) { return false; }
template <typename ...Args>
inline bool isSomeInt(Args...) { return false; }

namespace AttrTesting {

struct Obj1 { int var1; };
struct Obj2 { int var2; };

template <typename T>
constexpr auto has_var1(T t) -> decltype(t.var1, true) { return true; }

template <typename ...Args>
constexpr bool has_var1(Args...) { return false; }

template <typename T>
constexpr bool call_has_var1(T&& t) { return AttrTesting::has_var1(std::forward<T>(t)); }

template <int N, typename... T>
struct select_template_arg {};

template <typename T0, typename... T>
struct select_template_arg<0, T0, T...> {
    typedef T0 type;
};

template <int N, typename T0, typename... T>
struct select_template_arg<N, T0, T...> {
    typedef typename select_template_arg<N-1, T...>::type argument;
};

} // AttrTesting


namespace SomeNS {

template <typename T>
int some_foo(T) {
    return 42;
}

template <int T>
int some_bar() {
    return T;
}

inline std::string tuplify(std::ostringstream& out) {
    out << "NULL)";
    return out.str();
}

template<typename T, typename... Args>
std::string tuplify(std::ostringstream& out, T value, Args... args)
{
    out << value << ", ";
    return tuplify(out, args...);
}

} // namespace SomeNS


//===========================================================================
// using of static data
// TODO: this should live here instead of in test_templates.test08
/*
template <typename T> struct BaseClassWithStatic {
    static T const ref_value;
};

template <typename T>
T const BaseClassWithStatic<T>::ref_value = 42;

template <typename T>
struct DerivedClassUsingStatic : public BaseClassWithStatic<T> {
    using BaseClassWithStatic<T>::ref_value;

    explicit DerivedClassUsingStatic(T x) : BaseClassWithStatic<T>() {
        m_value = x > ref_value ? ref_value : x;
    }

    T m_value;
};
*/


//===========================================================================
// templated callable
class TemplatedCallable {
public:
    template <class I , class O = double>
    O operator() (const I& in) const { return O(in); }
};


//===========================================================================
// templated typedefs
namespace TemplatedTypedefs {

template<typename TYPE_IN, typename TYPE_OUT, size_t _vsize = 4>
struct BaseWithEnumAndTypedefs {
    enum { vsize = _vsize };
    typedef TYPE_IN in_type;
    typedef TYPE_OUT out_type;
};

template <typename TYPE_IN, typename TYPE_OUT, size_t _vsize = 4>
struct DerivedWithUsing : public BaseWithEnumAndTypedefs<TYPE_IN, TYPE_OUT, _vsize>
{
    typedef BaseWithEnumAndTypedefs<TYPE_IN, TYPE_OUT, _vsize> base_type;
    using base_type::vsize;
    using typename base_type::in_type;
    typedef typename base_type::in_type in_type_tt;
    using typename base_type::out_type;
};

struct SomeDummy {};

} // namespace TemplatedTypedefs


//===========================================================================
// hiding templated methods
namespace TemplateHiding {

struct Base {
    template<class T>
    int callme(T t = T(1)) { return 2*t; }
};

struct Derived : public Base {
    int callme(int t = 2) { return t; }
};

} // namespace TemplateHiding


//===========================================================================
// 'using' of templates
template<typename T> using DA_vector = std::vector<T>;

#if __cplusplus > 201402L
namespace using_problem {

template <typename T, size_t SZ>
struct vector {
    vector() : m_val(SZ) {}
    T m_val;
};

template <typename T, size_t ... sizes>
struct matryoshka {
    typedef T type;
};

template <typename T, size_t SZ, size_t ... sizes>
struct matryoshka<T, SZ, sizes ... > {
    typedef vector<typename matryoshka<T, sizes ...>::type, SZ> type;
};

template <typename T, size_t ... sizes>
using make_vector = typename matryoshka<T, sizes ...>::type;
    typedef make_vector<int, 2, 3> iiv_t;
};
#endif

namespace using_problem {

template<typename T>
class Base {
public:
    template<typename R>
    R get1(T t) { return t + R{5}; }
    T get2() { return T{5}; }
    template<typename R>
    R get3(T t) { return t + R{5}; }
    T get3() { return T{5}; }
};

template<typename T>
class Derived : public Base<T> {
public:
    typedef Base<T> _Mybase;
    using _Mybase::get1;
    using _Mybase::get2;
    using _Mybase::get3;
};

}


//===========================================================================
// template with r-value
namespace T_WithRValue {

template<typename T>
bool is_valid(T&& new_value) {
    return new_value != T{};
}

}
