#include <string>
#include <sstream>


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
    out.seekp(-2, out.cur); out << ')';
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
