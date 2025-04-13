#include <string>
#include <vector>


//===========================================================================
class OverloadA {
public:
    OverloadA();
    int i1, i2;
};

namespace NamespaceA {
    class OverloadA {
    public:
        OverloadA();
        int i1, i2;
    };

    class OverloadB {
    public:
        int f(const std::vector<int>* v);
    };
}

namespace NamespaceB {
    class OverloadA {
    public:
        OverloadA();
        int i1, i2;
    };
}

class OverloadB {
public:
    OverloadB();
    int i1, i2;
};


//===========================================================================
class OverloadC {
public:
    OverloadC();
    int get_int(OverloadA* a);
    int get_int(NamespaceA::OverloadA* a);
    int get_int(NamespaceB::OverloadA* a);
    int get_int(short* p);
    int get_int(OverloadB* b);
    int get_int(int* p);
};


//===========================================================================
class OverloadD {
public:
    OverloadD();
//   int get_int(void* p) { return *(int*)p; }
    int get_int(int* p);
    int get_int(OverloadB* b);
    int get_int(short* p);
    int get_int(NamespaceB::OverloadA* a);
    int get_int(NamespaceA::OverloadA* a);
    int get_int(OverloadA* a);
};


//===========================================================================
class OlAA {};
class OlBB;
struct OlCC {};
struct OlDD;

OlBB* get_OlBB();
OlDD* get_OlDD();


//===========================================================================
class MoreOverloads {
public:
    MoreOverloads();
    std::string call(const OlAA&);
    std::string call(const OlBB&, void* n=0);
    std::string call(const OlCC&);
    std::string call(const OlDD&);

    std::string call_unknown(const OlDD&);

    std::string call(double);
    std::string call(int);
    std::string call1(int);
    std::string call1(double);
};


//===========================================================================
class MoreOverloads2 {
public:
    MoreOverloads2();
    std::string call(const OlBB&);
    std::string call(const OlBB*);

    std::string call(const OlDD*, int);
    std::string call(const OlDD&, int);
};


//===========================================================================
class MoreBuiltinOverloads {
public:
    std::string method(int arg);
    std::string method(double arg);
    std::string method(bool arg);
    std::string method2(int arg);
    std::string method3(bool arg);

    std::string method4(int, double);
    std::string method4(int, char);
};


//===========================================================================
std::string global_builtin_overload(int, double);
std::string global_builtin_overload(int, char);


//===========================================================================
double calc_mean(long n, const float* a);
double calc_mean(long n, const double* a);
double calc_mean(long n, const int* a);
double calc_mean(long n, const short* a);
double calc_mean(long n, const long* a);

template<typename T>
double calc_mean_templ(long n, const T* a) {
    double sum = 0., sumw = 0.;
    const T* end = a+n;
    while (a != end) {
        sum += *a++;
        sumw += 1;
    }

    return sum/sumw;
}

template double calc_mean_templ<float> (long, const float*);
template double calc_mean_templ<double>(long, const double*);
template double calc_mean_templ<int>   (long, const int*);
template double calc_mean_templ<short> (long, const short*);
template double calc_mean_templ<long>  (long, const long*);
