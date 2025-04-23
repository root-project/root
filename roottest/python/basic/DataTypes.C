#include "DataTypes.h"


//===========================================================================
std::vector<EFruit> vecFruits{kCitrus, kApple};


//===========================================================================
CppyyTestData::CppyyTestData() : m_owns_arrays(false)
{
    m_bool    = false;
    m_char    = 'a';
    m_schar   = 'b';
    m_uchar   = 'c';
    m_short   = -11;
    m_ushort  =  11u;
    m_int     = -22;
    m_uint    =  22u;
    m_long    = -33l;
    m_ulong   =  33ul;
    m_llong   = -44ll;
    m_ullong  =  44ull;
    m_long64  = -55ll;
    m_ulong64 =  55ull;
    m_float   = -66.f;
    m_float16   = 16.f;
    m_double  = -77.;
    m_double32  = 32.;
    m_ldouble = -88.l;
    m_enum    = kNothing;
    m_voidp   = (void*)0;

    m_bool_array2   = new bool[N];
    m_char_array2   = new signed char[N];
    m_uchar_array2  = new unsigned char[N];
    m_short_array2  = new short[N];
    m_ushort_array2 = new unsigned short[N];
    m_int_array2    = new int[N];
    m_uint_array2   = new unsigned int[N];
    m_long_array2   = new long[N];
    m_ulong_array2  = new unsigned long[N];

    m_float_array2  = new float[N];
    m_double_array2 = new double[N];

    for (int i = 0; i < N; ++i) {
        m_bool_array[i]    =  bool(i%2);
        m_bool_array2[i]   =  bool((i+1)%2);
        m_char_array[i]    =  -1*i;
        m_char_array2[i]   =  -2*i;
        m_uchar_array[i]   =   3u*i;
        m_uchar_array2[i]  =   4u*i;
        m_short_array[i]   =  -1*i;
        m_short_array2[i]  =  -2*i;
        m_ushort_array[i]  =   3u*i;
        m_ushort_array2[i] =   4u*i;
        m_int_array[i]     =  -5*i;
        m_int_array2[i]    =  -6*i;
        m_uint_array[i]    =   7u*i;
        m_uint_array2[i]   =   8u*i;
        m_long_array[i]    =  -9l*i;
        m_long_array2[i]   = -10l*i;
        m_ulong_array[i]   =  11ul*i;
        m_ulong_array2[i]  =  12ul*i;

        m_float_array[i]   = -13.f*i;
        m_float_array2[i]  = -14.f*i;
        m_double_array[i]  = -15.*i;
        m_double_array2[i] = -16.*i;
    }

    m_owns_arrays = true;

    m_pod.m_int    = 888;
    m_pod.m_double = 3.14;

    m_ppod = &m_pod;
};

CppyyTestData::~CppyyTestData()
{
    destroy_arrays();
}

void CppyyTestData::destroy_arrays() {
    if (m_owns_arrays == true) {
        delete[] m_bool_array2;
        delete[] m_short_array2;
        delete[] m_ushort_array2;
        delete[] m_int_array2;
        delete[] m_uint_array2;
        delete[] m_long_array2;
        delete[] m_ulong_array2;

        delete[] m_float_array2;
        delete[] m_double_array2;

        m_owns_arrays = false;
    }
}

//- getters -----------------------------------------------------------------
bool                 CppyyTestData::get_bool()    { return m_bool; }
char                 CppyyTestData::get_char()    { return m_char; }
signed char          CppyyTestData::get_schar()   { return m_schar; }
unsigned char        CppyyTestData::get_uchar()   { return m_uchar; }
short                CppyyTestData::get_short()   { return m_short; }
unsigned short       CppyyTestData::get_ushort()  { return m_ushort; }
int                  CppyyTestData::get_int()     { return m_int; }
unsigned int         CppyyTestData::get_uint()    { return m_uint; }
long                 CppyyTestData::get_long()    { return m_long; }
unsigned long        CppyyTestData::get_ulong()   { return m_ulong; }
long long            CppyyTestData::get_llong()   { return m_llong; }
unsigned long long   CppyyTestData::get_ullong()  { return m_ullong; }
Long64_t             CppyyTestData::get_long64()  { return m_long64; }
ULong64_t            CppyyTestData::get_ulong64() { return m_ulong64; }
float                CppyyTestData::get_float()   { return m_float; }
Float16_t            CppyyTestData::get_float16() { return m_float16; }
double               CppyyTestData::get_double()  { return m_double; }
Double32_t           CppyyTestData::get_double32(){ return m_double32; }
long double          CppyyTestData::get_ldouble() { return m_ldouble; }
CppyyTestData::EWhat CppyyTestData::get_enum()    { return m_enum; }
void*                CppyyTestData::get_voidp()   { return m_voidp; }

bool*           CppyyTestData::get_bool_array()    { return m_bool_array; }
bool*           CppyyTestData::get_bool_array2()   { return m_bool_array2; }
signed char*    CppyyTestData::get_char_array()    { return m_char_array; }
signed char*    CppyyTestData::get_char_array2()   { return m_char_array2; }
unsigned char*  CppyyTestData::get_uchar_array()   { return m_uchar_array; }
unsigned char*  CppyyTestData::get_uchar_array2()  { return m_uchar_array2; }
short*          CppyyTestData::get_short_array()   { return m_short_array; }
short*          CppyyTestData::get_short_array2()  { return m_short_array2; }
unsigned short* CppyyTestData::get_ushort_array()  { return m_ushort_array; }
unsigned short* CppyyTestData::get_ushort_array2() { return m_ushort_array2; }
int*            CppyyTestData::get_int_array()     { return m_int_array; }
int*            CppyyTestData::get_int_array2()    { return m_int_array2; }
unsigned int*   CppyyTestData::get_uint_array()    { return m_uint_array; }
unsigned int*   CppyyTestData::get_uint_array2()   { return m_uint_array2; }
long*           CppyyTestData::get_long_array()    { return m_long_array; }
long*           CppyyTestData::get_long_array2()   { return m_long_array2; }
unsigned long*  CppyyTestData::get_ulong_array()   { return m_ulong_array; }
unsigned long*  CppyyTestData::get_ulong_array2()  { return m_ulong_array2; }

float*  CppyyTestData::get_float_array()   { return m_float_array; }
float*  CppyyTestData::get_float_array2()  { return m_float_array2; }
double* CppyyTestData::get_double_array()  { return m_double_array; }
double* CppyyTestData::get_double_array2() { return m_double_array2; }

CppyyTestPod   CppyyTestData::get_pod_val()     { return m_pod; }
CppyyTestPod*  CppyyTestData::get_pod_val_ptr() { return &m_pod; }
CppyyTestPod&  CppyyTestData::get_pod_val_ref() { return m_pod; }
CppyyTestPod*& CppyyTestData::get_pod_ptrref()  { return m_ppod; }

CppyyTestPod* CppyyTestData::get_pod_ptr() { return m_ppod; }

//- getters const-ref -------------------------------------------------------
const bool&                 CppyyTestData::get_bool_cr()    { return m_bool; }
const char&                 CppyyTestData::get_char_cr()    { return m_char; }
const signed char&          CppyyTestData::get_schar_cr()   { return m_schar; }
const unsigned char&        CppyyTestData::get_uchar_cr()   { return m_uchar; }
const short&                CppyyTestData::get_short_cr()   { return m_short; }
const unsigned short&       CppyyTestData::get_ushort_cr()  { return m_ushort; }
const int&                  CppyyTestData::get_int_cr()     { return m_int; }
const unsigned int&         CppyyTestData::get_uint_cr()    { return m_uint; }
const long&                 CppyyTestData::get_long_cr()    { return m_long; }
const unsigned long&        CppyyTestData::get_ulong_cr()   { return m_ulong; }
const long long&            CppyyTestData::get_llong_cr()   { return m_llong; }
const unsigned long long&   CppyyTestData::get_ullong_cr()  { return m_ullong; }
const Long64_t&             CppyyTestData::get_long64_cr()  { return m_long64; }
const ULong64_t&            CppyyTestData::get_ulong64_cr() { return m_ulong64; }
const float&                CppyyTestData::get_float_cr()   { return m_float; }
const Float16_t&            CppyyTestData::get_float16_cr() { return m_float16; }
const double&               CppyyTestData::get_double_cr()  { return m_double; }
const Double32_t&           CppyyTestData::get_double32_cr(){ return m_double32; }
const long double&          CppyyTestData::get_ldouble_cr() { return m_ldouble; }
const CppyyTestData::EWhat& CppyyTestData::get_enum_cr()    { return m_enum; }

//- getters ref -------------------------------------------------------------
bool&                 CppyyTestData::get_bool_r()    { return m_bool; }
char&                 CppyyTestData::get_char_r()    { return m_char; }
signed char&          CppyyTestData::get_schar_r()   { return m_schar; }
unsigned char&        CppyyTestData::get_uchar_r()   { return m_uchar; }
short&                CppyyTestData::get_short_r()   { return m_short; }
unsigned short&       CppyyTestData::get_ushort_r()  { return m_ushort; }
int&                  CppyyTestData::get_int_r()     { return m_int; }
unsigned int&         CppyyTestData::get_uint_r()    { return m_uint; }
long&                 CppyyTestData::get_long_r()    { return m_long; }
unsigned long&        CppyyTestData::get_ulong_r()   { return m_ulong; }
long long&            CppyyTestData::get_llong_r()   { return m_llong; }
unsigned long long&   CppyyTestData::get_ullong_r()  { return m_ullong; }
Long64_t&             CppyyTestData::get_long64_r()  { return m_long64; }
ULong64_t&            CppyyTestData::get_ulong64_r() { return m_ulong64; }
float&                CppyyTestData::get_float_r()   { return m_float; }
Float16_t&            CppyyTestData::get_float16_r() { return m_float16; }
double&               CppyyTestData::get_double_r()  { return m_double; }
Double32_t&           CppyyTestData::get_double32_r(){ return m_double32; }
long double&          CppyyTestData::get_ldouble_r() { return m_ldouble; }
CppyyTestData::EWhat& CppyyTestData::get_enum_r()    { return m_enum; }

//- setters -----------------------------------------------------------------
void CppyyTestData::set_bool(bool b)                       { m_bool    = b; }
void CppyyTestData::set_char(char c)                       { m_char    = c; }
void CppyyTestData::set_schar(signed char sc)              { m_schar   = sc; }
void CppyyTestData::set_uchar(unsigned char uc)            { m_uchar   = uc; }
void CppyyTestData::set_short(short s)                     { m_short   = s; }
void CppyyTestData::set_ushort(unsigned short us)          { m_ushort  = us; }
void CppyyTestData::set_int(int i)                         { m_int     = i; }
void CppyyTestData::set_uint(unsigned int ui)              { m_uint    = ui; }
void CppyyTestData::set_long(long l)                       { m_long    = l; }
void CppyyTestData::set_ulong(unsigned long ul)            { m_ulong   = ul; }
void CppyyTestData::set_llong(long long ll)                { m_llong   = ll; }
void CppyyTestData::set_ullong(unsigned long long ull)     { m_ullong  = ull; }
void CppyyTestData::set_long64(Long64_t l64)               { m_long64  = l64; }
void CppyyTestData::set_ulong64(ULong64_t ul64)            { m_ulong64 = ul64; }
void CppyyTestData::set_float(float f)                     { m_float   = f; }
void CppyyTestData::set_float16(Float16_t f)               { m_float16   = f; }
void CppyyTestData::set_double(double d)                   { m_double  = d; }
void CppyyTestData::set_double32(Double32_t d)             { m_double32  = d; }
void CppyyTestData::set_ldouble(long double ld)            { m_ldouble = ld; }
void CppyyTestData::set_enum(EWhat w)                      { m_enum    = w; }
void CppyyTestData::set_voidp(void* p)                     { m_voidp   = p; }

void CppyyTestData::set_pod_val(CppyyTestPod p)            { m_pod = p; }
void CppyyTestData::set_pod_ptr_in(CppyyTestPod* pp)       { m_pod = *pp; }
void CppyyTestData::set_pod_ptr_out(CppyyTestPod* pp)      { *pp = m_pod; }
void CppyyTestData::set_pod_ref(const CppyyTestPod& rp)    { m_pod = rp; }
void CppyyTestData::set_pod_ptrptr_in(CppyyTestPod** ppp)  { m_pod = **ppp; }
void CppyyTestData::set_pod_void_ptrptr_in(void** pp)        { m_pod = **((CppyyTestPod**)pp); }
void CppyyTestData::set_pod_ptrptr_out(CppyyTestPod** ppp) { delete *ppp; *ppp = new CppyyTestPod(m_pod); }
void CppyyTestData::set_pod_void_ptrptr_out(void** pp)       { delete *((CppyyTestPod**)pp);
                                                                 *((CppyyTestPod**)pp) = new CppyyTestPod(m_pod); }

void CppyyTestData::set_pod_ptr(CppyyTestPod* pp)          { m_ppod = pp; }

//- setters const-ref -------------------------------------------------------
void CppyyTestData::set_bool_cr(const bool& b)                   { m_bool    = b; }
void CppyyTestData::set_char_cr(const char& c)                   { m_char    = c; }
void CppyyTestData::set_schar_cr(const signed char& sc)          { m_schar   = sc; }
void CppyyTestData::set_uchar_cr(const unsigned char& uc)        { m_uchar   = uc; }
void CppyyTestData::set_short_cr(const short& s)                 { m_short   = s; }
void CppyyTestData::set_ushort_cr(const unsigned short& us)      { m_ushort  = us; }
void CppyyTestData::set_int_cr(const int& i)                     { m_int     = i; }
void CppyyTestData::set_uint_cr(const unsigned int& ui)          { m_uint    = ui; }
void CppyyTestData::set_long_cr(const long& l)                   { m_long    = l; }
void CppyyTestData::set_ulong_cr(const unsigned long& ul)        { m_ulong   = ul; }
void CppyyTestData::set_llong_cr(const long long& ll)            { m_llong   = ll; }
void CppyyTestData::set_ullong_cr(const unsigned long long& ull) { m_ullong  = ull; }
void CppyyTestData::set_long64_cr(const Long64_t& l64)           { m_long64  = l64; }
void CppyyTestData::set_ulong64_cr(const ULong64_t& ul64)        { m_ulong64 = ul64; }
void CppyyTestData::set_float_cr(const float& f)                 { m_float   = f; }
void CppyyTestData::set_float16_cr(const Float16_t& f)           { m_float16 = f; }
void CppyyTestData::set_double_cr(const double& d)               { m_double  = d; }
void CppyyTestData::set_double32_cr(const Double32_t& d)         { m_double32= d; }
void CppyyTestData::set_ldouble_cr(const long double& ld)        { m_ldouble = ld; }
void CppyyTestData::set_enum_cr(const EWhat& w)                  { m_enum    = w; }

//- passers -----------------------------------------------------------------
short*          CppyyTestData::pass_array(short* a)          { return a; }
unsigned short* CppyyTestData::pass_array(unsigned short* a) { return a; }
int*            CppyyTestData::pass_array(int* a)            { return a; }
unsigned int*   CppyyTestData::pass_array(unsigned int* a)   { return a; }
long*           CppyyTestData::pass_array(long* a)           { return a; }
unsigned long*  CppyyTestData::pass_array(unsigned long* a)  { return a; }
float*          CppyyTestData::pass_array(float* a)          { return a; }
double*         CppyyTestData::pass_array(double* a)         { return a; }

//- static data members -----------------------------------------------------
bool                 CppyyTestData::s_bool    = false;
char                 CppyyTestData::s_char    = 'c';
signed char          CppyyTestData::s_schar   = 's';
unsigned char        CppyyTestData::s_uchar   = 'u';
short                CppyyTestData::s_short   = -101;
unsigned short       CppyyTestData::s_ushort  =  255u;
int                  CppyyTestData::s_int     = -202;
unsigned int         CppyyTestData::s_uint    =  202u;
long                 CppyyTestData::s_long    = -303l;
unsigned long        CppyyTestData::s_ulong   =  303ul;
long long            CppyyTestData::s_llong   = -404ll;
unsigned long long   CppyyTestData::s_ullong  =  404ull;
Long64_t             CppyyTestData::s_long64  = -505ll;
ULong64_t            CppyyTestData::s_ulong64 = 505ull;
float                CppyyTestData::s_float   = -606.f;
Float16_t            CppyyTestData::s_float16 = -116.f;
double               CppyyTestData::s_double  = -707.;
Double32_t           CppyyTestData::s_double32= -132.;
long double          CppyyTestData::s_ldouble = -808.l;
CppyyTestData::EWhat CppyyTestData::s_enum    = CppyyTestData::kNothing;
void*                CppyyTestData::s_voidp   = (void*)0;

//- strings -----------------------------------------------------------------
const char* CppyyTestData::get_valid_string(const char* in) { return in; }
const char* CppyyTestData::get_invalid_string() { return (const char*)0; }


//= global functions ========================================================
long get_pod_address(CppyyTestData& c)
{
    return (long)&c.m_pod;
}

long get_int_address(CppyyTestData& c)
{
    return (long)&c.m_pod.m_int;
}

long get_double_address(CppyyTestData& c)
{
    return (long)&c.m_pod.m_double;
}


//= global variables/pointers ===============================================
bool               g_bool     = false;
char               g_char     = 'w';
signed char        g_schar    = 'v';
unsigned char      g_uchar    = 'u';
short              g_short    =  -88;
unsigned short     g_ushort   =   88u;
int                g_int      = -188;
unsigned int       g_uint     =  188u;
long               g_long     = -288;
unsigned long      g_ulong    =  288ul;
long long          g_llong    = -388ll;
unsigned long long g_ullong   =  388ull;
Long64_t           g_long64   = -488ll;
ULong64_t          g_ulong64  =  488ull;
float              g_float    = -588.f;
Float16_t          g_float16  = -16.f;
double             g_double   = -688.;
Double32_t         g_double32 = -32.;
long double        g_ldouble  = -788.l;
EFruit             g_enum     = kBanana;
void*              g_voidp    = nullptr;


//= global accessors ========================================================
void set_global_int(int i) {
    g_int = i;
}

int get_global_int() {
    return g_int;
}

CppyyTestPod* g_pod = (CppyyTestPod*)0;

bool is_global_pod(CppyyTestPod* t) {
    return t == g_pod;
}

void set_global_pod(CppyyTestPod* t) {
    g_pod = t;
}

CppyyTestPod* get_global_pod() {
    return g_pod;
}

CppyyTestPod* get_null_pod() {
    return (CppyyTestPod*)0;
}

// Python str and bytes to STL string and C string conversion
std::tuple<std::string,std::size_t> f_stlstring(std::string s)
{
    return std::tuple<std::string,std::size_t>(s, s.size());
}

std::tuple<char*,std::size_t> f_cstring(char* s)
{
    return std::tuple<char*,std::size_t>(s, std::string(s).size());
}

std::tuple<const char*,std::size_t> f_constcstring(const char* s)
{
    return std::tuple<const char*,std::size_t>(s, std::string(s).size());
}

