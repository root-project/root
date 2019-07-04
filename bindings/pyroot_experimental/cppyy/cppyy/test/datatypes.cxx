#include "datatypes.h"


//===========================================================================
std::vector<EFruit> vecFruits{kCitrus, kApple};


//===========================================================================
CppyyTestData::CppyyTestData() : m_const_int(17), m_owns_arrays(false)
{
    m_bool     = false;
    m_char     = 'a';
    m_schar    = 'b';
    m_uchar    = 'c';
    m_wchar    = L'D';
    m_int8     =  -9;
    m_uint8    =   9;
    m_short    = -11;
    m_ushort   =  11u;
    m_int      = -22;
    m_uint     =  22u;
    m_long     = -33l;
    m_ulong    =  33ul;
    m_llong    = -44ll;
    m_ullong   =  44ull;
    m_long64   = -55ll;
    m_ulong64  =  55ull;
    m_float    = -66.f;
    m_double   = -77.;
    m_ldouble  = -88.l;
    m_complex  = {99., 101.};
    m_icomplex = {121, 141};
    m_enum     = kNothing;
    m_voidp    = (void*)0;

    m_bool_array2     = new bool[N];
    m_uchar_array2    = new unsigned char[N];
    m_short_array2    = new short[N];
    m_ushort_array2   = new unsigned short[N];
    m_int_array2      = new int[N];
    m_uint_array2     = new unsigned int[N];
    m_long_array2     = new long[N];
    m_ulong_array2    = new unsigned long[N];

    m_float_array2    = new float[N];
    m_double_array2   = new double[N];
    m_complex_array2  = new complex_t[N];

    for (int i = 0; i < N; ++i) {
        m_bool_array[i]      =  bool(i%2);
        m_bool_array2[i]     =  bool((i+1)%2);
        m_uchar_array[i]     =   1u*i;
        m_uchar_array2[i]    =   2u*i;
        m_short_array[i]     =  -1*i;
        m_short_array2[i]    =  -2*i;
        m_ushort_array[i]    =   3u*i;
        m_ushort_array2[i]   =   4u*i;
        m_int_array[i]       =  -5*i;
        m_int_array2[i]      =  -6*i;
        m_uint_array[i]      =   7u*i;
        m_uint_array2[i]     =   8u*i;
        m_long_array[i]      =  -9l*i;
        m_long_array2[i]     = -10l*i;
        m_ulong_array[i]     =  11ul*i;
        m_ulong_array2[i]    =  12ul*i;

        m_float_array[i]     = -13.f*i;
        m_float_array2[i]    = -14.f*i;
        m_double_array[i]    = -15.*i;
        m_double_array2[i]   = -16.*i;
        m_complex_array2[i]  = {17.*i, 18.*i};
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
        delete[] m_uchar_array2;
        delete[] m_short_array2;
        delete[] m_ushort_array2;
        delete[] m_int_array2;
        delete[] m_uint_array2;
        delete[] m_long_array2;
        delete[] m_ulong_array2;

        delete[] m_float_array2;
        delete[] m_double_array2;
        delete[] m_complex_array2;

        m_owns_arrays = false;
    }
}

//- getters -----------------------------------------------------------------
bool                 CppyyTestData::get_bool()     { return m_bool; }
char                 CppyyTestData::get_char()     { return m_char; }
signed char          CppyyTestData::get_schar()    { return m_schar; }
unsigned char        CppyyTestData::get_uchar()    { return m_uchar; }
wchar_t              CppyyTestData::get_wchar()    { return m_wchar; }
int8_t               CppyyTestData::get_int8()     { return m_int8; }
uint8_t              CppyyTestData::get_uint8()    { return m_uint8; }
short                CppyyTestData::get_short()    { return m_short; }
unsigned short       CppyyTestData::get_ushort()   { return m_ushort; }
int                  CppyyTestData::get_int()      { return m_int; }
unsigned int         CppyyTestData::get_uint()     { return m_uint; }
long                 CppyyTestData::get_long()     { return m_long; }
unsigned long        CppyyTestData::get_ulong()    { return m_ulong; }
long long            CppyyTestData::get_llong()    { return m_llong; }
unsigned long long   CppyyTestData::get_ullong()   { return m_ullong; }
Long64_t             CppyyTestData::get_long64()   { return m_long64; }
ULong64_t            CppyyTestData::get_ulong64()  { return m_ulong64; }
float                CppyyTestData::get_float()    { return m_float; }
double               CppyyTestData::get_double()   { return m_double; }
long double          CppyyTestData::get_ldouble()  { return m_ldouble; }
long double          CppyyTestData::get_ldouble_def(long double ld) { return ld; }
complex_t            CppyyTestData::get_complex()  { return m_complex; }
icomplex_t           CppyyTestData::get_icomplex() { return m_icomplex; }
CppyyTestData::EWhat CppyyTestData::get_enum()     { return m_enum; }
void*                CppyyTestData::get_voidp()    { return m_voidp; }

bool*           CppyyTestData::get_bool_array()    { return m_bool_array; }
bool*           CppyyTestData::get_bool_array2()   { return m_bool_array2; }
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

float*      CppyyTestData::get_float_array()     { return m_float_array; }
float*      CppyyTestData::get_float_array2()    { return m_float_array2; }
double*     CppyyTestData::get_double_array()    { return m_double_array; }
double*     CppyyTestData::get_double_array2()   { return m_double_array2; }
complex_t*  CppyyTestData::get_complex_array()   { return m_complex_array; }
complex_t*  CppyyTestData::get_complex_array2()  { return m_complex_array2; }

CppyyTestPod   CppyyTestData::get_pod_val()     { return m_pod; }
CppyyTestPod*  CppyyTestData::get_pod_val_ptr() { return &m_pod; }
CppyyTestPod&  CppyyTestData::get_pod_val_ref() { return m_pod; }
CppyyTestPod*& CppyyTestData::get_pod_ptrref()  { return m_ppod; }

CppyyTestPod* CppyyTestData::get_pod_ptr() { return m_ppod; }

//- getters const-ref -------------------------------------------------------
const bool&                 CppyyTestData::get_bool_cr()     { return m_bool; }
const char&                 CppyyTestData::get_char_cr()     { return m_char; }
const signed char&          CppyyTestData::get_schar_cr()    { return m_schar; }
const unsigned char&        CppyyTestData::get_uchar_cr()    { return m_uchar; }
const wchar_t&              CppyyTestData::get_wchar_cr()    { return m_wchar; }
const int8_t&               CppyyTestData::get_int8_cr()     { return m_int8; }
const uint8_t&              CppyyTestData::get_uint8_cr()    { return m_uint8; }
const short&                CppyyTestData::get_short_cr()    { return m_short; }
const unsigned short&       CppyyTestData::get_ushort_cr()   { return m_ushort; }
const int&                  CppyyTestData::get_int_cr()      { return m_int; }
const unsigned int&         CppyyTestData::get_uint_cr()     { return m_uint; }
const long&                 CppyyTestData::get_long_cr()     { return m_long; }
const unsigned long&        CppyyTestData::get_ulong_cr()    { return m_ulong; }
const long long&            CppyyTestData::get_llong_cr()    { return m_llong; }
const unsigned long long&   CppyyTestData::get_ullong_cr()   { return m_ullong; }
const Long64_t&             CppyyTestData::get_long64_cr()   { return m_long64; }
const ULong64_t&            CppyyTestData::get_ulong64_cr()  { return m_ulong64; }
const float&                CppyyTestData::get_float_cr()    { return m_float; }
const double&               CppyyTestData::get_double_cr()   { return m_double; }
const long double&          CppyyTestData::get_ldouble_cr()  { return m_ldouble; }
const complex_t&            CppyyTestData::get_complex_cr()  { return m_complex; }
const icomplex_t&           CppyyTestData::get_icomplex_cr() { return m_icomplex; }
const CppyyTestData::EWhat& CppyyTestData::get_enum_cr()     { return m_enum; }

//- getters ref -------------------------------------------------------------
bool&                 CppyyTestData::get_bool_r()     { return m_bool; }
char&                 CppyyTestData::get_char_r()     { return m_char; }
signed char&          CppyyTestData::get_schar_r()    { return m_schar; }
unsigned char&        CppyyTestData::get_uchar_r()    { return m_uchar; }
wchar_t&              CppyyTestData::get_wchar_r()    { return m_wchar; }
int8_t&               CppyyTestData::get_int8_r()     { return m_int8; }
uint8_t&              CppyyTestData::get_uint8_r()    { return m_uint8; }
short&                CppyyTestData::get_short_r()    { return m_short; }
unsigned short&       CppyyTestData::get_ushort_r()   { return m_ushort; }
int&                  CppyyTestData::get_int_r()      { return m_int; }
unsigned int&         CppyyTestData::get_uint_r()     { return m_uint; }
long&                 CppyyTestData::get_long_r()     { return m_long; }
unsigned long&        CppyyTestData::get_ulong_r()    { return m_ulong; }
long long&            CppyyTestData::get_llong_r()    { return m_llong; }
unsigned long long&   CppyyTestData::get_ullong_r()   { return m_ullong; }
Long64_t&             CppyyTestData::get_long64_r()   { return m_long64; }
ULong64_t&            CppyyTestData::get_ulong64_r()  { return m_ulong64; }
float&                CppyyTestData::get_float_r()    { return m_float; }
double&               CppyyTestData::get_double_r()   { return m_double; }
long double&          CppyyTestData::get_ldouble_r()  { return m_ldouble; }
complex_t&            CppyyTestData::get_complex_r()  { return m_complex; }
icomplex_t&           CppyyTestData::get_icomplex_r() { return m_icomplex; }
CppyyTestData::EWhat& CppyyTestData::get_enum_r()     { return m_enum; }

//- setters -----------------------------------------------------------------
void CppyyTestData::set_bool(bool b)                       { m_bool     = b; }
void CppyyTestData::set_char(char c)                       { m_char     = c; }
void CppyyTestData::set_schar(signed char sc)              { m_schar    = sc; }
void CppyyTestData::set_uchar(unsigned char uc)            { m_uchar    = uc; }
void CppyyTestData::set_wchar(wchar_t wc)                  { m_wchar    = wc; }
void CppyyTestData::set_int8(int8_t s8)                    { m_int8     = s8; }
void CppyyTestData::set_uint8(uint8_t u8)                  { m_uint8    = u8; }
void CppyyTestData::set_short(short s)                     { m_short    = s; }
void CppyyTestData::set_ushort(unsigned short us)          { m_ushort   = us; }
void CppyyTestData::set_int(int i)                         { m_int      = i; }
void CppyyTestData::set_uint(unsigned int ui)              { m_uint     = ui; }
void CppyyTestData::set_long(long l)                       { m_long     = l; }
void CppyyTestData::set_ulong(unsigned long ul)            { m_ulong    = ul; }
void CppyyTestData::set_llong(long long ll)                { m_llong    = ll; }
void CppyyTestData::set_ullong(unsigned long long ull)     { m_ullong   = ull; }
void CppyyTestData::set_long64(Long64_t l64)               { m_long64   = l64; }
void CppyyTestData::set_ulong64(ULong64_t ul64)            { m_ulong64  = ul64; }
void CppyyTestData::set_float(float f)                     { m_float    = f; }
void CppyyTestData::set_double(double d)                   { m_double   = d; }
void CppyyTestData::set_ldouble(long double ld)            { m_ldouble  = ld; }
void CppyyTestData::set_complex(complex_t cd)              { m_complex  = cd; }
void CppyyTestData::set_icomplex(icomplex_t ci)            { m_icomplex = ci; }
void CppyyTestData::set_enum(EWhat w)                      { m_enum     = w; }
void CppyyTestData::set_voidp(void* p)                     { m_voidp    = p; }

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
void CppyyTestData::set_bool_cr(const bool& b)                   { m_bool     = b; }
void CppyyTestData::set_char_cr(const char& c)                   { m_char     = c; }
void CppyyTestData::set_schar_cr(const signed char& sc)          { m_schar    = sc; }
void CppyyTestData::set_uchar_cr(const unsigned char& uc)        { m_uchar    = uc; }
void CppyyTestData::set_wchar_cr(const wchar_t& wc)              { m_wchar    = wc; }
void CppyyTestData::set_int8_cr(const int8_t& s8)                { m_int8     = s8; }
void CppyyTestData::set_uint8_cr(const uint8_t& u8)              { m_uint8    = u8; }
void CppyyTestData::set_short_cr(const short& s)                 { m_short    = s; }
void CppyyTestData::set_ushort_cr(const unsigned short& us)      { m_ushort   = us; }
void CppyyTestData::set_int_cr(const int& i)                     { m_int      = i; }
void CppyyTestData::set_uint_cr(const unsigned int& ui)          { m_uint     = ui; }
void CppyyTestData::set_long_cr(const long& l)                   { m_long     = l; }
void CppyyTestData::set_ulong_cr(const unsigned long& ul)        { m_ulong    = ul; }
void CppyyTestData::set_llong_cr(const long long& ll)            { m_llong    = ll; }
void CppyyTestData::set_ullong_cr(const unsigned long long& ull) { m_ullong   = ull; }
void CppyyTestData::set_long64_cr(const Long64_t& l64)           { m_long64   = l64; }
void CppyyTestData::set_ulong64_cr(const ULong64_t& ul64)        { m_ulong64  = ul64; }
void CppyyTestData::set_float_cr(const float& f)                 { m_float    = f; }
void CppyyTestData::set_double_cr(const double& d)               { m_double   = d; }
void CppyyTestData::set_ldouble_cr(const long double& ld)        { m_ldouble  = ld; }
void CppyyTestData::set_complex_cr(const complex_t& cd)          { m_complex  = cd; }
void CppyyTestData::set_icomplex_cr(const icomplex_t& ci)        { m_icomplex = ci; }
void CppyyTestData::set_enum_cr(const EWhat& w)                  { m_enum     = w; }

//- setters r-value ---------------------------------------------------------
void CppyyTestData::set_bool_rv(bool&& b)                   { m_bool     = b; }
void CppyyTestData::set_char_rv(char&& c)                   { m_char     = c; }
void CppyyTestData::set_schar_rv(signed char&& sc)          { m_schar    = sc; }
void CppyyTestData::set_uchar_rv(unsigned char&& uc)        { m_uchar    = uc; }
void CppyyTestData::set_wchar_rv(wchar_t&& wc)              { m_wchar    = wc; }
void CppyyTestData::set_int8_rv(int8_t&& s8)                { m_int8     = s8; }
void CppyyTestData::set_uint8_rv(uint8_t&& u8)              { m_uint8    = u8; }
void CppyyTestData::set_short_rv(short&& s)                 { m_short    = s; }
void CppyyTestData::set_ushort_rv(unsigned short&& us)      { m_ushort   = us; }
void CppyyTestData::set_int_rv(int&& i)                     { m_int      = i; }
void CppyyTestData::set_uint_rv(unsigned int&& ui)          { m_uint     = ui; }
void CppyyTestData::set_long_rv(long&& l)                   { m_long     = l; }
void CppyyTestData::set_ulong_rv(unsigned long&& ul)        { m_ulong    = ul; }
void CppyyTestData::set_llong_rv(long long&& ll)            { m_llong    = ll; }
void CppyyTestData::set_ullong_rv(unsigned long long&& ull) { m_ullong   = ull; }
void CppyyTestData::set_long64_rv(Long64_t&& l64)           { m_long64   = l64; }
void CppyyTestData::set_ulong64_rv(ULong64_t&& ul64)        { m_ulong64  = ul64; }
void CppyyTestData::set_float_rv(float&& f)                 { m_float    = f; }
void CppyyTestData::set_double_rv(double&& d)               { m_double   = d; }
void CppyyTestData::set_ldouble_rv(long double&& ld)        { m_ldouble  = ld; }
void CppyyTestData::set_complex_rv(complex_t&& cd)          { m_complex  = cd; }
void CppyyTestData::set_icomplex_rv(icomplex_t&& ci)        { m_icomplex = ci; }
void CppyyTestData::set_enum_rv(EWhat&& w)                  { m_enum     = w; }

//- passers -----------------------------------------------------------------
unsigned char*  CppyyTestData::pass_array(unsigned char* a)  { return a; }
short*          CppyyTestData::pass_array(short* a)          { return a; }
unsigned short* CppyyTestData::pass_array(unsigned short* a) { return a; }
int*            CppyyTestData::pass_array(int* a)            { return a; }
unsigned int*   CppyyTestData::pass_array(unsigned int* a)   { return a; }
long*           CppyyTestData::pass_array(long* a)           { return a; }
unsigned long*  CppyyTestData::pass_array(unsigned long* a)  { return a; }
float*          CppyyTestData::pass_array(float* a)          { return a; }
double*         CppyyTestData::pass_array(double* a)         { return a; }
complex_t*      CppyyTestData::pass_array(complex_t* a)      { return a; }

//- static data members -----------------------------------------------------
bool                 CppyyTestData::s_bool     = false;
char                 CppyyTestData::s_char     = 'c';
signed char          CppyyTestData::s_schar    = 's';
unsigned char        CppyyTestData::s_uchar    = 'u';
wchar_t              CppyyTestData::s_wchar    = L'U';
int8_t               CppyyTestData::s_int8     = - 87;
uint8_t              CppyyTestData::s_uint8    =   87;
short                CppyyTestData::s_short    = -101;
unsigned short       CppyyTestData::s_ushort   =  255u;
int                  CppyyTestData::s_int      = -202;
unsigned int         CppyyTestData::s_uint     =  202u;
long                 CppyyTestData::s_long     = -303l;
unsigned long        CppyyTestData::s_ulong    =  303ul;
long long            CppyyTestData::s_llong    = -404ll;
unsigned long long   CppyyTestData::s_ullong   =  404ull;
Long64_t             CppyyTestData::s_long64   = -505ll;
ULong64_t            CppyyTestData::s_ulong64  = 505ull;
float                CppyyTestData::s_float    = -606.f;
double               CppyyTestData::s_double   = -707.;
long double          CppyyTestData::s_ldouble  = -808.l;
complex_t            CppyyTestData::s_complex  = {909., -909.};
icomplex_t           CppyyTestData::s_icomplex = {979, -979};
CppyyTestData::EWhat CppyyTestData::s_enum     = CppyyTestData::kNothing;
void*                CppyyTestData::s_voidp    = (void*)0;

//- strings -----------------------------------------------------------------
const char*    CppyyTestData::get_valid_string(const char* in) { return in; }
const char*    CppyyTestData::get_invalid_string() { return (const char*)0; }
const wchar_t* CppyyTestData::get_valid_wstring(const wchar_t* in) { return in; }
const wchar_t* CppyyTestData::get_invalid_wstring() { return (const wchar_t*)0; }


//= global functions ========================================================
intptr_t get_pod_address(CppyyTestData& c)
{
    return (intptr_t)&c.m_pod;
}

intptr_t get_int_address(CppyyTestData& c)
{
    return (intptr_t)&c.m_pod.m_int;
}

intptr_t get_double_address(CppyyTestData& c)
{
    return (intptr_t)&c.m_pod.m_double;
}


//= global variables/pointers ===============================================
bool               g_bool     = false;
char               g_char     = 'w';
signed char        g_schar    = 'v';
unsigned char      g_uchar    = 'u';
wchar_t            g_wchar    = L'U';
int8_t             g_int8     =  -66;
uint8_t            g_uint8    =   66;
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
double             g_double   = -688.;
long double        g_ldouble  = -788.l;
complex_t          g_complex  = {808., -808.};
icomplex_t         g_icomplex = {909,  -909};
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


//= function pointer passing ================================================
int sum_of_int(int i1, int i2) {
    return i1+i2;
}

int (*sum_of_int_ptr)(int, int) = sum_of_int;

double sum_of_double(double d1, double d2) {
    return d1+d2;
}

double call_double_double(double (*f)(double, double), double d1, double d2) {
    return f(d1, d2);
}


//= callable passing ========================================================
int call_int_int(int (*f)(int, int), int i1, int i2) {
    return f(i1, i2);
}

void call_void(void (*f)(int), int i) {
    f(i);
}

int call_refi(void (*f)(int&)) {
    int i = -1; f(i); return i;
}

int call_refl(void (*f)(long&)) {
    long l = -1L; f(l); return l;
}

int call_refd(void (*f)(double&)) {
    double d = -1.; f(d); return d;
}


StoreCallable::StoreCallable(double (*f)(double, double)) : fF(f) {
    /* empty */
}

void StoreCallable::set_callable(double (*f)(double, double)) {
    fF = f;
}

double StoreCallable::operator()(double d1, double d2) {
    return fF(d1, d2);
}

//= callable through std::function ==========================================
double call_double_double_sf(const std::function<double(double, double)>& f, double d1, double d2) {
    return f(d1, d2);
}

int call_int_int_sf(const std::function<int(int, int)>& f, int i1, int i2) {
    return f(i1, i2);
}

void call_void_sf(const std::function<void(int)>& f, int i) {
    f(i);
}

int call_refi_sf(const std::function<void(int&)>& f) {
    int i = -1; f(i); return i;
}

int call_refl_sf(const std::function<void(long&)>& f) {
    long l = -1L; f(l); return l;
}

int call_refd_sf(const std::function<void(double&)>& f) {
    double d = -1.; f(d); return d;
}


StoreCallable_sf::StoreCallable_sf(const std::function<double(double, double)>& f) : fF(f) {
    /* empty */
}

void StoreCallable_sf::set_callable(const std::function<double(double, double)>& f) {
    fF = f;
}

double StoreCallable_sf::operator()(double d1, double d2) {
    return fF(d1, d2);
}
