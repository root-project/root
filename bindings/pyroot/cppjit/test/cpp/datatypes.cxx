#include "datatypes.h"

//===========================================================================
std::vector<EFruit> vecFruits{kCitrus, kApple};

//===========================================================================
CppjitTestData::CppjitTestData() : m_const_int(17), m_owns_arrays(false) {
  m_bool = false;
  m_char = 'a';
  m_schar = 'b';
  m_uchar = 'c';
  m_wchar = L'D';
  m_char16 = u'\u00df';
  m_char32 = U'\u00df';
  m_byte = (std::byte)'d';
  m_int8 = -9;
  m_uint8 = 9;
  m_short = -11;
  m_ushort = 11u;
  m_int = -22;
  m_uint = 22u;
  m_long = -33l;
  m_ulong = 33ul;
  m_llong = -44ll;
  m_ullong = 44ull;
  m_long64 = -55ll;
  m_ulong64 = 55ull;
  m_float = -66.f;
  m_double = -77.;
  m_ldouble = -88.l;
  m_complex = {99., 101.};
  m_icomplex = {121, 141};
  m_ccomplex = {151., 161.};
  m_enum = kNothing;
  m_voidp = (void*)0;

  m_bool_array2 = new bool[N];
  m_schar_array2 = new signed char[N];
  m_uchar_array2 = new unsigned char[N];
  m_byte_array2 = new std::byte[N];
  m_int8_array2 = new int8_t[N];
  m_uint8_array2 = new uint8_t[N];
  m_short_array2 = new short[N];
  m_ushort_array2 = new unsigned short[N];
  m_int_array2 = new int[N];
  m_uint_array2 = new unsigned int[N];
  m_long_array2 = new long[N];
  m_ulong_array2 = new unsigned long[N];

  m_float_array2 = new float[N];
  m_double_array2 = new double[N];
  m_complex_array2 = new complex_t[N];
  m_ccomplex_array2 = new ccomplex_t[N];

  for (int i = 0; i < N; ++i) {
    m_bool_array[i] = bool(i % 2);
    m_bool_array2[i] = bool((i + 1) % 2);
    m_schar_array[i] = 1 * i;
    m_schar_array2[i] = 2 * i;
    m_uchar_array[i] = 1u * i;
    m_uchar_array2[i] = 2u * i;
    m_byte_array[i] = (std::byte)(3u * i);
    m_byte_array2[i] = (std::byte)(4u * i);
    m_int8_array[i] = -1 * i;
    m_int8_array2[i] = -2 * i;
    m_uint8_array[i] = 3u * i;
    m_uint8_array2[i] = 4u * i;
    m_short_array[i] = -5 * i;
    m_short_array2[i] = -6 * i;
    m_ushort_array[i] = 7u * i;
    m_ushort_array2[i] = 8u * i;
    m_int_array[i] = -9 * i;
    m_int_array2[i] = -10 * i;
    m_uint_array[i] = 11u * i;
    m_uint_array2[i] = 23u * i;
    m_long_array[i] = -13l * i;
    m_long_array2[i] = -14l * i;
    m_ulong_array[i] = 15ul * i;
    m_ulong_array2[i] = 18ul * i;

    m_float_array[i] = -13.f * i;
    m_float_array2[i] = -14.f * i;
    m_double_array[i] = -15. * i;
    m_double_array2[i] = -16. * i;
    m_complex_array2[i] = {17. * i, 18. * i};
    m_ccomplex_array2[i] = {19. * i, 20. * i};
  }

  m_owns_arrays = true;

  m_pod.m_int = 888;
  m_pod.m_double = 3.14;

  m_ppod = &m_pod;
};

CppjitTestData::~CppjitTestData() { destroy_arrays(); }

void CppjitTestData::destroy_arrays() {
  if (m_owns_arrays == true) {
    delete[] m_bool_array2;
    delete[] m_schar_array2;
    delete[] m_uchar_array2;
    delete[] m_byte_array2;
    delete[] m_int8_array2;
    delete[] m_uint8_array2;
    delete[] m_short_array2;
    delete[] m_ushort_array2;
    delete[] m_int_array2;
    delete[] m_uint_array2;
    delete[] m_long_array2;
    delete[] m_ulong_array2;

    delete[] m_float_array2;
    delete[] m_double_array2;
    delete[] m_complex_array2;
    delete[] m_ccomplex_array2;

    m_owns_arrays = false;
  }
}

//- getters -----------------------------------------------------------------
bool CppjitTestData::get_bool() { return m_bool; }
char CppjitTestData::get_char() { return m_char; }
signed char CppjitTestData::get_schar() { return m_schar; }
unsigned char CppjitTestData::get_uchar() { return m_uchar; }
wchar_t CppjitTestData::get_wchar() { return m_wchar; }
char16_t CppjitTestData::get_char16() { return m_char16; }
char32_t CppjitTestData::get_char32() { return m_char32; }
std::byte CppjitTestData::get_byte() { return m_byte; }
int8_t CppjitTestData::get_int8() { return m_int8; }
uint8_t CppjitTestData::get_uint8() { return m_uint8; }
short CppjitTestData::get_short() { return m_short; }
unsigned short CppjitTestData::get_ushort() { return m_ushort; }
int CppjitTestData::get_int() { return m_int; }
unsigned int CppjitTestData::get_uint() { return m_uint; }
long CppjitTestData::get_long() { return m_long; }
unsigned long CppjitTestData::get_ulong() { return m_ulong; }
long long CppjitTestData::get_llong() { return m_llong; }
unsigned long long CppjitTestData::get_ullong() { return m_ullong; }
Long64_t CppjitTestData::get_long64() { return m_long64; }
ULong64_t CppjitTestData::get_ulong64() { return m_ulong64; }
float CppjitTestData::get_float() { return m_float; }
double CppjitTestData::get_double() { return m_double; }
long double CppjitTestData::get_ldouble() { return m_ldouble; }
long double CppjitTestData::get_ldouble_def(long double ld) { return ld; }
complex_t CppjitTestData::get_complex() { return m_complex; }
icomplex_t CppjitTestData::get_icomplex() { return m_icomplex; }
ccomplex_t CppjitTestData::get_ccomplex() { return m_ccomplex; }
CppjitTestData::EWhat CppjitTestData::get_enum() { return m_enum; }
void* CppjitTestData::get_voidp() { return m_voidp; }

bool* CppjitTestData::get_bool_array() { return m_bool_array; }
bool* CppjitTestData::get_bool_array2() { return m_bool_array2; }
signed char* CppjitTestData::get_schar_array() { return m_schar_array; }
signed char* CppjitTestData::get_schar_array2() { return m_schar_array2; }
unsigned char* CppjitTestData::get_uchar_array() { return m_uchar_array; }
unsigned char* CppjitTestData::get_uchar_array2() { return m_uchar_array2; }
std::byte* CppjitTestData::get_byte_array() { return m_byte_array; }
std::byte* CppjitTestData::get_byte_array2() { return m_byte_array2; }
int8_t* CppjitTestData::get_int8_array() { return m_int8_array; }
int8_t* CppjitTestData::get_int8_array2() { return m_int8_array2; }
uint8_t* CppjitTestData::get_uint8_array() { return m_uint8_array; }
uint8_t* CppjitTestData::get_uint8_array2() { return m_uint8_array2; }
short* CppjitTestData::get_short_array() { return m_short_array; }
short* CppjitTestData::get_short_array2() { return m_short_array2; }
unsigned short* CppjitTestData::get_ushort_array() { return m_ushort_array; }
unsigned short* CppjitTestData::get_ushort_array2() { return m_ushort_array2; }
int* CppjitTestData::get_int_array() { return m_int_array; }
int* CppjitTestData::get_int_array2() { return m_int_array2; }
unsigned int* CppjitTestData::get_uint_array() { return m_uint_array; }
unsigned int* CppjitTestData::get_uint_array2() { return m_uint_array2; }
long* CppjitTestData::get_long_array() { return m_long_array; }
long* CppjitTestData::get_long_array2() { return m_long_array2; }
unsigned long* CppjitTestData::get_ulong_array() { return m_ulong_array; }
unsigned long* CppjitTestData::get_ulong_array2() { return m_ulong_array2; }

float* CppjitTestData::get_float_array() { return m_float_array; }
float* CppjitTestData::get_float_array2() { return m_float_array2; }
double* CppjitTestData::get_double_array() { return m_double_array; }
double* CppjitTestData::get_double_array2() { return m_double_array2; }
complex_t* CppjitTestData::get_complex_array() { return m_complex_array; }
complex_t* CppjitTestData::get_complex_array2() { return m_complex_array2; }
ccomplex_t* CppjitTestData::get_ccomplex_array() { return m_ccomplex_array; }
ccomplex_t* CppjitTestData::get_ccomplex_array2() { return m_ccomplex_array2; }

CppjitTestPod CppjitTestData::get_pod_val() { return m_pod; }
CppjitTestPod* CppjitTestData::get_pod_val_ptr() { return &m_pod; }
CppjitTestPod& CppjitTestData::get_pod_val_ref() { return m_pod; }
CppjitTestPod*& CppjitTestData::get_pod_ptrref() { return m_ppod; }

CppjitTestPod* CppjitTestData::get_pod_ptr() { return m_ppod; }

//- getters const-ref -------------------------------------------------------
const bool& CppjitTestData::get_bool_cr() { return m_bool; }
const char& CppjitTestData::get_char_cr() { return m_char; }
const signed char& CppjitTestData::get_schar_cr() { return m_schar; }
const unsigned char& CppjitTestData::get_uchar_cr() { return m_uchar; }
const wchar_t& CppjitTestData::get_wchar_cr() { return m_wchar; }
const char16_t& CppjitTestData::get_char16_cr() { return m_char16; }
const char32_t& CppjitTestData::get_char32_cr() { return m_char32; }
const std::byte& CppjitTestData::get_byte_cr() { return m_byte; }
const int8_t& CppjitTestData::get_int8_cr() { return m_int8; }
const uint8_t& CppjitTestData::get_uint8_cr() { return m_uint8; }
const short& CppjitTestData::get_short_cr() { return m_short; }
const unsigned short& CppjitTestData::get_ushort_cr() { return m_ushort; }
const int& CppjitTestData::get_int_cr() { return m_int; }
const unsigned int& CppjitTestData::get_uint_cr() { return m_uint; }
const long& CppjitTestData::get_long_cr() { return m_long; }
const unsigned long& CppjitTestData::get_ulong_cr() { return m_ulong; }
const long long& CppjitTestData::get_llong_cr() { return m_llong; }
const unsigned long long& CppjitTestData::get_ullong_cr() { return m_ullong; }
const Long64_t& CppjitTestData::get_long64_cr() { return m_long64; }
const ULong64_t& CppjitTestData::get_ulong64_cr() { return m_ulong64; }
const float& CppjitTestData::get_float_cr() { return m_float; }
const double& CppjitTestData::get_double_cr() { return m_double; }
const long double& CppjitTestData::get_ldouble_cr() { return m_ldouble; }
const complex_t& CppjitTestData::get_complex_cr() { return m_complex; }
const icomplex_t& CppjitTestData::get_icomplex_cr() { return m_icomplex; }
const ccomplex_t& CppjitTestData::get_ccomplex_cr() { return m_ccomplex; }
const CppjitTestData::EWhat& CppjitTestData::get_enum_cr() { return m_enum; }

//- getters ref -------------------------------------------------------------
bool& CppjitTestData::get_bool_r() { return m_bool; }
char& CppjitTestData::get_char_r() { return m_char; }
signed char& CppjitTestData::get_schar_r() { return m_schar; }
unsigned char& CppjitTestData::get_uchar_r() { return m_uchar; }
wchar_t& CppjitTestData::get_wchar_r() { return m_wchar; }
char16_t& CppjitTestData::get_char16_r() { return m_char16; }
char32_t& CppjitTestData::get_char32_r() { return m_char32; }
std::byte& CppjitTestData::get_byte_r() { return m_byte; }
int8_t& CppjitTestData::get_int8_r() { return m_int8; }
uint8_t& CppjitTestData::get_uint8_r() { return m_uint8; }
short& CppjitTestData::get_short_r() { return m_short; }
unsigned short& CppjitTestData::get_ushort_r() { return m_ushort; }
int& CppjitTestData::get_int_r() { return m_int; }
unsigned int& CppjitTestData::get_uint_r() { return m_uint; }
long& CppjitTestData::get_long_r() { return m_long; }
unsigned long& CppjitTestData::get_ulong_r() { return m_ulong; }
long long& CppjitTestData::get_llong_r() { return m_llong; }
unsigned long long& CppjitTestData::get_ullong_r() { return m_ullong; }
Long64_t& CppjitTestData::get_long64_r() { return m_long64; }
ULong64_t& CppjitTestData::get_ulong64_r() { return m_ulong64; }
float& CppjitTestData::get_float_r() { return m_float; }
double& CppjitTestData::get_double_r() { return m_double; }
long double& CppjitTestData::get_ldouble_r() { return m_ldouble; }
complex_t& CppjitTestData::get_complex_r() { return m_complex; }
icomplex_t& CppjitTestData::get_icomplex_r() { return m_icomplex; }
ccomplex_t& CppjitTestData::get_ccomplex_r() { return m_ccomplex; }
CppjitTestData::EWhat& CppjitTestData::get_enum_r() { return m_enum; }

//- setters -----------------------------------------------------------------
void CppjitTestData::set_bool(bool b) { m_bool = b; }
void CppjitTestData::set_char(char c) { m_char = c; }
void CppjitTestData::set_schar(signed char sc) { m_schar = sc; }
void CppjitTestData::set_uchar(unsigned char uc) { m_uchar = uc; }
void CppjitTestData::set_wchar(wchar_t wc) { m_wchar = wc; }
void CppjitTestData::set_char16(char16_t c16) { m_char16 = c16; }
void CppjitTestData::set_char32(char32_t c32) { m_char32 = c32; }
void CppjitTestData::set_byte(std::byte b) { m_byte = b; }
void CppjitTestData::set_int8(int8_t s8) { m_int8 = s8; }
void CppjitTestData::set_uint8(uint8_t u8) { m_uint8 = u8; }
void CppjitTestData::set_short(short s) { m_short = s; }
void CppjitTestData::set_ushort(unsigned short us) { m_ushort = us; }
void CppjitTestData::set_int(int i) { m_int = i; }
void CppjitTestData::set_uint(unsigned int ui) { m_uint = ui; }
void CppjitTestData::set_long(long l) { m_long = l; }
void CppjitTestData::set_ulong(unsigned long ul) { m_ulong = ul; }
void CppjitTestData::set_llong(long long ll) { m_llong = ll; }
void CppjitTestData::set_ullong(unsigned long long ull) { m_ullong = ull; }
void CppjitTestData::set_long64(Long64_t l64) { m_long64 = l64; }
void CppjitTestData::set_ulong64(ULong64_t ul64) { m_ulong64 = ul64; }
void CppjitTestData::set_float(float f) { m_float = f; }
void CppjitTestData::set_double(double d) { m_double = d; }
void CppjitTestData::set_ldouble(long double ld) { m_ldouble = ld; }
void CppjitTestData::set_complex(complex_t cd) { m_complex = cd; }
void CppjitTestData::set_icomplex(icomplex_t ci) { m_icomplex = ci; }
void CppjitTestData::set_ccomplex(ccomplex_t cd) { m_ccomplex = cd; }
void CppjitTestData::set_enum(EWhat w) { m_enum = w; }
void CppjitTestData::set_voidp(void* p) { m_voidp = p; }

void CppjitTestData::set_pod_val(CppjitTestPod p) { m_pod = p; }
void CppjitTestData::set_pod_ptr_in(CppjitTestPod* pp) { m_pod = *pp; }
void CppjitTestData::set_pod_ptr_out(CppjitTestPod* pp) { *pp = m_pod; }
void CppjitTestData::set_pod_ref(const CppjitTestPod& rp) { m_pod = rp; }
void CppjitTestData::set_pod_ptrptr_in(CppjitTestPod** ppp) { m_pod = **ppp; }
void CppjitTestData::set_pod_void_ptrptr_in(void** pp) {
  m_pod = **((CppjitTestPod**)pp);
}
void CppjitTestData::set_pod_ptrptr_out(CppjitTestPod** ppp) {
  delete *ppp;
  *ppp = new CppjitTestPod(m_pod);
}
void CppjitTestData::set_pod_void_ptrptr_out(void** pp) {
  delete *((CppjitTestPod**)pp);
  *((CppjitTestPod**)pp) = new CppjitTestPod(m_pod);
}

void CppjitTestData::set_pod_ptr(CppjitTestPod* pp) { m_ppod = pp; }

//- setters const-ref -------------------------------------------------------
void CppjitTestData::set_bool_cr(const bool& b) { m_bool = b; }
void CppjitTestData::set_char_cr(const char& c) { m_char = c; }
void CppjitTestData::set_schar_cr(const signed char& sc) { m_schar = sc; }
void CppjitTestData::set_uchar_cr(const unsigned char& uc) { m_uchar = uc; }
void CppjitTestData::set_wchar_cr(const wchar_t& wc) { m_wchar = wc; }
void CppjitTestData::set_char16_cr(const char16_t& c16) { m_char16 = c16; }
void CppjitTestData::set_char32_cr(const char32_t& c32) { m_char32 = c32; }
void CppjitTestData::set_byte_cr(const std::byte& b) { m_byte = b; }
void CppjitTestData::set_int8_cr(const int8_t& s8) { m_int8 = s8; }
void CppjitTestData::set_uint8_cr(const uint8_t& u8) { m_uint8 = u8; }
void CppjitTestData::set_short_cr(const short& s) { m_short = s; }
void CppjitTestData::set_ushort_cr(const unsigned short& us) { m_ushort = us; }
void CppjitTestData::set_int_cr(const int& i) { m_int = i; }
void CppjitTestData::set_uint_cr(const unsigned int& ui) { m_uint = ui; }
void CppjitTestData::set_long_cr(const long& l) { m_long = l; }
void CppjitTestData::set_ulong_cr(const unsigned long& ul) { m_ulong = ul; }
void CppjitTestData::set_llong_cr(const long long& ll) { m_llong = ll; }
void CppjitTestData::set_ullong_cr(const unsigned long long& ull) {
  m_ullong = ull;
}
void CppjitTestData::set_long64_cr(const Long64_t& l64) { m_long64 = l64; }
void CppjitTestData::set_ulong64_cr(const ULong64_t& ul64) { m_ulong64 = ul64; }
void CppjitTestData::set_float_cr(const float& f) { m_float = f; }
void CppjitTestData::set_double_cr(const double& d) { m_double = d; }
void CppjitTestData::set_ldouble_cr(const long double& ld) { m_ldouble = ld; }
void CppjitTestData::set_complex_cr(const complex_t& cd) { m_complex = cd; }
void CppjitTestData::set_icomplex_cr(const icomplex_t& ci) { m_icomplex = ci; }
void CppjitTestData::set_ccomplex_cr(const ccomplex_t& cd) { m_ccomplex = cd; }
void CppjitTestData::set_enum_cr(const EWhat& w) { m_enum = w; }

//- setters ref -------------------------------------------------------------
void CppjitTestData::set_bool_r(bool& b) { b = true; }
void CppjitTestData::set_char_r(char& c) { c = 'a'; }
void CppjitTestData::set_wchar_r(wchar_t& wc) { wc = 'b'; }
void CppjitTestData::set_char16_r(char16_t& c16) { c16 = u'\u6c24'; }
void CppjitTestData::set_char32_r(char32_t& c32) { c32 = U'\U0001f34e'; }
void CppjitTestData::set_schar_r(signed char& sc) { sc = 'c'; }
void CppjitTestData::set_uchar_r(unsigned char& uc) { uc = 'd'; }
void CppjitTestData::set_byte_r(std::byte& b) { b = (std::byte)'e'; }
void CppjitTestData::set_short_r(short& s) { s = -1; }
void CppjitTestData::set_ushort_r(unsigned short& us) { us = 2; }
void CppjitTestData::set_int_r(int& i) { i = -3; }
void CppjitTestData::set_uint_r(unsigned int& ui) { ui = 4; }
void CppjitTestData::set_long_r(long& l) { l = -5; }
void CppjitTestData::set_ulong_r(unsigned long& ul) { ul = 6; }
void CppjitTestData::set_llong_r(long long& ll) { ll = -7; }
void CppjitTestData::set_ullong_r(unsigned long long& ull) { ull = 8; }
void CppjitTestData::set_float_r(float& f) { f = 5.f; }
void CppjitTestData::set_double_r(double& d) { d = -5.; }
void CppjitTestData::set_ldouble_r(long double& ld) { ld = 10.l; }

//- setters ptr -------------------------------------------------------------
void CppjitTestData::set_bool_p(bool* b) { *b = true; }
void CppjitTestData::set_char_p(char* c) { *c = 'a'; }
void CppjitTestData::set_wchar_p(wchar_t* wc) { *wc = 'b'; }
void CppjitTestData::set_char16_p(char16_t* c16) { *c16 = u'\u6c24'; }
void CppjitTestData::set_char32_p(char32_t* c32) { *c32 = U'\U0001f34e'; }
void CppjitTestData::set_schar_p(signed char* sc) { *sc = 'c'; }
void CppjitTestData::set_uchar_p(unsigned char* uc) { *uc = 'd'; }
void CppjitTestData::set_byte_p(std::byte* b) { *b = (std::byte)'e'; }
void CppjitTestData::set_int8_p(int8_t* i8) { *i8 = -27; }
void CppjitTestData::set_uint8_p(uint8_t* ui8) { *ui8 = 28; }
void CppjitTestData::set_short_p(short* s) { *s = -1; }
void CppjitTestData::set_ushort_p(unsigned short* us) { *us = 2; }
void CppjitTestData::set_int_p(int* i) { *i = -3; }
void CppjitTestData::set_uint_p(unsigned int* ui) { *ui = 4; }
void CppjitTestData::set_long_p(long* l) { *l = -5; }
void CppjitTestData::set_ulong_p(unsigned long* ul) { *ul = 6; }
void CppjitTestData::set_llong_p(long long* ll) { *ll = -7; }
void CppjitTestData::set_ullong_p(unsigned long long* ull) { *ull = 8; }
void CppjitTestData::set_float_p(float* f) { *f = 5.f; }
void CppjitTestData::set_double_p(double* d) { *d = -5.; }
void CppjitTestData::set_ldouble_p(long double* ld) { *ld = 10.l; }

//- setters ptrptr ----------------------------------------------------------
void CppjitTestData::set_bool_ppa(bool** b) {
  (*b) = new bool[3];
  (*b)[0] = true;
  (*b)[1] = false;
  (*b)[2] = true;
}
void CppjitTestData::set_char_ppa(char** c) {
  (*c) = new char[3];
  (*c)[0] = 'a';
  (*c)[1] = 'b';
  (*c)[2] = 'c';
}
void CppjitTestData::set_wchar_ppa(wchar_t** wc) {
  (*wc) = new wchar_t[3];
  (*wc)[0] = 'd';
  (*wc)[1] = 'e';
  (*wc)[2] = 'f';
}
void CppjitTestData::set_char16_ppa(char16_t** c16) {
  (*c16) = new char16_t[3];
  (*c16)[0] = u'\u6c24';
  (*c16)[1] = u'\u6c25';
  (*c16)[2] = u'\u6c26';
}
void CppjitTestData::set_char32_ppa(char32_t** c32) {
  (*c32) = new char32_t[3];
  (*c32)[0] = U'\U0001f34d';
  (*c32)[1] = U'\U0001f34e';
  (*c32)[2] = U'\U0001f34f';
}
void CppjitTestData::set_schar_ppa(signed char** sc) {
  (*sc) = new signed char[3];
  (*sc)[0] = 'g';
  (*sc)[1] = 'h';
  (*sc)[2] = 'j';
}
void CppjitTestData::set_uchar_ppa(unsigned char** uc) {
  (*uc) = new unsigned char[3];
  (*uc)[0] = 'k';
  (*uc)[1] = 'l';
  (*uc)[2] = 'm';
}
void CppjitTestData::set_byte_ppa(std::byte** b) {
  (*b) = new std::byte[3];
  (*b)[0] = (std::byte)'n';
  (*b)[1] = (std::byte)'o';
  (*b)[2] = (std::byte)'p';
}
void CppjitTestData::set_int8_ppa(int8_t** i8) {
  (*i8) = new int8_t[3];
  (*i8)[0] = -27;
  (*i8)[1] = -28;
  (*i8)[2] = -29;
}
void CppjitTestData::set_uint8_ppa(uint8_t** ui8) {
  (*ui8) = new uint8_t[3];
  (*ui8)[0] = 28;
  (*ui8)[1] = 29;
  (*ui8)[2] = 30;
}
void CppjitTestData::set_short_ppa(short** s) {
  (*s) = new short[3];
  (*s)[0] = -1;
  (*s)[1] = -2;
  (*s)[2] = -3;
}
void CppjitTestData::set_ushort_ppa(unsigned short** us) {
  (*us) = new unsigned short[3];
  (*us)[0] = 4;
  (*us)[1] = 5;
  (*us)[2] = 6;
}
void CppjitTestData::set_int_ppa(int** i) {
  (*i) = new int[3];
  (*i)[0] = -7;
  (*i)[1] = -8;
  (*i)[2] = -9;
}
void CppjitTestData::set_uint_ppa(unsigned int** ui) {
  (*ui) = new unsigned int[3];
  (*ui)[0] = 10;
  (*ui)[1] = 11;
  (*ui)[2] = 12;
}
void CppjitTestData::set_long_ppa(long** l) {
  (*l) = new long[3];
  (*l)[0] = -13;
  (*l)[1] = -14;
  (*l)[2] = -15;
}
void CppjitTestData::set_ulong_ppa(unsigned long** ul) {
  (*ul) = new unsigned long[3];
  (*ul)[0] = 16;
  (*ul)[1] = 17;
  (*ul)[2] = 18;
}
void CppjitTestData::set_llong_ppa(long long** ll) {
  (*ll) = new long long[3];
  (*ll)[0] = -19;
  (*ll)[1] = -20;
  (*ll)[2] = -21;
}
void CppjitTestData::set_ullong_ppa(unsigned long long** ull) {
  (*ull) = new unsigned long long[3];
  (*ull)[0] = 22;
  (*ull)[1] = 23;
  (*ull)[2] = 24;
}
void CppjitTestData::set_float_ppa(float** f) {
  (*f) = new float[3];
  (*f)[0] = 5.f;
  (*f)[1] = 10.f;
  (*f)[2] = 20.f;
}
void CppjitTestData::set_double_ppa(double** d) {
  (*d) = new double[3];
  (*d)[0] = -5;
  (*d)[1] = -10.;
  (*d)[2] = -20.;
}
void CppjitTestData::set_ldouble_ppa(long double** ld) {
  (*ld) = new long double[3];
  (*ld)[0] = 5.l;
  (*ld)[1] = 10.f;
  (*ld)[2] = 20.l;
}

intptr_t CppjitTestData::set_char_ppm(char** c) {
  *c = (char*)malloc(4 * sizeof(char));
  return (intptr_t)*c;
}
intptr_t CppjitTestData::set_cchar_ppm(const char** cc) {
  *cc = (const char*)malloc(4 * sizeof(char));
  return (intptr_t)*cc;
}
intptr_t CppjitTestData::set_wchar_ppm(wchar_t** w) {
  *w = (wchar_t*)malloc(4 * sizeof(wchar_t));
  return (intptr_t)*w;
}
intptr_t CppjitTestData::set_char16_ppm(char16_t** c16) {
  *c16 = (char16_t*)malloc(4 * sizeof(char16_t));
  return (intptr_t)*c16;
}
intptr_t CppjitTestData::set_char32_ppm(char32_t** c32) {
  *c32 = (char32_t*)malloc(4 * sizeof(char32_t));
  return (intptr_t)*c32;
}
intptr_t CppjitTestData::set_cwchar_ppm(const wchar_t** cw) {
  *cw = (const wchar_t*)malloc(4 * sizeof(wchar_t));
  return (intptr_t)*cw;
}
intptr_t CppjitTestData::set_cchar16_ppm(const char16_t** c16) {
  *c16 = (const char16_t*)malloc(4 * sizeof(char16_t));
  return (intptr_t)*c16;
}
intptr_t CppjitTestData::set_cchar32_ppm(const char32_t** c32) {
  *c32 = (const char32_t*)malloc(4 * sizeof(char32_t));
  return (intptr_t)*c32;
}
intptr_t CppjitTestData::set_void_ppm(void** v) {
  *v = malloc(4 * sizeof(void*));
  return (intptr_t)*v;
}

intptr_t CppjitTestData::freeit(void* ptr) {
  intptr_t out = (intptr_t)ptr;
  free(ptr);
  return out;
}

//- setters r-value ---------------------------------------------------------
void CppjitTestData::set_bool_rv(bool&& b) { m_bool = b; }
void CppjitTestData::set_char_rv(char&& c) { m_char = c; }
void CppjitTestData::set_schar_rv(signed char&& sc) { m_schar = sc; }
void CppjitTestData::set_uchar_rv(unsigned char&& uc) { m_uchar = uc; }
void CppjitTestData::set_wchar_rv(wchar_t&& wc) { m_wchar = wc; }
void CppjitTestData::set_char16_rv(char16_t&& c16) { m_char16 = c16; }
void CppjitTestData::set_char32_rv(char32_t&& c32) { m_char32 = c32; }
void CppjitTestData::set_byte_rv(std::byte&& b) { m_byte = b; }
void CppjitTestData::set_int8_rv(int8_t&& s8) { m_int8 = s8; }
void CppjitTestData::set_uint8_rv(uint8_t&& u8) { m_uint8 = u8; }
void CppjitTestData::set_short_rv(short&& s) { m_short = s; }
void CppjitTestData::set_ushort_rv(unsigned short&& us) { m_ushort = us; }
void CppjitTestData::set_int_rv(int&& i) { m_int = i; }
void CppjitTestData::set_uint_rv(unsigned int&& ui) { m_uint = ui; }
void CppjitTestData::set_long_rv(long&& l) { m_long = l; }
void CppjitTestData::set_ulong_rv(unsigned long&& ul) { m_ulong = ul; }
void CppjitTestData::set_llong_rv(long long&& ll) { m_llong = ll; }
void CppjitTestData::set_ullong_rv(unsigned long long&& ull) { m_ullong = ull; }
void CppjitTestData::set_long64_rv(Long64_t&& l64) { m_long64 = l64; }
void CppjitTestData::set_ulong64_rv(ULong64_t&& ul64) { m_ulong64 = ul64; }
void CppjitTestData::set_float_rv(float&& f) { m_float = f; }
void CppjitTestData::set_double_rv(double&& d) { m_double = d; }
void CppjitTestData::set_ldouble_rv(long double&& ld) { m_ldouble = ld; }
void CppjitTestData::set_complex_rv(complex_t&& cd) { m_complex = cd; }
void CppjitTestData::set_icomplex_rv(icomplex_t&& ci) { m_icomplex = ci; }
void CppjitTestData::set_ccomplex_rv(ccomplex_t&& cd) { m_ccomplex = cd; }
void CppjitTestData::set_enum_rv(EWhat&& w) { m_enum = w; }

//- passers -----------------------------------------------------------------
unsigned char* CppjitTestData::pass_array(unsigned char* a) { return a; }
short* CppjitTestData::pass_array(short* a) { return a; }
unsigned short* CppjitTestData::pass_array(unsigned short* a) { return a; }
int* CppjitTestData::pass_array(int* a) { return a; }
unsigned int* CppjitTestData::pass_array(unsigned int* a) { return a; }
long* CppjitTestData::pass_array(long* a) { return a; }
unsigned long* CppjitTestData::pass_array(unsigned long* a) { return a; }
float* CppjitTestData::pass_array(float* a) { return a; }
double* CppjitTestData::pass_array(double* a) { return a; }
complex_t* CppjitTestData::pass_array(complex_t* a) { return a; }
ccomplex_t* CppjitTestData::pass_array(ccomplex_t* a) { return a; }

//- static data members -----------------------------------------------------
bool CppjitTestData::s_bool = false;
char CppjitTestData::s_char = 'c';
signed char CppjitTestData::s_schar = 's';
unsigned char CppjitTestData::s_uchar = 'u';
wchar_t CppjitTestData::s_wchar = L'U';
char16_t CppjitTestData::s_char16 = u'\u6c29';
char32_t CppjitTestData::s_char32 = U'\U0001f34b';
std::byte CppjitTestData::s_byte = (std::byte)'b';
int8_t CppjitTestData::s_int8 = -87;
uint8_t CppjitTestData::s_uint8 = 87;
short CppjitTestData::s_short = -101;
unsigned short CppjitTestData::s_ushort = 255u;
int CppjitTestData::s_int = -202;
unsigned int CppjitTestData::s_uint = 202u;
long CppjitTestData::s_long = -303l;
unsigned long CppjitTestData::s_ulong = 303ul;
long long CppjitTestData::s_llong = -404ll;
unsigned long long CppjitTestData::s_ullong = 404ull;
Long64_t CppjitTestData::s_long64 = -505ll;
ULong64_t CppjitTestData::s_ulong64 = 505ull;
float CppjitTestData::s_float = -606.f;
double CppjitTestData::s_double = -707.;
long double CppjitTestData::s_ldouble = -808.l;
complex_t CppjitTestData::s_complex = {909., -909.};
icomplex_t CppjitTestData::s_icomplex = {979, -979};
ccomplex_t CppjitTestData::s_ccomplex = {919., -919.};
CppjitTestData::EWhat CppjitTestData::s_enum = CppjitTestData::kNothing;
void* CppjitTestData::s_voidp = (void*)0;
std::string CppjitTestData::s_strv = "Hello";
std::string* CppjitTestData::s_strp = nullptr;

//- strings -----------------------------------------------------------------
const char* CppjitTestData::get_valid_string(const char* in) { return in; }
const char* CppjitTestData::get_invalid_string() { return (const char*)0; }
const wchar_t* CppjitTestData::get_valid_wstring(const wchar_t* in) {
  return in;
}
const wchar_t* CppjitTestData::get_invalid_wstring() {
  return (const wchar_t*)0;
}
const char16_t* CppjitTestData::get_valid_string16(const char16_t* in) {
  return in;
}
const char16_t* CppjitTestData::get_invalid_string16() {
  return (const char16_t*)0;
}
const char32_t* CppjitTestData::get_valid_string32(const char32_t* in) {
  return in;
}
const char32_t* CppjitTestData::get_invalid_string32() {
  return (const char32_t*)0;
}

//= global functions ========================================================
intptr_t get_pod_address(CppjitTestData& c) { return (intptr_t)&c.m_pod; }

intptr_t get_int_address(CppjitTestData& c) { return (intptr_t)&c.m_pod.m_int; }

intptr_t get_double_address(CppjitTestData& c) {
  return (intptr_t)&c.m_pod.m_double;
}

//= global variables/pointers ===============================================
bool g_bool = false;
char g_char = 'w';
signed char g_schar = 'v';
unsigned char g_uchar = 'u';
wchar_t g_wchar = L'U';
char16_t g_char16 = u'\u6c21';
char32_t g_char32 = U'\u6c21';
std::byte g_byte = (std::byte)'x';
int8_t g_int8 = -66;
uint8_t g_uint8 = 66;
short g_short = -88;
unsigned short g_ushort = 88u;
int g_int = -188;
unsigned int g_uint = 188u;
long g_long = -288;
unsigned long g_ulong = 288ul;
long long g_llong = -388ll;
unsigned long long g_ullong = 388ull;
Long64_t g_long64 = -488ll;
ULong64_t g_ulong64 = 488ull;
float g_float = -588.f;
double g_double = -688.;
long double g_ldouble = -788.l;
complex_t g_complex = {808., -808.};
icomplex_t g_icomplex = {909, -909};
ccomplex_t g_ccomplex = {858., -858.};
EFruit g_enum = kBanana;
void* g_voidp = nullptr;

//= global accessors ========================================================
void set_global_int(int i) { g_int = i; }

int get_global_int() { return g_int; }

CppjitTestPod* g_pod = (CppjitTestPod*)0;

bool is_global_pod(CppjitTestPod* t) { return t == g_pod; }

void set_global_pod(CppjitTestPod* t) { g_pod = t; }

CppjitTestPod* get_global_pod() { return g_pod; }

CppjitTestPod* get_null_pod() { return (CppjitTestPod*)0; }

std::string g_some_global_string = "C++";
std::string get_some_global_string() { return g_some_global_string; }
std::string g_some_global_string2 = "C++";
std::string get_some_global_string2() { return g_some_global_string2; }

const char16_t* g_some_global_string16 = u"z\u00df\u6c34";
const char32_t* g_some_global_string32 = U"z\u00df\u6c34\U0001f34c";

std::string SomeStaticDataNS::s_some_static_string = "C++";
std::string SomeStaticDataNS::get_some_static_string() {
  return s_some_static_string;
}
std::string SomeStaticDataNS::s_some_static_string2 = "C++";
std::string SomeStaticDataNS::get_some_static_string2() {
  return s_some_static_string2;
}

StorableData gData{5.};

//= special case of "byte" arrays ===========================================
int64_t sum_uc_data(unsigned char* data, int size) {
  int64_t total = 0;
  for (int i = 0; i < size; ++i)
    total += int64_t(data[i]);
  return total;
}

int64_t sum_byte_data(std::byte* data, int size) {
  return sum_uc_data((unsigned char*)data, size);
}

//= function pointer passing ================================================
int sum_of_int1(int i1, int i2) { return i1 + i2; }

int sum_of_int2(int i1, int i2) { return 2 * i1 + i2; }

int (*sum_of_int_ptr)(int, int) = sum_of_int1;

int call_sum_of_int(int i1, int i2) {
  if (sum_of_int_ptr)
    return (*sum_of_int_ptr)(i1, i2);
  return -1;
}

double sum_of_double(double d1, double d2) { return d1 + d2; }

double call_double_double(double (*f)(double, double), double d1, double d2) {
  if (!f)
    return -1.;
  return f(d1, d2);
}

//= callable passing ========================================================
int call_int_int(int (*f)(int, int), int i1, int i2) { return f(i1, i2); }

void call_void(void (*f)(int), int i) { f(i); }

int call_refi(void (*f)(int&)) {
  int i = -1;
  f(i);
  return i;
}

int call_refl(void (*f)(long&)) {
  long l = -1L;
  f(l);
  return l;
}

int call_refd(void (*f)(double&)) {
  double d = -1.;
  f(d);
  return d;
}

StoreCallable::StoreCallable(double (*f)(double, double)) : fF(f) {
  /* empty */
}

void StoreCallable::set_callable(double (*f)(double, double)) { fF = f; }

double StoreCallable::operator()(double d1, double d2) { return fF(d1, d2); }

//= callable through std::function ==========================================
double call_double_double_sf(const std::function<double(double, double)>& f,
                             double d1, double d2) {
  return f(d1, d2);
}

int call_int_int_sf(const std::function<int(int, int)>& f, int i1, int i2) {
  return f(i1, i2);
}

void call_void_sf(const std::function<void(int)>& f, int i) { f(i); }

int call_refi_sf(const std::function<void(int&)>& f) {
  int i = -1;
  f(i);
  return i;
}

int call_refl_sf(const std::function<void(long&)>& f) {
  long l = -1L;
  f(l);
  return l;
}

int call_refd_sf(const std::function<void(double&)>& f) {
  double d = -1.;
  f(d);
  return d;
}

StoreCallable_sf::StoreCallable_sf(
    const std::function<double(double, double)>& f)
    : fF(f) {
  /* empty */
}

void StoreCallable_sf::set_callable(
    const std::function<double(double, double)>& f) {
  fF = f;
}

double StoreCallable_sf::operator()(double d1, double d2) { return fF(d1, d2); }

//= array of C strings passing ==============================================
std::vector<std::string>
ArrayOfCStrings::takes_array_of_cstrings(const char* args[], int len) {
  std::vector<std::string> v;
  v.reserve(len);
  for (int i = 0; i < len; ++i)
    v.emplace_back(args[i]);

  return v;
}

//= aggregate testing ======================================================
int AggregateTest::Aggregate1::sInt = 17;
int AggregateTest::Aggregate2::sInt = 27;

//= multi-dim arrays =======================================================
namespace MultiDimArrays {

template <typename T> static inline T** allocate_2d(size_t N, size_t M) {
  T** arr = (T**)malloc(sizeof(void*) * N);
  for (size_t i = 0; i < N; ++i)
    arr[i] = (T*)malloc(sizeof(T) * M);
  return arr;
}

static inline void free_2d(void** arr, size_t N) {
  if (arr) {
    for (size_t i = 0; i < N; ++i)
      free(arr[i]);
    free(arr);
  }
}

template <typename T>
static inline T*** allocate_3d(size_t N, size_t M, size_t K) {
  T*** arr = (T***)malloc(sizeof(void*) * N);
  for (size_t i = 0; i < N; ++i) {
    arr[i] = (T**)malloc(sizeof(void*) * M);
    for (size_t j = 0; j < M; ++j)
      arr[i][j] = (T*)malloc(sizeof(T) * K);
  }
  return arr;
}

static inline void free_3d(void*** arr, size_t N, size_t M) {
  if (arr) {
    for (size_t i = 0; i < N; ++i) {
      for (size_t j = 0; j < M; ++j)
        free(arr[i][j]);
      free(arr[i]);
    }
    free(arr);
  }
}

} // namespace MultiDimArrays

MultiDimArrays::DataHolder::DataHolder() {
  m_short2a = allocate_2d<short>(5, 7);
  m_unsigned_short2a = allocate_2d<unsigned short>(5, 7);
  m_int2a = allocate_2d<int>(5, 7);
  m_unsigned_int2a = allocate_2d<unsigned int>(5, 7);
  m_long2a = allocate_2d<long>(5, 7);
  m_unsigned_long2a = allocate_2d<unsigned long>(5, 7);
  m_long_long2a = allocate_2d<long long>(5, 7);
  m_unsigned_long_long2a = allocate_2d<unsigned long long>(5, 7);
  m_float2a = allocate_2d<float>(5, 7);
  m_double2a = allocate_2d<double>(5, 7);

  for (size_t i = 0; i < 5; ++i) {
    for (size_t j = 0; j < 7; ++j) {
      size_t val = 5 * i + j;
      m_short2a[i][j] = (short)val;
      m_unsigned_short2a[i][j] = (unsigned short)val;
      m_int2a[i][j] = (int)val;
      m_unsigned_int2a[i][j] = (unsigned int)val;
      m_long2a[i][j] = (long)val;
      m_unsigned_long2a[i][j] = (unsigned long)val;
      m_long_long2a[i][j] = (long long)val;
      m_unsigned_long_long2a[i][j] = (unsigned long long)val;
      m_float2a[i][j] = (float)val;
      m_double2a[i][j] = (double)val;
    }
  }

  m_short2b = nullptr;
  m_unsigned_short2b = nullptr;
  m_int2b = nullptr;
  m_unsigned_int2b = nullptr;
  m_long2b = nullptr;
  m_unsigned_long2b = nullptr;
  m_long_long2b = nullptr;
  m_unsigned_long_long2b = nullptr;
  m_float2b = nullptr;
  m_double2b = nullptr;

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 5; ++j) {
      size_t val = 3 * i + j;
      m_short2c[i][j] = (short)val;
      m_unsigned_short2c[i][j] = (unsigned short)val;
      m_int2c[i][j] = (int)val;
      m_unsigned_int2c[i][j] = (unsigned int)val;
      m_long2c[i][j] = (long)val;
      m_unsigned_long2c[i][j] = (unsigned long)val;
      m_long_long2c[i][j] = (long long)val;
      m_unsigned_long_long2c[i][j] = (unsigned long long)val;
      m_float2c[i][j] = (float)val;
      m_double2c[i][j] = (double)val;

      for (size_t k = 0; k < 7; ++k) {
        val = 3 * i + 2 * j + k;
        m_short3c[i][j][k] = (short)val;
        m_unsigned_short3c[i][j][k] = (unsigned short)val;
        m_int3c[i][j][k] = (int)val;
        m_unsigned_int3c[i][j][k] = (unsigned int)val;
        m_long3c[i][j][k] = (long)val;
        m_unsigned_long3c[i][j][k] = (unsigned long)val;
        m_long_long3c[i][j][k] = (long long)val;
        m_unsigned_long_long3c[i][j][k] = (unsigned long long)val;
        m_float3c[i][j][k] = (float)val;
        m_double3c[i][j][k] = (double)val;
      }
    }
  }

  m_short3a = allocate_3d<short>(5, 7, 11);
  m_unsigned_short3a = allocate_3d<unsigned short>(5, 7, 11);
  m_int3a = allocate_3d<int>(5, 7, 11);
  m_unsigned_int3a = allocate_3d<unsigned int>(5, 7, 11);
  m_long3a = allocate_3d<long>(5, 7, 11);
  m_unsigned_long3a = allocate_3d<unsigned long>(5, 7, 11);
  m_long_long3a = allocate_3d<long long>(5, 7, 11);
  m_unsigned_long_long3a = allocate_3d<unsigned long long>(5, 7, 11);
  m_float3a = allocate_3d<float>(5, 7, 11);
  m_double3a = allocate_3d<double>(5, 7, 11);

  for (size_t i = 0; i < 5; ++i) {
    for (size_t j = 0; j < 7; ++j) {
      for (size_t k = 0; k < 11; ++k) {
        size_t val = 7 * i + 3 * j + k;
        m_short3a[i][j][k] = (short)val;
        m_unsigned_short3a[i][j][k] = (unsigned short)val;
        m_int3a[i][j][k] = (int)val;
        m_unsigned_int3a[i][j][k] = (unsigned int)val;
        m_long3a[i][j][k] = (long)val;
        m_unsigned_long3a[i][j][k] = (unsigned long)val;
        m_long_long3a[i][j][k] = (long long)val;
        m_unsigned_long_long3a[i][j][k] = (unsigned long long)val;
        m_float3a[i][j][k] = (float)val;
        m_double3a[i][j][k] = (double)val;
      }
    }
  }
}

MultiDimArrays::DataHolder::~DataHolder() {
  free_2d((void**)m_short2a, 5);
  free_2d((void**)m_unsigned_short2a, 5);
  free_2d((void**)m_int2a, 5);
  free_2d((void**)m_unsigned_int2a, 5);
  free_2d((void**)m_long2a, 5);
  free_2d((void**)m_unsigned_long2a, 5);
  free_2d((void**)m_long_long2a, 5);
  free_2d((void**)m_unsigned_long_long2a, 5);
  free_2d((void**)m_float2a, 5);
  free_2d((void**)m_double2a, 5);

  free_2d((void**)m_short2b, 5);
  free_2d((void**)m_unsigned_short2b, 5);
  free_2d((void**)m_int2b, 5);
  free_2d((void**)m_unsigned_int2b, 5);
  free_2d((void**)m_long2b, 5);
  free_2d((void**)m_unsigned_long2b, 5);
  free_2d((void**)m_long_long2b, 5);
  free_2d((void**)m_unsigned_long_long2b, 5);
  free_2d((void**)m_float2b, 5);
  free_2d((void**)m_double2b, 5);

  free_3d((void***)m_short3a, 5, 7);
  free_3d((void***)m_unsigned_short3a, 5, 7);
  free_3d((void***)m_int3a, 5, 7);
  free_3d((void***)m_unsigned_int3a, 5, 7);
  free_3d((void***)m_long3a, 5, 7);
  free_3d((void***)m_unsigned_long3a, 5, 7);
  free_3d((void***)m_long_long3a, 5, 7);
  free_3d((void***)m_unsigned_long_long3a, 5, 7);
  free_3d((void***)m_float3a, 5, 7);
  free_3d((void***)m_double3a, 5, 7);
}

#define MULTIDIM_ARRAYS_NEW2D(type, name)                                      \
  type** MultiDimArrays::DataHolder::new_##name##2d(int N, int M) {            \
    type** arr = allocate_2d<type>(N, M);                                      \
    for (size_t i = 0; i < N; ++i) {                                           \
      for (size_t j = 0; j < M; ++j) {                                         \
        size_t val = 7 * i + j;                                                \
        arr[i][j] = (type)val;                                                 \
      }                                                                        \
    }                                                                          \
    return arr;                                                                \
  }

MULTIDIM_ARRAYS_NEW2D(short, short)
MULTIDIM_ARRAYS_NEW2D(unsigned short, ushort)
MULTIDIM_ARRAYS_NEW2D(int, int)
MULTIDIM_ARRAYS_NEW2D(unsigned int, uint)
MULTIDIM_ARRAYS_NEW2D(long, long)
MULTIDIM_ARRAYS_NEW2D(unsigned long, ulong)
MULTIDIM_ARRAYS_NEW2D(long long, llong)
MULTIDIM_ARRAYS_NEW2D(unsigned long long, ullong)
MULTIDIM_ARRAYS_NEW2D(float, float)
MULTIDIM_ARRAYS_NEW2D(double, double)

//===========================================================================
namespace Int8_Uint8_Arrays {
int8_t test[6] = {-0x12, -0x34, -0x56, -0x78};
uint8_t utest[6] = {0x12, 0x34, 0x56, 0x78};
} // namespace Int8_Uint8_Arrays
