/*
  File: roottest/python/basic/Operators.C
  Author: WLavrijsen@lbl.gov
  Created: 04/15/05
  Last: 05/20/10
*/

class Number  {
public:
   Number() { m_int = 0; }
   Number( int i ) { m_int = i; }

   int AsInt() { return m_int; }

   Number operator+( const Number& n ) const { return Number( m_int + n.m_int ); }
   Number operator+( int n ) const { return Number( m_int + n ); }
   Number operator-( const Number& n ) const { return Number( m_int - n.m_int ); }
   Number operator-( int n ) const { return Number( m_int - n ); }
   Number operator*( const Number& n ) const { return Number( m_int * n.m_int ); }
   Number operator*( int n ) const { return Number( m_int * n ); }
   Number operator/( const Number& n ) const { return Number( m_int / n.m_int ); }
   Number operator/( int n ) const { return Number( m_int / n ); }
   Number operator%( const Number& n ) const { return Number( m_int % n.m_int ); }
   Number operator%( int n ) const { return Number( m_int % n ); }

   Number& operator+=( const Number& n ) { m_int += n.m_int; return *this; }
   Number& operator-=( const Number& n ) { m_int -= n.m_int; return *this; }
   Number& operator*=( const Number& n ) { m_int *= n.m_int; return *this; }
   Number& operator/=( const Number& n ) { m_int /= n.m_int; return *this; }
   Number& operator%=( const Number& n ) { m_int %= n.m_int; return *this; }

   Number operator-() { return Number( -m_int ); }

   bool operator<( const Number& n ) const { return m_int < n.m_int; }
   bool operator>( const Number& n ) const { return m_int > n.m_int; }
   bool operator<=( const Number& n ) const { return m_int <= n.m_int; }
   bool operator>=( const Number& n ) const { return m_int >= n.m_int; }
   bool operator!=( const Number& n ) const { return m_int != n.m_int; }
   bool operator==( const Number& n ) const { return m_int == n.m_int; }

   operator bool() { return m_int != 0; }

   Number operator&( const Number& n ) const { return Number( m_int & n.m_int ); }
   Number operator|( const Number& n ) const { return Number( m_int | n.m_int ); }
   Number operator^( const Number& n ) const { return Number( m_int ^ n.m_int ); }

   Number& operator&=( const Number& n ) { m_int &= n.m_int; return *this; }
   Number& operator|=( const Number& n ) { m_int |= n.m_int; return *this; }
   Number& operator^=( const Number& n ) { m_int ^= n.m_int; return *this; }

   Number operator<<( int i ) const { return Number( m_int << i ); }
   Number operator>>( int i ) const { return Number( m_int >> i ); }

private:
   int m_int;
};

//----------------------------------------------------------------------------
struct OperatorCharStar {
   OperatorCharStar() : m_str( (char*)"OperatorCharStar" ) {}
   operator char*() { return m_str; }
   char* m_str;
};

struct OperatorConstCharStar {
   OperatorConstCharStar() : m_str( "OperatorConstCharStar" ) {}
   operator const char*() { return m_str; }
   const char* m_str;
};

struct OperatorInt {
   operator int() { return m_int; }
   int m_int;
};

struct OperatorLong {
#ifdef _WIN64
   operator int64_t() { return m_long; }
   int64_t m_long;
#else
   operator long() { return m_long; }
   long m_long;
#endif
};

struct OperatorDouble {
   operator double() { return m_double; }
   double m_double;
};

struct OperatorShort {
   operator short() { return m_short; }
   unsigned short m_short;
};

struct OperatorUnsignedInt {
   operator unsigned int() { return m_uint; }
   unsigned int m_uint;
};

struct OperatorUnsignedLong {
#ifdef _WIN64
   operator uint64_t() { return m_ulong; }
   uint64_t m_ulong;
#else
   operator unsigned long() { return m_ulong; }
   unsigned long m_ulong;
#endif
};

struct OperatorFloat {
   operator float() { return m_float; }
   float m_float;
};
