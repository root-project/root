class Number  {
public:
   Number() { m_int = 0; }
   Number( int i ) { m_int = i; }

   int AsInt() { return m_int; }

   Number operator+( const Number& n ) const { return Number( m_int + n.m_int ); }
   Number operator-( const Number& n ) const { return Number( m_int - n.m_int ); }
   Number operator*( const Number& n ) const { return Number( m_int * n.m_int ); }
   Number operator/( const Number& n ) const { return Number( m_int / n.m_int ); }
   Number operator%( const Number& n ) const { return Number( m_int % n.m_int ); }

   Number& operator+=( const Number& n ) { m_int += n.m_int; return *this; }
   Number& operator-=( const Number& n ) { m_int -= n.m_int; return *this; }
   Number& operator*=( const Number& n ) { m_int *= n.m_int; return *this; }
   Number& operator/=( const Number& n ) { m_int /= n.m_int; return *this; }
   Number& operator%=( const Number& n ) { m_int %= n.m_int; return *this; }

   bool operator<( const Number& n ) const { return m_int < n.m_int; }
   bool operator>( const Number& n ) const { return m_int > n.m_int; }
   bool operator<=( const Number& n ) const { return m_int <= n.m_int; }
   bool operator>=( const Number& n ) const { return m_int >= n.m_int; }
   bool operator!=( const Number& n ) const { return m_int != n.m_int; }
   bool operator==( const Number& n ) const { return m_int == n.m_int; }

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
