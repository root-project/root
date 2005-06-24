// @(#)root/mathcore:$Name:  $:$Id: GenVectorIO.hv 1.0 2005/06/23 12:00:00 moneta Exp $
// Authors: W. Brown, M. Fischler, L. Moneta, A. Zsenei   06/2005 

 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2005 , LCG ROOT MathLib Team (FNAL component)        *
  *                                                                    *
  *                                                                    *
  **********************************************************************/

// Support templates (class and function) for stream i/o of vectors
//     This is a utuility to allow for control, via manipulators, of the
//     form of
//
// Created by: W. E. Brown and M. Fischler at Tue Jun 21 2005
//
// Last update: Tue Jun 21 2005
//
#ifndef ROOT_Math_GenVectorIO 
#define ROOT_Math_GenVectorIO 1

#include <cctype>
#include <iostream>


namespace ROOT  {
namespace Math  {

namespace detail  {


// -------- Manipulator support ----------


enum manip_t { open, sep, close, bitforbit };


inline  int
  ios_data( int k )
{
  static int const  ios_data[4]  = { std::ios::xalloc()  // open
                                   , std::ios::xalloc()  // sep
                                   , std::ios::xalloc()  // close
                                   , std::ios::xalloc()  // bitforbit
                                   };

  return ios_data[k];

}  // ios_data()


template< class char_t, class traits_t >
  inline  char_t
  get_manip( std::basic_ios<char_t,traits_t> & ios
           , manip_t m
           )
{
  char_t  ch  = static_cast<char_t>( ios.iword( ios_data(m) ) );
  if( ch )  return ch;

  switch( m )
  { default        : return ios.widen( '?' );
    case open      : return ios.widen( '(' );
    case close     : return ios.widen( ')' );
    case sep       : return ios.widen( ',' );
    case bitforbit : return ch;
  }

}  // get_manip<>()


template< class char_t, class traits_t >
  inline  void
  set_manip( std::basic_ios<char_t,traits_t> & ios
           , manip_t m
           , char_t ch
           )
{
  ios.iword( ios_data(m) ) = static_cast<long>(ch);

}  // set_manip<>()


template< class char_t >
  class manipulator
{
public:
  explicit
    manipulator( manip_t m
               , char_t  ch = 0
               )
    : m(m)
    , ch(ch)
  { }

  template< class traits_t >
    void
    set( std::basic_ios<char_t,traits_t> & ios ) const
  {
    set_manip<char_t>( ios, m, ch );
  }

private:
  manip_t  m;
  char_t   ch;

};  // manipulator<>


template< class char_t, class traits_t >
  inline
  std::basic_istream<char_t,traits_t> &
  require_delim( std::basic_istream<char_t,traits_t> & is
               , manip_t m
               )
{
  char_t delim = get_manip( is, m );
  if( std::isspace(delim) )  return is;

  char_t ch;
  is >> ch;
  if( ch != delim )
    is.setstate( std::ios::failbit );

  return is;

}  // require_delim<>()


}  // namespace detail


// --------- Functions that allow a user to control vector I/O ----------


template< class char_t, class traits_t >
  inline
  std::basic_ios<char_t,traits_t> &
  operator << ( std::basic_ios<char_t,traits_t>   & ios
              , detail::manipulator<char_t> const & manip
              )

{
  manip.set(ios);
  return ios;

}  // op<< <>()


template< class char_t, class traits_t >
  inline
  std::basic_ios<char_t,traits_t> &
  operator >> ( std::basic_ios<char_t,traits_t>   & ios
              , detail::manipulator<char_t> const & manip
              )

{
  manip.set(ios);
  return ios;

}  // op>> <>()


template< class char_t >
  inline
  detail::manipulator<char_t>
  set_open( char_t ch )
{
  return detail::manipulator<char_t>( detail::open, ch );

}  // set_open<>()


template< class char_t >
  inline
  detail::manipulator<char_t>
  set_separator( char_t ch )
{
  return detail::manipulator<char_t>( detail::sep, ch );

}  // set_separator<>()


template< class char_t >
  inline
  detail::manipulator<char_t>
  set_close( char_t ch )
{
  return detail::manipulator<char_t>( detail::close, ch );

}  // set_close<>()


template< class char_t >
  inline
  detail::manipulator<char_t>
  human_readable( char_t ch )
{
  return detail::manipulator<char_t>( detail::bitforbit, '\00' );

}  // human_readable<>()


template< class char_t >
  inline
  detail::manipulator<char_t>
  machine_readable( char_t ch )
{
  return detail::manipulator<char_t>( detail::bitforbit, '\01' );

}  // machine_readable<>()



}  // namespace ROOT
}  // namespace Math


#endif  // ROOT_Math_GenVectorIO 
