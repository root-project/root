// @(#)root/mathcore:$Id$
// Authors: W. Brown, M. Fischler, L. Moneta    2005

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
#ifndef ROOT_Math_GenVector_GenVectorIO
#define ROOT_Math_GenVector_GenVectorIO  1

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
    : fMan(m)
    , fChar(ch)
  { }

  template< class traits_t >
    void
    set( std::basic_ios<char_t,traits_t> & ios ) const
  {
    set_manip<char_t>( ios, fMan, fChar );
  }

private:
  manip_t  fMan;
  char_t   fChar;

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


template< class char_t, class traits_t >
  inline
  std::basic_ostream<char_t,traits_t> &
  operator << ( std::basic_ostream<char_t,traits_t> & os
              , detail::manipulator<char_t> const   & manip
              )

{
  manip.set(os);
  return os;

}  // op<< <>()


template< class char_t, class traits_t >
  inline
  std::basic_istream<char_t,traits_t> &
  operator >> ( std::basic_istream<char_t,traits_t> & is
              , detail::manipulator<char_t> const   & manip
              )

{
  manip.set(is);
  return is;

}  // op>> <>()

}  // namespace detail


// --------- Functions that allow a user to control vector I/O ----------



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


template< class char_t, class traits_t >
  inline
  std::basic_ios<char_t,traits_t> &
  human_readable( std::basic_ios<char_t,traits_t> & ios )
{
  ios.iword( ios_data(detail::bitforbit) ) = 0L;
  return ios;

}  // human_readable<>()


template< class char_t, class traits_t >
  inline
  std::basic_ios<char_t,traits_t> &
  machine_readable( std::basic_ios<char_t,traits_t> & ios )
{
  ios.iword( ios_data(detail::bitforbit) ) = 1L;
  return ios;

}  // machine_readable<>()



}  // namespace ROOT
}  // namespace Math


#endif  // ROOT_Math_GenVector_GenVectorIO
