// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; tab-width: 8 -*-
//
// TRInternalFunction_Impl__ctors.h: based in Rcpp R/C++ interface class library -- generated helper code for TRInternalFunction_Impl.h
//
// Copyright (C) 2010 - 2013 Dirk Eddelbuettel and Romain Francois
// Modified by Omar Zapata for ROOTR
//
// This file is based in Rcpp.

#ifndef ROOT_R_TRInternalFunction_Impl_ctors
#define ROOT_R_TRInternalFunction_Impl_ctors

template <typename OUT>
TRInternalFunction_Impl(OUT(*fun)(void))
{
   set(Rcpp::XPtr< Rcpp::CppFunction0<OUT> >(new Rcpp::CppFunction0<OUT>(fun), false)) ;
}




template <typename OUT, typename U0>
TRInternalFunction_Impl(OUT(*fun)(U0 u0))
{
   set(Rcpp::XPtr< Rcpp::CppFunction1<OUT, U0> >(new Rcpp::CppFunction1<OUT, U0>(fun), false)) ;
}




template <typename OUT, typename U0, typename U1>
TRInternalFunction_Impl(OUT(*fun)(U0 u0, U1 u1))
{
   set(Rcpp::XPtr< Rcpp::CppFunction2<OUT, U0, U1> >(new Rcpp::CppFunction2<OUT, U0, U1>(fun), false)) ;
}




template <typename OUT, typename U0, typename U1, typename U2>
TRInternalFunction_Impl(OUT(*fun)(U0 u0, U1 u1, U2 u2))
{
   set(Rcpp::XPtr< Rcpp::CppFunction3<OUT, U0, U1, U2> >(new Rcpp::CppFunction3<OUT, U0, U1, U2>(fun), false)) ;
}




template <typename OUT, typename U0, typename U1, typename U2, typename U3>
TRInternalFunction_Impl(OUT(*fun)(U0 u0, U1 u1, U2 u2, U3 u3))
{
   set(Rcpp::XPtr< Rcpp::CppFunction4<OUT, U0, U1, U2, U3> >(new Rcpp::CppFunction4<OUT, U0, U1, U2, U3>(fun), false)) ;
}




template <typename OUT, typename U0, typename U1, typename U2, typename U3, typename U4>
TRInternalFunction_Impl(OUT(*fun)(U0 u0, U1 u1, U2 u2, U3 u3, U4 u4))
{
   set(Rcpp::XPtr< Rcpp::CppFunction5<OUT, U0, U1, U2, U3, U4> >(new Rcpp::CppFunction5<OUT, U0, U1, U2, U3, U4>(fun), false)) ;
}




template <typename OUT, typename U0, typename U1, typename U2, typename U3, typename U4, typename U5>
TRInternalFunction_Impl(OUT(*fun)(U0 u0, U1 u1, U2 u2, U3 u3, U4 u4, U5 u5))
{
   set(Rcpp::XPtr< Rcpp::CppFunction6<OUT, U0, U1, U2, U3, U4, U5> >(new Rcpp::CppFunction6<OUT, U0, U1, U2, U3, U4, U5>(fun), false)) ;
}




template <typename OUT, typename U0, typename U1, typename U2, typename U3, typename U4, typename U5, typename U6>
TRInternalFunction_Impl(OUT(*fun)(U0 u0, U1 u1, U2 u2, U3 u3, U4 u4, U5 u5, U6 u6))
{
   set(Rcpp::XPtr< Rcpp::CppFunction7<OUT, U0, U1, U2, U3, U4, U5, U6> >(new Rcpp::CppFunction7<OUT, U0, U1, U2, U3, U4, U5, U6>(fun), false)) ;
}




template <typename OUT, typename U0, typename U1, typename U2, typename U3, typename U4, typename U5, typename U6, typename U7>
TRInternalFunction_Impl(OUT(*fun)(U0 u0, U1 u1, U2 u2, U3 u3, U4 u4, U5 u5, U6 u6, U7 u7))
{
   set(Rcpp::XPtr< Rcpp::CppFunction8<OUT, U0, U1, U2, U3, U4, U5, U6, U7> >(new Rcpp::CppFunction8<OUT, U0, U1, U2, U3, U4, U5, U6, U7>(fun), false)) ;
}




template <typename OUT, typename U0, typename U1, typename U2, typename U3, typename U4, typename U5, typename U6, typename U7, typename U8>
TRInternalFunction_Impl(OUT(*fun)(U0 u0, U1 u1, U2 u2, U3 u3, U4 u4, U5 u5, U6 u6, U7 u7, U8 u8))
{
   set(Rcpp::XPtr< Rcpp::CppFunction9<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8> >(new Rcpp::CppFunction9<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8>(fun), false)) ;
}




template <typename OUT, typename U0, typename U1, typename U2, typename U3, typename U4, typename U5, typename U6, typename U7, typename U8, typename U9>
TRInternalFunction_Impl(OUT(*fun)(U0 u0, U1 u1, U2 u2, U3 u3, U4 u4, U5 u5, U6 u6, U7 u7, U8 u8, U9 u9))
{
   set(Rcpp::XPtr< Rcpp::CppFunction10<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9> >(new Rcpp::CppFunction10<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9>(fun), false)) ;
}




template <typename OUT, typename U0, typename U1, typename U2, typename U3, typename U4, typename U5, typename U6, typename U7, typename U8, typename U9, typename U10>
TRInternalFunction_Impl(OUT(*fun)(U0 u0, U1 u1, U2 u2, U3 u3, U4 u4, U5 u5, U6 u6, U7 u7, U8 u8, U9 u9, U10 u10))
{
   set(Rcpp::XPtr< Rcpp::CppFunction11<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10> >(new Rcpp::CppFunction11<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10>(fun), false)) ;
}




template <typename OUT, typename U0, typename U1, typename U2, typename U3, typename U4, typename U5, typename U6, typename U7, typename U8, typename U9, typename U10, typename U11>
TRInternalFunction_Impl(OUT(*fun)(U0 u0, U1 u1, U2 u2, U3 u3, U4 u4, U5 u5, U6 u6, U7 u7, U8 u8, U9 u9, U10 u10, U11 u11))
{
   set(Rcpp::XPtr< Rcpp::CppFunction12<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11> >(new Rcpp::CppFunction12<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11>(fun), false)) ;
}




template <typename OUT, typename U0, typename U1, typename U2, typename U3, typename U4, typename U5, typename U6, typename U7, typename U8, typename U9, typename U10, typename U11, typename U12>
TRInternalFunction_Impl(OUT(*fun)(U0 u0, U1 u1, U2 u2, U3 u3, U4 u4, U5 u5, U6 u6, U7 u7, U8 u8, U9 u9, U10 u10, U11 u11, U12 u12))
{
   set(Rcpp::XPtr< Rcpp::CppFunction13<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12> >(new Rcpp::CppFunction13<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12>(fun), false)) ;
}




template <typename OUT, typename U0, typename U1, typename U2, typename U3, typename U4, typename U5, typename U6, typename U7, typename U8, typename U9, typename U10, typename U11, typename U12, typename U13>
TRInternalFunction_Impl(OUT(*fun)(U0 u0, U1 u1, U2 u2, U3 u3, U4 u4, U5 u5, U6 u6, U7 u7, U8 u8, U9 u9, U10 u10, U11 u11, U12 u12, U13 u13))
{
   set(Rcpp::XPtr< Rcpp::CppFunction14<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13> >(new Rcpp::CppFunction14<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13>(fun), false)) ;
}




template <typename OUT, typename U0, typename U1, typename U2, typename U3, typename U4, typename U5, typename U6, typename U7, typename U8, typename U9, typename U10, typename U11, typename U12, typename U13, typename U14>
TRInternalFunction_Impl(OUT(*fun)(U0 u0, U1 u1, U2 u2, U3 u3, U4 u4, U5 u5, U6 u6, U7 u7, U8 u8, U9 u9, U10 u10, U11 u11, U12 u12, U13 u13, U14 u14))
{
   set(Rcpp::XPtr< Rcpp::CppFunction15<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14> >(new Rcpp::CppFunction15<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14>(fun), false)) ;
}




template <typename OUT, typename U0, typename U1, typename U2, typename U3, typename U4, typename U5, typename U6, typename U7, typename U8, typename U9, typename U10, typename U11, typename U12, typename U13, typename U14, typename U15>
TRInternalFunction_Impl(OUT(*fun)(U0 u0, U1 u1, U2 u2, U3 u3, U4 u4, U5 u5, U6 u6, U7 u7, U8 u8, U9 u9, U10 u10, U11 u11, U12 u12, U13 u13, U14 u14, U15 u15))
{
   set(Rcpp::XPtr< Rcpp::CppFunction16<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15> >(new Rcpp::CppFunction16<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15>(fun), false)) ;
}




template <typename OUT, typename U0, typename U1, typename U2, typename U3, typename U4, typename U5, typename U6, typename U7, typename U8, typename U9, typename U10, typename U11, typename U12, typename U13, typename U14, typename U15, typename U16>
TRInternalFunction_Impl(OUT(*fun)(U0 u0, U1 u1, U2 u2, U3 u3, U4 u4, U5 u5, U6 u6, U7 u7, U8 u8, U9 u9, U10 u10, U11 u11, U12 u12, U13 u13, U14 u14, U15 u15, U16 u16))
{
   set(Rcpp::XPtr< Rcpp::CppFunction17<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16> >(new Rcpp::CppFunction17<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16>(fun), false)) ;
}




template <typename OUT, typename U0, typename U1, typename U2, typename U3, typename U4, typename U5, typename U6, typename U7, typename U8, typename U9, typename U10, typename U11, typename U12, typename U13, typename U14, typename U15, typename U16, typename U17>
TRInternalFunction_Impl(OUT(*fun)(U0 u0, U1 u1, U2 u2, U3 u3, U4 u4, U5 u5, U6 u6, U7 u7, U8 u8, U9 u9, U10 u10, U11 u11, U12 u12, U13 u13, U14 u14, U15 u15, U16 u16, U17 u17))
{
   set(Rcpp::XPtr< Rcpp::CppFunction18<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17> >(new Rcpp::CppFunction18<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17>(fun), false)) ;
}




template <typename OUT, typename U0, typename U1, typename U2, typename U3, typename U4, typename U5, typename U6, typename U7, typename U8, typename U9, typename U10, typename U11, typename U12, typename U13, typename U14, typename U15, typename U16, typename U17, typename U18>
TRInternalFunction_Impl(OUT(*fun)(U0 u0, U1 u1, U2 u2, U3 u3, U4 u4, U5 u5, U6 u6, U7 u7, U8 u8, U9 u9, U10 u10, U11 u11, U12 u12, U13 u13, U14 u14, U15 u15, U16 u16, U17 u17, U18 u18))
{
   set(Rcpp::XPtr< Rcpp::CppFunction19<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18> >(new Rcpp::CppFunction19<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18>(fun), false)) ;
}




template <typename OUT, typename U0, typename U1, typename U2, typename U3, typename U4, typename U5, typename U6, typename U7, typename U8, typename U9, typename U10, typename U11, typename U12, typename U13, typename U14, typename U15, typename U16, typename U17, typename U18, typename U19>
TRInternalFunction_Impl(OUT(*fun)(U0 u0, U1 u1, U2 u2, U3 u3, U4 u4, U5 u5, U6 u6, U7 u7, U8 u8, U9 u9, U10 u10, U11 u11, U12 u12, U13 u13, U14 u14, U15 u15, U16 u16, U17 u17, U18 u18, U19 u19))
{
   set(Rcpp::XPtr< Rcpp::CppFunction20<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19> >(new Rcpp::CppFunction20<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19>(fun), false)) ;
}




template <typename OUT, typename U0, typename U1, typename U2, typename U3, typename U4, typename U5, typename U6, typename U7, typename U8, typename U9, typename U10, typename U11, typename U12, typename U13, typename U14, typename U15, typename U16, typename U17, typename U18, typename U19, typename U20>
TRInternalFunction_Impl(OUT(*fun)(U0 u0, U1 u1, U2 u2, U3 u3, U4 u4, U5 u5, U6 u6, U7 u7, U8 u8, U9 u9, U10 u10, U11 u11, U12 u12, U13 u13, U14 u14, U15 u15, U16 u16, U17 u17, U18 u18, U19 u19, U20 u20))
{
   set(Rcpp::XPtr< Rcpp::CppFunction21<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20> >(new Rcpp::CppFunction21<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20>(fun), false)) ;
}




template <typename OUT, typename U0, typename U1, typename U2, typename U3, typename U4, typename U5, typename U6, typename U7, typename U8, typename U9, typename U10, typename U11, typename U12, typename U13, typename U14, typename U15, typename U16, typename U17, typename U18, typename U19, typename U20, typename U21>
TRInternalFunction_Impl(OUT(*fun)(U0 u0, U1 u1, U2 u2, U3 u3, U4 u4, U5 u5, U6 u6, U7 u7, U8 u8, U9 u9, U10 u10, U11 u11, U12 u12, U13 u13, U14 u14, U15 u15, U16 u16, U17 u17, U18 u18, U19 u19, U20 u20, U21 u21))
{
   set(Rcpp::XPtr< Rcpp::CppFunction22<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21> >(new Rcpp::CppFunction22<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21>(fun), false)) ;
}




template <typename OUT, typename U0, typename U1, typename U2, typename U3, typename U4, typename U5, typename U6, typename U7, typename U8, typename U9, typename U10, typename U11, typename U12, typename U13, typename U14, typename U15, typename U16, typename U17, typename U18, typename U19, typename U20, typename U21, typename U22>
TRInternalFunction_Impl(OUT(*fun)(U0 u0, U1 u1, U2 u2, U3 u3, U4 u4, U5 u5, U6 u6, U7 u7, U8 u8, U9 u9, U10 u10, U11 u11, U12 u12, U13 u13, U14 u14, U15 u15, U16 u16, U17 u17, U18 u18, U19 u19, U20 u20, U21 u21, U22 u22))
{
   set(Rcpp::XPtr< Rcpp::CppFunction23<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22> >(new Rcpp::CppFunction23<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22>(fun), false)) ;
}




template <typename OUT, typename U0, typename U1, typename U2, typename U3, typename U4, typename U5, typename U6, typename U7, typename U8, typename U9, typename U10, typename U11, typename U12, typename U13, typename U14, typename U15, typename U16, typename U17, typename U18, typename U19, typename U20, typename U21, typename U22, typename U23>
TRInternalFunction_Impl(OUT(*fun)(U0 u0, U1 u1, U2 u2, U3 u3, U4 u4, U5 u5, U6 u6, U7 u7, U8 u8, U9 u9, U10 u10, U11 u11, U12 u12, U13 u13, U14 u14, U15 u15, U16 u16, U17 u17, U18 u18, U19 u19, U20 u20, U21 u21, U22 u22, U23 u23))
{
   set(Rcpp::XPtr< Rcpp::CppFunction24<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23> >(new Rcpp::CppFunction24<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23>(fun), false)) ;
}




template <typename OUT, typename U0, typename U1, typename U2, typename U3, typename U4, typename U5, typename U6, typename U7, typename U8, typename U9, typename U10, typename U11, typename U12, typename U13, typename U14, typename U15, typename U16, typename U17, typename U18, typename U19, typename U20, typename U21, typename U22, typename U23, typename U24>
TRInternalFunction_Impl(OUT(*fun)(U0 u0, U1 u1, U2 u2, U3 u3, U4 u4, U5 u5, U6 u6, U7 u7, U8 u8, U9 u9, U10 u10, U11 u11, U12 u12, U13 u13, U14 u14, U15 u15, U16 u16, U17 u17, U18 u18, U19 u19, U20 u20, U21 u21, U22 u22, U23 u23, U24 u24))
{
   set(Rcpp::XPtr< Rcpp::CppFunction25<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24> >(new Rcpp::CppFunction25<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24>(fun), false)) ;
}




template <typename OUT, typename U0, typename U1, typename U2, typename U3, typename U4, typename U5, typename U6, typename U7, typename U8, typename U9, typename U10, typename U11, typename U12, typename U13, typename U14, typename U15, typename U16, typename U17, typename U18, typename U19, typename U20, typename U21, typename U22, typename U23, typename U24, typename U25>
TRInternalFunction_Impl(OUT(*fun)(U0 u0, U1 u1, U2 u2, U3 u3, U4 u4, U5 u5, U6 u6, U7 u7, U8 u8, U9 u9, U10 u10, U11 u11, U12 u12, U13 u13, U14 u14, U15 u15, U16 u16, U17 u17, U18 u18, U19 u19, U20 u20, U21 u21, U22 u22, U23 u23, U24 u24, U25 u25))
{
   set(Rcpp::XPtr< Rcpp::CppFunction26<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25> >(new Rcpp::CppFunction26<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25>(fun), false)) ;
}




template <typename OUT, typename U0, typename U1, typename U2, typename U3, typename U4, typename U5, typename U6, typename U7, typename U8, typename U9, typename U10, typename U11, typename U12, typename U13, typename U14, typename U15, typename U16, typename U17, typename U18, typename U19, typename U20, typename U21, typename U22, typename U23, typename U24, typename U25, typename U26>
TRInternalFunction_Impl(OUT(*fun)(U0 u0, U1 u1, U2 u2, U3 u3, U4 u4, U5 u5, U6 u6, U7 u7, U8 u8, U9 u9, U10 u10, U11 u11, U12 u12, U13 u13, U14 u14, U15 u15, U16 u16, U17 u17, U18 u18, U19 u19, U20 u20, U21 u21, U22 u22, U23 u23, U24 u24, U25 u25, U26 u26))
{
   set(Rcpp::XPtr< Rcpp::CppFunction27<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26> >(new Rcpp::CppFunction27<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26>(fun), false)) ;
}




template <typename OUT, typename U0, typename U1, typename U2, typename U3, typename U4, typename U5, typename U6, typename U7, typename U8, typename U9, typename U10, typename U11, typename U12, typename U13, typename U14, typename U15, typename U16, typename U17, typename U18, typename U19, typename U20, typename U21, typename U22, typename U23, typename U24, typename U25, typename U26, typename U27>
TRInternalFunction_Impl(OUT(*fun)(U0 u0, U1 u1, U2 u2, U3 u3, U4 u4, U5 u5, U6 u6, U7 u7, U8 u8, U9 u9, U10 u10, U11 u11, U12 u12, U13 u13, U14 u14, U15 u15, U16 u16, U17 u17, U18 u18, U19 u19, U20 u20, U21 u21, U22 u22, U23 u23, U24 u24, U25 u25, U26 u26, U27 u27))
{
   set(Rcpp::XPtr< Rcpp::CppFunction28<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27> >(new Rcpp::CppFunction28<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27>(fun), false)) ;
}




template <typename OUT, typename U0, typename U1, typename U2, typename U3, typename U4, typename U5, typename U6, typename U7, typename U8, typename U9, typename U10, typename U11, typename U12, typename U13, typename U14, typename U15, typename U16, typename U17, typename U18, typename U19, typename U20, typename U21, typename U22, typename U23, typename U24, typename U25, typename U26, typename U27, typename U28>
TRInternalFunction_Impl(OUT(*fun)(U0 u0, U1 u1, U2 u2, U3 u3, U4 u4, U5 u5, U6 u6, U7 u7, U8 u8, U9 u9, U10 u10, U11 u11, U12 u12, U13 u13, U14 u14, U15 u15, U16 u16, U17 u17, U18 u18, U19 u19, U20 u20, U21 u21, U22 u22, U23 u23, U24 u24, U25 u25, U26 u26, U27 u27, U28 u28))
{
   set(Rcpp::XPtr< Rcpp::CppFunction29<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28> >(new Rcpp::CppFunction29<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28>(fun), false)) ;
}




template <typename OUT, typename U0, typename U1, typename U2, typename U3, typename U4, typename U5, typename U6, typename U7, typename U8, typename U9, typename U10, typename U11, typename U12, typename U13, typename U14, typename U15, typename U16, typename U17, typename U18, typename U19, typename U20, typename U21, typename U22, typename U23, typename U24, typename U25, typename U26, typename U27, typename U28, typename U29>
TRInternalFunction_Impl(OUT(*fun)(U0 u0, U1 u1, U2 u2, U3 u3, U4 u4, U5 u5, U6 u6, U7 u7, U8 u8, U9 u9, U10 u10, U11 u11, U12 u12, U13 u13, U14 u14, U15 u15, U16 u16, U17 u17, U18 u18, U19 u19, U20 u20, U21 u21, U22 u22, U23 u23, U24 u24, U25 u25, U26 u26, U27 u27, U28 u28, U29 u29))
{
   set(Rcpp::XPtr< Rcpp::CppFunction30<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29> >(new Rcpp::CppFunction30<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29>(fun), false)) ;
}




template <typename OUT, typename U0, typename U1, typename U2, typename U3, typename U4, typename U5, typename U6, typename U7, typename U8, typename U9, typename U10, typename U11, typename U12, typename U13, typename U14, typename U15, typename U16, typename U17, typename U18, typename U19, typename U20, typename U21, typename U22, typename U23, typename U24, typename U25, typename U26, typename U27, typename U28, typename U29, typename U30>
TRInternalFunction_Impl(OUT(*fun)(U0 u0, U1 u1, U2 u2, U3 u3, U4 u4, U5 u5, U6 u6, U7 u7, U8 u8, U9 u9, U10 u10, U11 u11, U12 u12, U13 u13, U14 u14, U15 u15, U16 u16, U17 u17, U18 u18, U19 u19, U20 u20, U21 u21, U22 u22, U23 u23, U24 u24, U25 u25, U26 u26, U27 u27, U28 u28, U29 u29, U30 u30))
{
   set(Rcpp::XPtr< Rcpp::CppFunction31<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30> >(new Rcpp::CppFunction31<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30>(fun), false)) ;
}




template <typename OUT, typename U0, typename U1, typename U2, typename U3, typename U4, typename U5, typename U6, typename U7, typename U8, typename U9, typename U10, typename U11, typename U12, typename U13, typename U14, typename U15, typename U16, typename U17, typename U18, typename U19, typename U20, typename U21, typename U22, typename U23, typename U24, typename U25, typename U26, typename U27, typename U28, typename U29, typename U30, typename U31>
TRInternalFunction_Impl(OUT(*fun)(U0 u0, U1 u1, U2 u2, U3 u3, U4 u4, U5 u5, U6 u6, U7 u7, U8 u8, U9 u9, U10 u10, U11 u11, U12 u12, U13 u13, U14 u14, U15 u15, U16 u16, U17 u17, U18 u18, U19 u19, U20 u20, U21 u21, U22 u22, U23 u23, U24 u24, U25 u25, U26 u26, U27 u27, U28 u28, U29 u29, U30 u30, U31 u31))
{
   set(Rcpp::XPtr< Rcpp::CppFunction32<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31> >(new Rcpp::CppFunction32<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31>(fun), false)) ;
}




template <typename OUT, typename U0, typename U1, typename U2, typename U3, typename U4, typename U5, typename U6, typename U7, typename U8, typename U9, typename U10, typename U11, typename U12, typename U13, typename U14, typename U15, typename U16, typename U17, typename U18, typename U19, typename U20, typename U21, typename U22, typename U23, typename U24, typename U25, typename U26, typename U27, typename U28, typename U29, typename U30, typename U31, typename U32>
TRInternalFunction_Impl(OUT(*fun)(U0 u0, U1 u1, U2 u2, U3 u3, U4 u4, U5 u5, U6 u6, U7 u7, U8 u8, U9 u9, U10 u10, U11 u11, U12 u12, U13 u13, U14 u14, U15 u15, U16 u16, U17 u17, U18 u18, U19 u19, U20 u20, U21 u21, U22 u22, U23 u23, U24 u24, U25 u25, U26 u26, U27 u27, U28 u28, U29 u29, U30 u30, U31 u31, U32 u32))
{
   set(Rcpp::XPtr< Rcpp::CppFunction33<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32> >(new Rcpp::CppFunction33<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32>(fun), false)) ;
}




template <typename OUT, typename U0, typename U1, typename U2, typename U3, typename U4, typename U5, typename U6, typename U7, typename U8, typename U9, typename U10, typename U11, typename U12, typename U13, typename U14, typename U15, typename U16, typename U17, typename U18, typename U19, typename U20, typename U21, typename U22, typename U23, typename U24, typename U25, typename U26, typename U27, typename U28, typename U29, typename U30, typename U31, typename U32, typename U33>
TRInternalFunction_Impl(OUT(*fun)(U0 u0, U1 u1, U2 u2, U3 u3, U4 u4, U5 u5, U6 u6, U7 u7, U8 u8, U9 u9, U10 u10, U11 u11, U12 u12, U13 u13, U14 u14, U15 u15, U16 u16, U17 u17, U18 u18, U19 u19, U20 u20, U21 u21, U22 u22, U23 u23, U24 u24, U25 u25, U26 u26, U27 u27, U28 u28, U29 u29, U30 u30, U31 u31, U32 u32, U33 u33))
{
   set(Rcpp::XPtr< Rcpp::CppFunction34<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33> >(new Rcpp::CppFunction34<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33>(fun), false)) ;
}




template <typename OUT, typename U0, typename U1, typename U2, typename U3, typename U4, typename U5, typename U6, typename U7, typename U8, typename U9, typename U10, typename U11, typename U12, typename U13, typename U14, typename U15, typename U16, typename U17, typename U18, typename U19, typename U20, typename U21, typename U22, typename U23, typename U24, typename U25, typename U26, typename U27, typename U28, typename U29, typename U30, typename U31, typename U32, typename U33, typename U34>
TRInternalFunction_Impl(OUT(*fun)(U0 u0, U1 u1, U2 u2, U3 u3, U4 u4, U5 u5, U6 u6, U7 u7, U8 u8, U9 u9, U10 u10, U11 u11, U12 u12, U13 u13, U14 u14, U15 u15, U16 u16, U17 u17, U18 u18, U19 u19, U20 u20, U21 u21, U22 u22, U23 u23, U24 u24, U25 u25, U26 u26, U27 u27, U28 u28, U29 u29, U30 u30, U31 u31, U32 u32, U33 u33, U34 u34))
{
   set(Rcpp::XPtr< Rcpp::CppFunction35<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34> >(new Rcpp::CppFunction35<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34>(fun), false)) ;
}




template <typename OUT, typename U0, typename U1, typename U2, typename U3, typename U4, typename U5, typename U6, typename U7, typename U8, typename U9, typename U10, typename U11, typename U12, typename U13, typename U14, typename U15, typename U16, typename U17, typename U18, typename U19, typename U20, typename U21, typename U22, typename U23, typename U24, typename U25, typename U26, typename U27, typename U28, typename U29, typename U30, typename U31, typename U32, typename U33, typename U34, typename U35>
TRInternalFunction_Impl(OUT(*fun)(U0 u0, U1 u1, U2 u2, U3 u3, U4 u4, U5 u5, U6 u6, U7 u7, U8 u8, U9 u9, U10 u10, U11 u11, U12 u12, U13 u13, U14 u14, U15 u15, U16 u16, U17 u17, U18 u18, U19 u19, U20 u20, U21 u21, U22 u22, U23 u23, U24 u24, U25 u25, U26 u26, U27 u27, U28 u28, U29 u29, U30 u30, U31 u31, U32 u32, U33 u33, U34 u34, U35 u35))
{
   set(Rcpp::XPtr< Rcpp::CppFunction36<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35> >(new Rcpp::CppFunction36<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35>(fun), false)) ;
}




template <typename OUT, typename U0, typename U1, typename U2, typename U3, typename U4, typename U5, typename U6, typename U7, typename U8, typename U9, typename U10, typename U11, typename U12, typename U13, typename U14, typename U15, typename U16, typename U17, typename U18, typename U19, typename U20, typename U21, typename U22, typename U23, typename U24, typename U25, typename U26, typename U27, typename U28, typename U29, typename U30, typename U31, typename U32, typename U33, typename U34, typename U35, typename U36>
TRInternalFunction_Impl(OUT(*fun)(U0 u0, U1 u1, U2 u2, U3 u3, U4 u4, U5 u5, U6 u6, U7 u7, U8 u8, U9 u9, U10 u10, U11 u11, U12 u12, U13 u13, U14 u14, U15 u15, U16 u16, U17 u17, U18 u18, U19 u19, U20 u20, U21 u21, U22 u22, U23 u23, U24 u24, U25 u25, U26 u26, U27 u27, U28 u28, U29 u29, U30 u30, U31 u31, U32 u32, U33 u33, U34 u34, U35 u35, U36 u36))
{
   set(Rcpp::XPtr< Rcpp::CppFunction37<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35, U36> >(new Rcpp::CppFunction37<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35, U36>(fun), false)) ;
}




template <typename OUT, typename U0, typename U1, typename U2, typename U3, typename U4, typename U5, typename U6, typename U7, typename U8, typename U9, typename U10, typename U11, typename U12, typename U13, typename U14, typename U15, typename U16, typename U17, typename U18, typename U19, typename U20, typename U21, typename U22, typename U23, typename U24, typename U25, typename U26, typename U27, typename U28, typename U29, typename U30, typename U31, typename U32, typename U33, typename U34, typename U35, typename U36, typename U37>
TRInternalFunction_Impl(OUT(*fun)(U0 u0, U1 u1, U2 u2, U3 u3, U4 u4, U5 u5, U6 u6, U7 u7, U8 u8, U9 u9, U10 u10, U11 u11, U12 u12, U13 u13, U14 u14, U15 u15, U16 u16, U17 u17, U18 u18, U19 u19, U20 u20, U21 u21, U22 u22, U23 u23, U24 u24, U25 u25, U26 u26, U27 u27, U28 u28, U29 u29, U30 u30, U31 u31, U32 u32, U33 u33, U34 u34, U35 u35, U36 u36, U37 u37))
{
   set(Rcpp::XPtr< Rcpp::CppFunction38<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35, U36, U37> >(new Rcpp::CppFunction38<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35, U36, U37>(fun), false)) ;
}




template <typename OUT, typename U0, typename U1, typename U2, typename U3, typename U4, typename U5, typename U6, typename U7, typename U8, typename U9, typename U10, typename U11, typename U12, typename U13, typename U14, typename U15, typename U16, typename U17, typename U18, typename U19, typename U20, typename U21, typename U22, typename U23, typename U24, typename U25, typename U26, typename U27, typename U28, typename U29, typename U30, typename U31, typename U32, typename U33, typename U34, typename U35, typename U36, typename U37, typename U38>
TRInternalFunction_Impl(OUT(*fun)(U0 u0, U1 u1, U2 u2, U3 u3, U4 u4, U5 u5, U6 u6, U7 u7, U8 u8, U9 u9, U10 u10, U11 u11, U12 u12, U13 u13, U14 u14, U15 u15, U16 u16, U17 u17, U18 u18, U19 u19, U20 u20, U21 u21, U22 u22, U23 u23, U24 u24, U25 u25, U26 u26, U27 u27, U28 u28, U29 u29, U30 u30, U31 u31, U32 u32, U33 u33, U34 u34, U35 u35, U36 u36, U37 u37, U38 u38))
{
   set(Rcpp::XPtr< Rcpp::CppFunction39<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35, U36, U37, U38> >(new Rcpp::CppFunction39<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35, U36, U37, U38>(fun), false)) ;
}




template <typename OUT, typename U0, typename U1, typename U2, typename U3, typename U4, typename U5, typename U6, typename U7, typename U8, typename U9, typename U10, typename U11, typename U12, typename U13, typename U14, typename U15, typename U16, typename U17, typename U18, typename U19, typename U20, typename U21, typename U22, typename U23, typename U24, typename U25, typename U26, typename U27, typename U28, typename U29, typename U30, typename U31, typename U32, typename U33, typename U34, typename U35, typename U36, typename U37, typename U38, typename U39>
TRInternalFunction_Impl(OUT(*fun)(U0 u0, U1 u1, U2 u2, U3 u3, U4 u4, U5 u5, U6 u6, U7 u7, U8 u8, U9 u9, U10 u10, U11 u11, U12 u12, U13 u13, U14 u14, U15 u15, U16 u16, U17 u17, U18 u18, U19 u19, U20 u20, U21 u21, U22 u22, U23 u23, U24 u24, U25 u25, U26 u26, U27 u27, U28 u28, U29 u29, U30 u30, U31 u31, U32 u32, U33 u33, U34 u34, U35 u35, U36 u36, U37 u37, U38 u38, U39 u39))
{
   set(Rcpp::XPtr< Rcpp::CppFunction40<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35, U36, U37, U38, U39> >(new Rcpp::CppFunction40<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35, U36, U37, U38, U39>(fun), false)) ;
}




template <typename OUT, typename U0, typename U1, typename U2, typename U3, typename U4, typename U5, typename U6, typename U7, typename U8, typename U9, typename U10, typename U11, typename U12, typename U13, typename U14, typename U15, typename U16, typename U17, typename U18, typename U19, typename U20, typename U21, typename U22, typename U23, typename U24, typename U25, typename U26, typename U27, typename U28, typename U29, typename U30, typename U31, typename U32, typename U33, typename U34, typename U35, typename U36, typename U37, typename U38, typename U39, typename U40>
TRInternalFunction_Impl(OUT(*fun)(U0 u0, U1 u1, U2 u2, U3 u3, U4 u4, U5 u5, U6 u6, U7 u7, U8 u8, U9 u9, U10 u10, U11 u11, U12 u12, U13 u13, U14 u14, U15 u15, U16 u16, U17 u17, U18 u18, U19 u19, U20 u20, U21 u21, U22 u22, U23 u23, U24 u24, U25 u25, U26 u26, U27 u27, U28 u28, U29 u29, U30 u30, U31 u31, U32 u32, U33 u33, U34 u34, U35 u35, U36 u36, U37 u37, U38 u38, U39 u39, U40 u40))
{
   set(Rcpp::XPtr< Rcpp::CppFunction41<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35, U36, U37, U38, U39, U40> >(new Rcpp::CppFunction41<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35, U36, U37, U38, U39, U40>(fun), false)) ;
}




template <typename OUT, typename U0, typename U1, typename U2, typename U3, typename U4, typename U5, typename U6, typename U7, typename U8, typename U9, typename U10, typename U11, typename U12, typename U13, typename U14, typename U15, typename U16, typename U17, typename U18, typename U19, typename U20, typename U21, typename U22, typename U23, typename U24, typename U25, typename U26, typename U27, typename U28, typename U29, typename U30, typename U31, typename U32, typename U33, typename U34, typename U35, typename U36, typename U37, typename U38, typename U39, typename U40, typename U41>
TRInternalFunction_Impl(OUT(*fun)(U0 u0, U1 u1, U2 u2, U3 u3, U4 u4, U5 u5, U6 u6, U7 u7, U8 u8, U9 u9, U10 u10, U11 u11, U12 u12, U13 u13, U14 u14, U15 u15, U16 u16, U17 u17, U18 u18, U19 u19, U20 u20, U21 u21, U22 u22, U23 u23, U24 u24, U25 u25, U26 u26, U27 u27, U28 u28, U29 u29, U30 u30, U31 u31, U32 u32, U33 u33, U34 u34, U35 u35, U36 u36, U37 u37, U38 u38, U39 u39, U40 u40, U41 u41))
{
   set(Rcpp::XPtr< Rcpp::CppFunction42<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35, U36, U37, U38, U39, U40, U41> >(new Rcpp::CppFunction42<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35, U36, U37, U38, U39, U40, U41>(fun), false)) ;
}




template <typename OUT, typename U0, typename U1, typename U2, typename U3, typename U4, typename U5, typename U6, typename U7, typename U8, typename U9, typename U10, typename U11, typename U12, typename U13, typename U14, typename U15, typename U16, typename U17, typename U18, typename U19, typename U20, typename U21, typename U22, typename U23, typename U24, typename U25, typename U26, typename U27, typename U28, typename U29, typename U30, typename U31, typename U32, typename U33, typename U34, typename U35, typename U36, typename U37, typename U38, typename U39, typename U40, typename U41, typename U42>
TRInternalFunction_Impl(OUT(*fun)(U0 u0, U1 u1, U2 u2, U3 u3, U4 u4, U5 u5, U6 u6, U7 u7, U8 u8, U9 u9, U10 u10, U11 u11, U12 u12, U13 u13, U14 u14, U15 u15, U16 u16, U17 u17, U18 u18, U19 u19, U20 u20, U21 u21, U22 u22, U23 u23, U24 u24, U25 u25, U26 u26, U27 u27, U28 u28, U29 u29, U30 u30, U31 u31, U32 u32, U33 u33, U34 u34, U35 u35, U36 u36, U37 u37, U38 u38, U39 u39, U40 u40, U41 u41, U42 u42))
{
   set(Rcpp::XPtr< Rcpp::CppFunction43<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35, U36, U37, U38, U39, U40, U41, U42> >(new Rcpp::CppFunction43<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35, U36, U37, U38, U39, U40, U41, U42>(fun), false)) ;
}




template <typename OUT, typename U0, typename U1, typename U2, typename U3, typename U4, typename U5, typename U6, typename U7, typename U8, typename U9, typename U10, typename U11, typename U12, typename U13, typename U14, typename U15, typename U16, typename U17, typename U18, typename U19, typename U20, typename U21, typename U22, typename U23, typename U24, typename U25, typename U26, typename U27, typename U28, typename U29, typename U30, typename U31, typename U32, typename U33, typename U34, typename U35, typename U36, typename U37, typename U38, typename U39, typename U40, typename U41, typename U42, typename U43>
TRInternalFunction_Impl(OUT(*fun)(U0 u0, U1 u1, U2 u2, U3 u3, U4 u4, U5 u5, U6 u6, U7 u7, U8 u8, U9 u9, U10 u10, U11 u11, U12 u12, U13 u13, U14 u14, U15 u15, U16 u16, U17 u17, U18 u18, U19 u19, U20 u20, U21 u21, U22 u22, U23 u23, U24 u24, U25 u25, U26 u26, U27 u27, U28 u28, U29 u29, U30 u30, U31 u31, U32 u32, U33 u33, U34 u34, U35 u35, U36 u36, U37 u37, U38 u38, U39 u39, U40 u40, U41 u41, U42 u42, U43 u43))
{
   set(Rcpp::XPtr< Rcpp::CppFunction44<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35, U36, U37, U38, U39, U40, U41, U42, U43> >(new Rcpp::CppFunction44<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35, U36, U37, U38, U39, U40, U41, U42, U43>(fun), false)) ;
}




template <typename OUT, typename U0, typename U1, typename U2, typename U3, typename U4, typename U5, typename U6, typename U7, typename U8, typename U9, typename U10, typename U11, typename U12, typename U13, typename U14, typename U15, typename U16, typename U17, typename U18, typename U19, typename U20, typename U21, typename U22, typename U23, typename U24, typename U25, typename U26, typename U27, typename U28, typename U29, typename U30, typename U31, typename U32, typename U33, typename U34, typename U35, typename U36, typename U37, typename U38, typename U39, typename U40, typename U41, typename U42, typename U43, typename U44>
TRInternalFunction_Impl(OUT(*fun)(U0 u0, U1 u1, U2 u2, U3 u3, U4 u4, U5 u5, U6 u6, U7 u7, U8 u8, U9 u9, U10 u10, U11 u11, U12 u12, U13 u13, U14 u14, U15 u15, U16 u16, U17 u17, U18 u18, U19 u19, U20 u20, U21 u21, U22 u22, U23 u23, U24 u24, U25 u25, U26 u26, U27 u27, U28 u28, U29 u29, U30 u30, U31 u31, U32 u32, U33 u33, U34 u34, U35 u35, U36 u36, U37 u37, U38 u38, U39 u39, U40 u40, U41 u41, U42 u42, U43 u43, U44 u44))
{
   set(Rcpp::XPtr< Rcpp::CppFunction45<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35, U36, U37, U38, U39, U40, U41, U42, U43, U44> >(new Rcpp::CppFunction45<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35, U36, U37, U38, U39, U40, U41, U42, U43, U44>(fun), false)) ;
}




template <typename OUT, typename U0, typename U1, typename U2, typename U3, typename U4, typename U5, typename U6, typename U7, typename U8, typename U9, typename U10, typename U11, typename U12, typename U13, typename U14, typename U15, typename U16, typename U17, typename U18, typename U19, typename U20, typename U21, typename U22, typename U23, typename U24, typename U25, typename U26, typename U27, typename U28, typename U29, typename U30, typename U31, typename U32, typename U33, typename U34, typename U35, typename U36, typename U37, typename U38, typename U39, typename U40, typename U41, typename U42, typename U43, typename U44, typename U45>
TRInternalFunction_Impl(OUT(*fun)(U0 u0, U1 u1, U2 u2, U3 u3, U4 u4, U5 u5, U6 u6, U7 u7, U8 u8, U9 u9, U10 u10, U11 u11, U12 u12, U13 u13, U14 u14, U15 u15, U16 u16, U17 u17, U18 u18, U19 u19, U20 u20, U21 u21, U22 u22, U23 u23, U24 u24, U25 u25, U26 u26, U27 u27, U28 u28, U29 u29, U30 u30, U31 u31, U32 u32, U33 u33, U34 u34, U35 u35, U36 u36, U37 u37, U38 u38, U39 u39, U40 u40, U41 u41, U42 u42, U43 u43, U44 u44, U45 u45))
{
   set(Rcpp::XPtr< Rcpp::CppFunction46<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35, U36, U37, U38, U39, U40, U41, U42, U43, U44, U45> >(new Rcpp::CppFunction46<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35, U36, U37, U38, U39, U40, U41, U42, U43, U44, U45>(fun), false)) ;
}




template <typename OUT, typename U0, typename U1, typename U2, typename U3, typename U4, typename U5, typename U6, typename U7, typename U8, typename U9, typename U10, typename U11, typename U12, typename U13, typename U14, typename U15, typename U16, typename U17, typename U18, typename U19, typename U20, typename U21, typename U22, typename U23, typename U24, typename U25, typename U26, typename U27, typename U28, typename U29, typename U30, typename U31, typename U32, typename U33, typename U34, typename U35, typename U36, typename U37, typename U38, typename U39, typename U40, typename U41, typename U42, typename U43, typename U44, typename U45, typename U46>
TRInternalFunction_Impl(OUT(*fun)(U0 u0, U1 u1, U2 u2, U3 u3, U4 u4, U5 u5, U6 u6, U7 u7, U8 u8, U9 u9, U10 u10, U11 u11, U12 u12, U13 u13, U14 u14, U15 u15, U16 u16, U17 u17, U18 u18, U19 u19, U20 u20, U21 u21, U22 u22, U23 u23, U24 u24, U25 u25, U26 u26, U27 u27, U28 u28, U29 u29, U30 u30, U31 u31, U32 u32, U33 u33, U34 u34, U35 u35, U36 u36, U37 u37, U38 u38, U39 u39, U40 u40, U41 u41, U42 u42, U43 u43, U44 u44, U45 u45, U46 u46))
{
   set(Rcpp::XPtr< Rcpp::CppFunction47<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35, U36, U37, U38, U39, U40, U41, U42, U43, U44, U45, U46> >(new Rcpp::CppFunction47<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35, U36, U37, U38, U39, U40, U41, U42, U43, U44, U45, U46>(fun), false)) ;
}




template <typename OUT, typename U0, typename U1, typename U2, typename U3, typename U4, typename U5, typename U6, typename U7, typename U8, typename U9, typename U10, typename U11, typename U12, typename U13, typename U14, typename U15, typename U16, typename U17, typename U18, typename U19, typename U20, typename U21, typename U22, typename U23, typename U24, typename U25, typename U26, typename U27, typename U28, typename U29, typename U30, typename U31, typename U32, typename U33, typename U34, typename U35, typename U36, typename U37, typename U38, typename U39, typename U40, typename U41, typename U42, typename U43, typename U44, typename U45, typename U46, typename U47>
TRInternalFunction_Impl(OUT(*fun)(U0 u0, U1 u1, U2 u2, U3 u3, U4 u4, U5 u5, U6 u6, U7 u7, U8 u8, U9 u9, U10 u10, U11 u11, U12 u12, U13 u13, U14 u14, U15 u15, U16 u16, U17 u17, U18 u18, U19 u19, U20 u20, U21 u21, U22 u22, U23 u23, U24 u24, U25 u25, U26 u26, U27 u27, U28 u28, U29 u29, U30 u30, U31 u31, U32 u32, U33 u33, U34 u34, U35 u35, U36 u36, U37 u37, U38 u38, U39 u39, U40 u40, U41 u41, U42 u42, U43 u43, U44 u44, U45 u45, U46 u46, U47 u47))
{
   set(Rcpp::XPtr< Rcpp::CppFunction48<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35, U36, U37, U38, U39, U40, U41, U42, U43, U44, U45, U46, U47> >(new Rcpp::CppFunction48<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35, U36, U37, U38, U39, U40, U41, U42, U43, U44, U45, U46, U47>(fun), false)) ;
}




template <typename OUT, typename U0, typename U1, typename U2, typename U3, typename U4, typename U5, typename U6, typename U7, typename U8, typename U9, typename U10, typename U11, typename U12, typename U13, typename U14, typename U15, typename U16, typename U17, typename U18, typename U19, typename U20, typename U21, typename U22, typename U23, typename U24, typename U25, typename U26, typename U27, typename U28, typename U29, typename U30, typename U31, typename U32, typename U33, typename U34, typename U35, typename U36, typename U37, typename U38, typename U39, typename U40, typename U41, typename U42, typename U43, typename U44, typename U45, typename U46, typename U47, typename U48>
TRInternalFunction_Impl(OUT(*fun)(U0 u0, U1 u1, U2 u2, U3 u3, U4 u4, U5 u5, U6 u6, U7 u7, U8 u8, U9 u9, U10 u10, U11 u11, U12 u12, U13 u13, U14 u14, U15 u15, U16 u16, U17 u17, U18 u18, U19 u19, U20 u20, U21 u21, U22 u22, U23 u23, U24 u24, U25 u25, U26 u26, U27 u27, U28 u28, U29 u29, U30 u30, U31 u31, U32 u32, U33 u33, U34 u34, U35 u35, U36 u36, U37 u37, U38 u38, U39 u39, U40 u40, U41 u41, U42 u42, U43 u43, U44 u44, U45 u45, U46 u46, U47 u47, U48 u48))
{
   set(Rcpp::XPtr< Rcpp::CppFunction49<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35, U36, U37, U38, U39, U40, U41, U42, U43, U44, U45, U46, U47, U48> >(new Rcpp::CppFunction49<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35, U36, U37, U38, U39, U40, U41, U42, U43, U44, U45, U46, U47, U48>(fun), false)) ;
}




template <typename OUT, typename U0, typename U1, typename U2, typename U3, typename U4, typename U5, typename U6, typename U7, typename U8, typename U9, typename U10, typename U11, typename U12, typename U13, typename U14, typename U15, typename U16, typename U17, typename U18, typename U19, typename U20, typename U21, typename U22, typename U23, typename U24, typename U25, typename U26, typename U27, typename U28, typename U29, typename U30, typename U31, typename U32, typename U33, typename U34, typename U35, typename U36, typename U37, typename U38, typename U39, typename U40, typename U41, typename U42, typename U43, typename U44, typename U45, typename U46, typename U47, typename U48, typename U49>
TRInternalFunction_Impl(OUT(*fun)(U0 u0, U1 u1, U2 u2, U3 u3, U4 u4, U5 u5, U6 u6, U7 u7, U8 u8, U9 u9, U10 u10, U11 u11, U12 u12, U13 u13, U14 u14, U15 u15, U16 u16, U17 u17, U18 u18, U19 u19, U20 u20, U21 u21, U22 u22, U23 u23, U24 u24, U25 u25, U26 u26, U27 u27, U28 u28, U29 u29, U30 u30, U31 u31, U32 u32, U33 u33, U34 u34, U35 u35, U36 u36, U37 u37, U38 u38, U39 u39, U40 u40, U41 u41, U42 u42, U43 u43, U44 u44, U45 u45, U46 u46, U47 u47, U48 u48, U49 u49))
{
   set(Rcpp::XPtr< Rcpp::CppFunction50<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35, U36, U37, U38, U39, U40, U41, U42, U43, U44, U45, U46, U47, U48, U49> >(new Rcpp::CppFunction50<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35, U36, U37, U38, U39, U40, U41, U42, U43, U44, U45, U46, U47, U48, U49>(fun), false)) ;
}




template <typename OUT, typename U0, typename U1, typename U2, typename U3, typename U4, typename U5, typename U6, typename U7, typename U8, typename U9, typename U10, typename U11, typename U12, typename U13, typename U14, typename U15, typename U16, typename U17, typename U18, typename U19, typename U20, typename U21, typename U22, typename U23, typename U24, typename U25, typename U26, typename U27, typename U28, typename U29, typename U30, typename U31, typename U32, typename U33, typename U34, typename U35, typename U36, typename U37, typename U38, typename U39, typename U40, typename U41, typename U42, typename U43, typename U44, typename U45, typename U46, typename U47, typename U48, typename U49, typename U50>
TRInternalFunction_Impl(OUT(*fun)(U0 u0, U1 u1, U2 u2, U3 u3, U4 u4, U5 u5, U6 u6, U7 u7, U8 u8, U9 u9, U10 u10, U11 u11, U12 u12, U13 u13, U14 u14, U15 u15, U16 u16, U17 u17, U18 u18, U19 u19, U20 u20, U21 u21, U22 u22, U23 u23, U24 u24, U25 u25, U26 u26, U27 u27, U28 u28, U29 u29, U30 u30, U31 u31, U32 u32, U33 u33, U34 u34, U35 u35, U36 u36, U37 u37, U38 u38, U39 u39, U40 u40, U41 u41, U42 u42, U43 u43, U44 u44, U45 u45, U46 u46, U47 u47, U48 u48, U49 u49, U50 u50))
{
   set(Rcpp::XPtr< Rcpp::CppFunction51<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35, U36, U37, U38, U39, U40, U41, U42, U43, U44, U45, U46, U47, U48, U49, U50> >(new Rcpp::CppFunction51<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35, U36, U37, U38, U39, U40, U41, U42, U43, U44, U45, U46, U47, U48, U49, U50>(fun), false)) ;
}




template <typename OUT, typename U0, typename U1, typename U2, typename U3, typename U4, typename U5, typename U6, typename U7, typename U8, typename U9, typename U10, typename U11, typename U12, typename U13, typename U14, typename U15, typename U16, typename U17, typename U18, typename U19, typename U20, typename U21, typename U22, typename U23, typename U24, typename U25, typename U26, typename U27, typename U28, typename U29, typename U30, typename U31, typename U32, typename U33, typename U34, typename U35, typename U36, typename U37, typename U38, typename U39, typename U40, typename U41, typename U42, typename U43, typename U44, typename U45, typename U46, typename U47, typename U48, typename U49, typename U50, typename U51>
TRInternalFunction_Impl(OUT(*fun)(U0 u0, U1 u1, U2 u2, U3 u3, U4 u4, U5 u5, U6 u6, U7 u7, U8 u8, U9 u9, U10 u10, U11 u11, U12 u12, U13 u13, U14 u14, U15 u15, U16 u16, U17 u17, U18 u18, U19 u19, U20 u20, U21 u21, U22 u22, U23 u23, U24 u24, U25 u25, U26 u26, U27 u27, U28 u28, U29 u29, U30 u30, U31 u31, U32 u32, U33 u33, U34 u34, U35 u35, U36 u36, U37 u37, U38 u38, U39 u39, U40 u40, U41 u41, U42 u42, U43 u43, U44 u44, U45 u45, U46 u46, U47 u47, U48 u48, U49 u49, U50 u50, U51 u51))
{
   set(Rcpp::XPtr< Rcpp::CppFunction52<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35, U36, U37, U38, U39, U40, U41, U42, U43, U44, U45, U46, U47, U48, U49, U50, U51> >(new Rcpp::CppFunction52<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35, U36, U37, U38, U39, U40, U41, U42, U43, U44, U45, U46, U47, U48, U49, U50, U51>(fun), false)) ;
}




template <typename OUT, typename U0, typename U1, typename U2, typename U3, typename U4, typename U5, typename U6, typename U7, typename U8, typename U9, typename U10, typename U11, typename U12, typename U13, typename U14, typename U15, typename U16, typename U17, typename U18, typename U19, typename U20, typename U21, typename U22, typename U23, typename U24, typename U25, typename U26, typename U27, typename U28, typename U29, typename U30, typename U31, typename U32, typename U33, typename U34, typename U35, typename U36, typename U37, typename U38, typename U39, typename U40, typename U41, typename U42, typename U43, typename U44, typename U45, typename U46, typename U47, typename U48, typename U49, typename U50, typename U51, typename U52>
TRInternalFunction_Impl(OUT(*fun)(U0 u0, U1 u1, U2 u2, U3 u3, U4 u4, U5 u5, U6 u6, U7 u7, U8 u8, U9 u9, U10 u10, U11 u11, U12 u12, U13 u13, U14 u14, U15 u15, U16 u16, U17 u17, U18 u18, U19 u19, U20 u20, U21 u21, U22 u22, U23 u23, U24 u24, U25 u25, U26 u26, U27 u27, U28 u28, U29 u29, U30 u30, U31 u31, U32 u32, U33 u33, U34 u34, U35 u35, U36 u36, U37 u37, U38 u38, U39 u39, U40 u40, U41 u41, U42 u42, U43 u43, U44 u44, U45 u45, U46 u46, U47 u47, U48 u48, U49 u49, U50 u50, U51 u51, U52 u52))
{
   set(Rcpp::XPtr< Rcpp::CppFunction53<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35, U36, U37, U38, U39, U40, U41, U42, U43, U44, U45, U46, U47, U48, U49, U50, U51, U52> >(new Rcpp::CppFunction53<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35, U36, U37, U38, U39, U40, U41, U42, U43, U44, U45, U46, U47, U48, U49, U50, U51, U52>(fun), false)) ;
}




template <typename OUT, typename U0, typename U1, typename U2, typename U3, typename U4, typename U5, typename U6, typename U7, typename U8, typename U9, typename U10, typename U11, typename U12, typename U13, typename U14, typename U15, typename U16, typename U17, typename U18, typename U19, typename U20, typename U21, typename U22, typename U23, typename U24, typename U25, typename U26, typename U27, typename U28, typename U29, typename U30, typename U31, typename U32, typename U33, typename U34, typename U35, typename U36, typename U37, typename U38, typename U39, typename U40, typename U41, typename U42, typename U43, typename U44, typename U45, typename U46, typename U47, typename U48, typename U49, typename U50, typename U51, typename U52, typename U53>
TRInternalFunction_Impl(OUT(*fun)(U0 u0, U1 u1, U2 u2, U3 u3, U4 u4, U5 u5, U6 u6, U7 u7, U8 u8, U9 u9, U10 u10, U11 u11, U12 u12, U13 u13, U14 u14, U15 u15, U16 u16, U17 u17, U18 u18, U19 u19, U20 u20, U21 u21, U22 u22, U23 u23, U24 u24, U25 u25, U26 u26, U27 u27, U28 u28, U29 u29, U30 u30, U31 u31, U32 u32, U33 u33, U34 u34, U35 u35, U36 u36, U37 u37, U38 u38, U39 u39, U40 u40, U41 u41, U42 u42, U43 u43, U44 u44, U45 u45, U46 u46, U47 u47, U48 u48, U49 u49, U50 u50, U51 u51, U52 u52, U53 u53))
{
   set(Rcpp::XPtr< Rcpp::CppFunction54<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35, U36, U37, U38, U39, U40, U41, U42, U43, U44, U45, U46, U47, U48, U49, U50, U51, U52, U53> >(new Rcpp::CppFunction54<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35, U36, U37, U38, U39, U40, U41, U42, U43, U44, U45, U46, U47, U48, U49, U50, U51, U52, U53>(fun), false)) ;
}




template <typename OUT, typename U0, typename U1, typename U2, typename U3, typename U4, typename U5, typename U6, typename U7, typename U8, typename U9, typename U10, typename U11, typename U12, typename U13, typename U14, typename U15, typename U16, typename U17, typename U18, typename U19, typename U20, typename U21, typename U22, typename U23, typename U24, typename U25, typename U26, typename U27, typename U28, typename U29, typename U30, typename U31, typename U32, typename U33, typename U34, typename U35, typename U36, typename U37, typename U38, typename U39, typename U40, typename U41, typename U42, typename U43, typename U44, typename U45, typename U46, typename U47, typename U48, typename U49, typename U50, typename U51, typename U52, typename U53, typename U54>
TRInternalFunction_Impl(OUT(*fun)(U0 u0, U1 u1, U2 u2, U3 u3, U4 u4, U5 u5, U6 u6, U7 u7, U8 u8, U9 u9, U10 u10, U11 u11, U12 u12, U13 u13, U14 u14, U15 u15, U16 u16, U17 u17, U18 u18, U19 u19, U20 u20, U21 u21, U22 u22, U23 u23, U24 u24, U25 u25, U26 u26, U27 u27, U28 u28, U29 u29, U30 u30, U31 u31, U32 u32, U33 u33, U34 u34, U35 u35, U36 u36, U37 u37, U38 u38, U39 u39, U40 u40, U41 u41, U42 u42, U43 u43, U44 u44, U45 u45, U46 u46, U47 u47, U48 u48, U49 u49, U50 u50, U51 u51, U52 u52, U53 u53, U54 u54))
{
   set(Rcpp::XPtr< Rcpp::CppFunction55<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35, U36, U37, U38, U39, U40, U41, U42, U43, U44, U45, U46, U47, U48, U49, U50, U51, U52, U53, U54> >(new Rcpp::CppFunction55<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35, U36, U37, U38, U39, U40, U41, U42, U43, U44, U45, U46, U47, U48, U49, U50, U51, U52, U53, U54>(fun), false)) ;
}




template <typename OUT, typename U0, typename U1, typename U2, typename U3, typename U4, typename U5, typename U6, typename U7, typename U8, typename U9, typename U10, typename U11, typename U12, typename U13, typename U14, typename U15, typename U16, typename U17, typename U18, typename U19, typename U20, typename U21, typename U22, typename U23, typename U24, typename U25, typename U26, typename U27, typename U28, typename U29, typename U30, typename U31, typename U32, typename U33, typename U34, typename U35, typename U36, typename U37, typename U38, typename U39, typename U40, typename U41, typename U42, typename U43, typename U44, typename U45, typename U46, typename U47, typename U48, typename U49, typename U50, typename U51, typename U52, typename U53, typename U54, typename U55>
TRInternalFunction_Impl(OUT(*fun)(U0 u0, U1 u1, U2 u2, U3 u3, U4 u4, U5 u5, U6 u6, U7 u7, U8 u8, U9 u9, U10 u10, U11 u11, U12 u12, U13 u13, U14 u14, U15 u15, U16 u16, U17 u17, U18 u18, U19 u19, U20 u20, U21 u21, U22 u22, U23 u23, U24 u24, U25 u25, U26 u26, U27 u27, U28 u28, U29 u29, U30 u30, U31 u31, U32 u32, U33 u33, U34 u34, U35 u35, U36 u36, U37 u37, U38 u38, U39 u39, U40 u40, U41 u41, U42 u42, U43 u43, U44 u44, U45 u45, U46 u46, U47 u47, U48 u48, U49 u49, U50 u50, U51 u51, U52 u52, U53 u53, U54 u54, U55 u55))
{
   set(Rcpp::XPtr< Rcpp::CppFunction56<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35, U36, U37, U38, U39, U40, U41, U42, U43, U44, U45, U46, U47, U48, U49, U50, U51, U52, U53, U54, U55> >(new Rcpp::CppFunction56<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35, U36, U37, U38, U39, U40, U41, U42, U43, U44, U45, U46, U47, U48, U49, U50, U51, U52, U53, U54, U55>(fun), false)) ;
}




template <typename OUT, typename U0, typename U1, typename U2, typename U3, typename U4, typename U5, typename U6, typename U7, typename U8, typename U9, typename U10, typename U11, typename U12, typename U13, typename U14, typename U15, typename U16, typename U17, typename U18, typename U19, typename U20, typename U21, typename U22, typename U23, typename U24, typename U25, typename U26, typename U27, typename U28, typename U29, typename U30, typename U31, typename U32, typename U33, typename U34, typename U35, typename U36, typename U37, typename U38, typename U39, typename U40, typename U41, typename U42, typename U43, typename U44, typename U45, typename U46, typename U47, typename U48, typename U49, typename U50, typename U51, typename U52, typename U53, typename U54, typename U55, typename U56>
TRInternalFunction_Impl(OUT(*fun)(U0 u0, U1 u1, U2 u2, U3 u3, U4 u4, U5 u5, U6 u6, U7 u7, U8 u8, U9 u9, U10 u10, U11 u11, U12 u12, U13 u13, U14 u14, U15 u15, U16 u16, U17 u17, U18 u18, U19 u19, U20 u20, U21 u21, U22 u22, U23 u23, U24 u24, U25 u25, U26 u26, U27 u27, U28 u28, U29 u29, U30 u30, U31 u31, U32 u32, U33 u33, U34 u34, U35 u35, U36 u36, U37 u37, U38 u38, U39 u39, U40 u40, U41 u41, U42 u42, U43 u43, U44 u44, U45 u45, U46 u46, U47 u47, U48 u48, U49 u49, U50 u50, U51 u51, U52 u52, U53 u53, U54 u54, U55 u55, U56 u56))
{
   set(Rcpp::XPtr< Rcpp::CppFunction57<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35, U36, U37, U38, U39, U40, U41, U42, U43, U44, U45, U46, U47, U48, U49, U50, U51, U52, U53, U54, U55, U56> >(new Rcpp::CppFunction57<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35, U36, U37, U38, U39, U40, U41, U42, U43, U44, U45, U46, U47, U48, U49, U50, U51, U52, U53, U54, U55, U56>(fun), false)) ;
}




template <typename OUT, typename U0, typename U1, typename U2, typename U3, typename U4, typename U5, typename U6, typename U7, typename U8, typename U9, typename U10, typename U11, typename U12, typename U13, typename U14, typename U15, typename U16, typename U17, typename U18, typename U19, typename U20, typename U21, typename U22, typename U23, typename U24, typename U25, typename U26, typename U27, typename U28, typename U29, typename U30, typename U31, typename U32, typename U33, typename U34, typename U35, typename U36, typename U37, typename U38, typename U39, typename U40, typename U41, typename U42, typename U43, typename U44, typename U45, typename U46, typename U47, typename U48, typename U49, typename U50, typename U51, typename U52, typename U53, typename U54, typename U55, typename U56, typename U57>
TRInternalFunction_Impl(OUT(*fun)(U0 u0, U1 u1, U2 u2, U3 u3, U4 u4, U5 u5, U6 u6, U7 u7, U8 u8, U9 u9, U10 u10, U11 u11, U12 u12, U13 u13, U14 u14, U15 u15, U16 u16, U17 u17, U18 u18, U19 u19, U20 u20, U21 u21, U22 u22, U23 u23, U24 u24, U25 u25, U26 u26, U27 u27, U28 u28, U29 u29, U30 u30, U31 u31, U32 u32, U33 u33, U34 u34, U35 u35, U36 u36, U37 u37, U38 u38, U39 u39, U40 u40, U41 u41, U42 u42, U43 u43, U44 u44, U45 u45, U46 u46, U47 u47, U48 u48, U49 u49, U50 u50, U51 u51, U52 u52, U53 u53, U54 u54, U55 u55, U56 u56, U57 u57))
{
   set(Rcpp::XPtr< Rcpp::CppFunction58<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35, U36, U37, U38, U39, U40, U41, U42, U43, U44, U45, U46, U47, U48, U49, U50, U51, U52, U53, U54, U55, U56, U57> >(new Rcpp::CppFunction58<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35, U36, U37, U38, U39, U40, U41, U42, U43, U44, U45, U46, U47, U48, U49, U50, U51, U52, U53, U54, U55, U56, U57>(fun), false)) ;
}




template <typename OUT, typename U0, typename U1, typename U2, typename U3, typename U4, typename U5, typename U6, typename U7, typename U8, typename U9, typename U10, typename U11, typename U12, typename U13, typename U14, typename U15, typename U16, typename U17, typename U18, typename U19, typename U20, typename U21, typename U22, typename U23, typename U24, typename U25, typename U26, typename U27, typename U28, typename U29, typename U30, typename U31, typename U32, typename U33, typename U34, typename U35, typename U36, typename U37, typename U38, typename U39, typename U40, typename U41, typename U42, typename U43, typename U44, typename U45, typename U46, typename U47, typename U48, typename U49, typename U50, typename U51, typename U52, typename U53, typename U54, typename U55, typename U56, typename U57, typename U58>
TRInternalFunction_Impl(OUT(*fun)(U0 u0, U1 u1, U2 u2, U3 u3, U4 u4, U5 u5, U6 u6, U7 u7, U8 u8, U9 u9, U10 u10, U11 u11, U12 u12, U13 u13, U14 u14, U15 u15, U16 u16, U17 u17, U18 u18, U19 u19, U20 u20, U21 u21, U22 u22, U23 u23, U24 u24, U25 u25, U26 u26, U27 u27, U28 u28, U29 u29, U30 u30, U31 u31, U32 u32, U33 u33, U34 u34, U35 u35, U36 u36, U37 u37, U38 u38, U39 u39, U40 u40, U41 u41, U42 u42, U43 u43, U44 u44, U45 u45, U46 u46, U47 u47, U48 u48, U49 u49, U50 u50, U51 u51, U52 u52, U53 u53, U54 u54, U55 u55, U56 u56, U57 u57, U58 u58))
{
   set(Rcpp::XPtr< Rcpp::CppFunction59<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35, U36, U37, U38, U39, U40, U41, U42, U43, U44, U45, U46, U47, U48, U49, U50, U51, U52, U53, U54, U55, U56, U57, U58> >(new Rcpp::CppFunction59<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35, U36, U37, U38, U39, U40, U41, U42, U43, U44, U45, U46, U47, U48, U49, U50, U51, U52, U53, U54, U55, U56, U57, U58>(fun), false)) ;
}




template <typename OUT, typename U0, typename U1, typename U2, typename U3, typename U4, typename U5, typename U6, typename U7, typename U8, typename U9, typename U10, typename U11, typename U12, typename U13, typename U14, typename U15, typename U16, typename U17, typename U18, typename U19, typename U20, typename U21, typename U22, typename U23, typename U24, typename U25, typename U26, typename U27, typename U28, typename U29, typename U30, typename U31, typename U32, typename U33, typename U34, typename U35, typename U36, typename U37, typename U38, typename U39, typename U40, typename U41, typename U42, typename U43, typename U44, typename U45, typename U46, typename U47, typename U48, typename U49, typename U50, typename U51, typename U52, typename U53, typename U54, typename U55, typename U56, typename U57, typename U58, typename U59>
TRInternalFunction_Impl(OUT(*fun)(U0 u0, U1 u1, U2 u2, U3 u3, U4 u4, U5 u5, U6 u6, U7 u7, U8 u8, U9 u9, U10 u10, U11 u11, U12 u12, U13 u13, U14 u14, U15 u15, U16 u16, U17 u17, U18 u18, U19 u19, U20 u20, U21 u21, U22 u22, U23 u23, U24 u24, U25 u25, U26 u26, U27 u27, U28 u28, U29 u29, U30 u30, U31 u31, U32 u32, U33 u33, U34 u34, U35 u35, U36 u36, U37 u37, U38 u38, U39 u39, U40 u40, U41 u41, U42 u42, U43 u43, U44 u44, U45 u45, U46 u46, U47 u47, U48 u48, U49 u49, U50 u50, U51 u51, U52 u52, U53 u53, U54 u54, U55 u55, U56 u56, U57 u57, U58 u58, U59 u59))
{
   set(Rcpp::XPtr< Rcpp::CppFunction60<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35, U36, U37, U38, U39, U40, U41, U42, U43, U44, U45, U46, U47, U48, U49, U50, U51, U52, U53, U54, U55, U56, U57, U58, U59> >(new Rcpp::CppFunction60<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35, U36, U37, U38, U39, U40, U41, U42, U43, U44, U45, U46, U47, U48, U49, U50, U51, U52, U53, U54, U55, U56, U57, U58, U59>(fun), false)) ;
}




template <typename OUT, typename U0, typename U1, typename U2, typename U3, typename U4, typename U5, typename U6, typename U7, typename U8, typename U9, typename U10, typename U11, typename U12, typename U13, typename U14, typename U15, typename U16, typename U17, typename U18, typename U19, typename U20, typename U21, typename U22, typename U23, typename U24, typename U25, typename U26, typename U27, typename U28, typename U29, typename U30, typename U31, typename U32, typename U33, typename U34, typename U35, typename U36, typename U37, typename U38, typename U39, typename U40, typename U41, typename U42, typename U43, typename U44, typename U45, typename U46, typename U47, typename U48, typename U49, typename U50, typename U51, typename U52, typename U53, typename U54, typename U55, typename U56, typename U57, typename U58, typename U59, typename U60>
TRInternalFunction_Impl(OUT(*fun)(U0 u0, U1 u1, U2 u2, U3 u3, U4 u4, U5 u5, U6 u6, U7 u7, U8 u8, U9 u9, U10 u10, U11 u11, U12 u12, U13 u13, U14 u14, U15 u15, U16 u16, U17 u17, U18 u18, U19 u19, U20 u20, U21 u21, U22 u22, U23 u23, U24 u24, U25 u25, U26 u26, U27 u27, U28 u28, U29 u29, U30 u30, U31 u31, U32 u32, U33 u33, U34 u34, U35 u35, U36 u36, U37 u37, U38 u38, U39 u39, U40 u40, U41 u41, U42 u42, U43 u43, U44 u44, U45 u45, U46 u46, U47 u47, U48 u48, U49 u49, U50 u50, U51 u51, U52 u52, U53 u53, U54 u54, U55 u55, U56 u56, U57 u57, U58 u58, U59 u59, U60 u60))
{
   set(Rcpp::XPtr< Rcpp::CppFunction61<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35, U36, U37, U38, U39, U40, U41, U42, U43, U44, U45, U46, U47, U48, U49, U50, U51, U52, U53, U54, U55, U56, U57, U58, U59, U60> >(new Rcpp::CppFunction61<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35, U36, U37, U38, U39, U40, U41, U42, U43, U44, U45, U46, U47, U48, U49, U50, U51, U52, U53, U54, U55, U56, U57, U58, U59, U60>(fun), false)) ;
}




template <typename OUT, typename U0, typename U1, typename U2, typename U3, typename U4, typename U5, typename U6, typename U7, typename U8, typename U9, typename U10, typename U11, typename U12, typename U13, typename U14, typename U15, typename U16, typename U17, typename U18, typename U19, typename U20, typename U21, typename U22, typename U23, typename U24, typename U25, typename U26, typename U27, typename U28, typename U29, typename U30, typename U31, typename U32, typename U33, typename U34, typename U35, typename U36, typename U37, typename U38, typename U39, typename U40, typename U41, typename U42, typename U43, typename U44, typename U45, typename U46, typename U47, typename U48, typename U49, typename U50, typename U51, typename U52, typename U53, typename U54, typename U55, typename U56, typename U57, typename U58, typename U59, typename U60, typename U61>
TRInternalFunction_Impl(OUT(*fun)(U0 u0, U1 u1, U2 u2, U3 u3, U4 u4, U5 u5, U6 u6, U7 u7, U8 u8, U9 u9, U10 u10, U11 u11, U12 u12, U13 u13, U14 u14, U15 u15, U16 u16, U17 u17, U18 u18, U19 u19, U20 u20, U21 u21, U22 u22, U23 u23, U24 u24, U25 u25, U26 u26, U27 u27, U28 u28, U29 u29, U30 u30, U31 u31, U32 u32, U33 u33, U34 u34, U35 u35, U36 u36, U37 u37, U38 u38, U39 u39, U40 u40, U41 u41, U42 u42, U43 u43, U44 u44, U45 u45, U46 u46, U47 u47, U48 u48, U49 u49, U50 u50, U51 u51, U52 u52, U53 u53, U54 u54, U55 u55, U56 u56, U57 u57, U58 u58, U59 u59, U60 u60, U61 u61))
{
   set(Rcpp::XPtr< Rcpp::CppFunction62<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35, U36, U37, U38, U39, U40, U41, U42, U43, U44, U45, U46, U47, U48, U49, U50, U51, U52, U53, U54, U55, U56, U57, U58, U59, U60, U61> >(new Rcpp::CppFunction62<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35, U36, U37, U38, U39, U40, U41, U42, U43, U44, U45, U46, U47, U48, U49, U50, U51, U52, U53, U54, U55, U56, U57, U58, U59, U60, U61>(fun), false)) ;
}




template <typename OUT, typename U0, typename U1, typename U2, typename U3, typename U4, typename U5, typename U6, typename U7, typename U8, typename U9, typename U10, typename U11, typename U12, typename U13, typename U14, typename U15, typename U16, typename U17, typename U18, typename U19, typename U20, typename U21, typename U22, typename U23, typename U24, typename U25, typename U26, typename U27, typename U28, typename U29, typename U30, typename U31, typename U32, typename U33, typename U34, typename U35, typename U36, typename U37, typename U38, typename U39, typename U40, typename U41, typename U42, typename U43, typename U44, typename U45, typename U46, typename U47, typename U48, typename U49, typename U50, typename U51, typename U52, typename U53, typename U54, typename U55, typename U56, typename U57, typename U58, typename U59, typename U60, typename U61, typename U62>
TRInternalFunction_Impl(OUT(*fun)(U0 u0, U1 u1, U2 u2, U3 u3, U4 u4, U5 u5, U6 u6, U7 u7, U8 u8, U9 u9, U10 u10, U11 u11, U12 u12, U13 u13, U14 u14, U15 u15, U16 u16, U17 u17, U18 u18, U19 u19, U20 u20, U21 u21, U22 u22, U23 u23, U24 u24, U25 u25, U26 u26, U27 u27, U28 u28, U29 u29, U30 u30, U31 u31, U32 u32, U33 u33, U34 u34, U35 u35, U36 u36, U37 u37, U38 u38, U39 u39, U40 u40, U41 u41, U42 u42, U43 u43, U44 u44, U45 u45, U46 u46, U47 u47, U48 u48, U49 u49, U50 u50, U51 u51, U52 u52, U53 u53, U54 u54, U55 u55, U56 u56, U57 u57, U58 u58, U59 u59, U60 u60, U61 u61, U62 u62))
{
   set(Rcpp::XPtr< Rcpp::CppFunction63<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35, U36, U37, U38, U39, U40, U41, U42, U43, U44, U45, U46, U47, U48, U49, U50, U51, U52, U53, U54, U55, U56, U57, U58, U59, U60, U61, U62> >(new Rcpp::CppFunction63<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35, U36, U37, U38, U39, U40, U41, U42, U43, U44, U45, U46, U47, U48, U49, U50, U51, U52, U53, U54, U55, U56, U57, U58, U59, U60, U61, U62>(fun), false)) ;
}




template <typename OUT, typename U0, typename U1, typename U2, typename U3, typename U4, typename U5, typename U6, typename U7, typename U8, typename U9, typename U10, typename U11, typename U12, typename U13, typename U14, typename U15, typename U16, typename U17, typename U18, typename U19, typename U20, typename U21, typename U22, typename U23, typename U24, typename U25, typename U26, typename U27, typename U28, typename U29, typename U30, typename U31, typename U32, typename U33, typename U34, typename U35, typename U36, typename U37, typename U38, typename U39, typename U40, typename U41, typename U42, typename U43, typename U44, typename U45, typename U46, typename U47, typename U48, typename U49, typename U50, typename U51, typename U52, typename U53, typename U54, typename U55, typename U56, typename U57, typename U58, typename U59, typename U60, typename U61, typename U62, typename U63>
TRInternalFunction_Impl(OUT(*fun)(U0 u0, U1 u1, U2 u2, U3 u3, U4 u4, U5 u5, U6 u6, U7 u7, U8 u8, U9 u9, U10 u10, U11 u11, U12 u12, U13 u13, U14 u14, U15 u15, U16 u16, U17 u17, U18 u18, U19 u19, U20 u20, U21 u21, U22 u22, U23 u23, U24 u24, U25 u25, U26 u26, U27 u27, U28 u28, U29 u29, U30 u30, U31 u31, U32 u32, U33 u33, U34 u34, U35 u35, U36 u36, U37 u37, U38 u38, U39 u39, U40 u40, U41 u41, U42 u42, U43 u43, U44 u44, U45 u45, U46 u46, U47 u47, U48 u48, U49 u49, U50 u50, U51 u51, U52 u52, U53 u53, U54 u54, U55 u55, U56 u56, U57 u57, U58 u58, U59 u59, U60 u60, U61 u61, U62 u62, U63 u63))
{
   set(Rcpp::XPtr< Rcpp::CppFunction64<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35, U36, U37, U38, U39, U40, U41, U42, U43, U44, U45, U46, U47, U48, U49, U50, U51, U52, U53, U54, U55, U56, U57, U58, U59, U60, U61, U62, U63> >(new Rcpp::CppFunction64<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35, U36, U37, U38, U39, U40, U41, U42, U43, U44, U45, U46, U47, U48, U49, U50, U51, U52, U53, U54, U55, U56, U57, U58, U59, U60, U61, U62, U63>(fun), false)) ;
}




template <typename OUT, typename U0, typename U1, typename U2, typename U3, typename U4, typename U5, typename U6, typename U7, typename U8, typename U9, typename U10, typename U11, typename U12, typename U13, typename U14, typename U15, typename U16, typename U17, typename U18, typename U19, typename U20, typename U21, typename U22, typename U23, typename U24, typename U25, typename U26, typename U27, typename U28, typename U29, typename U30, typename U31, typename U32, typename U33, typename U34, typename U35, typename U36, typename U37, typename U38, typename U39, typename U40, typename U41, typename U42, typename U43, typename U44, typename U45, typename U46, typename U47, typename U48, typename U49, typename U50, typename U51, typename U52, typename U53, typename U54, typename U55, typename U56, typename U57, typename U58, typename U59, typename U60, typename U61, typename U62, typename U63, typename U64>
TRInternalFunction_Impl(OUT(*fun)(U0 u0, U1 u1, U2 u2, U3 u3, U4 u4, U5 u5, U6 u6, U7 u7, U8 u8, U9 u9, U10 u10, U11 u11, U12 u12, U13 u13, U14 u14, U15 u15, U16 u16, U17 u17, U18 u18, U19 u19, U20 u20, U21 u21, U22 u22, U23 u23, U24 u24, U25 u25, U26 u26, U27 u27, U28 u28, U29 u29, U30 u30, U31 u31, U32 u32, U33 u33, U34 u34, U35 u35, U36 u36, U37 u37, U38 u38, U39 u39, U40 u40, U41 u41, U42 u42, U43 u43, U44 u44, U45 u45, U46 u46, U47 u47, U48 u48, U49 u49, U50 u50, U51 u51, U52 u52, U53 u53, U54 u54, U55 u55, U56 u56, U57 u57, U58 u58, U59 u59, U60 u60, U61 u61, U62 u62, U63 u63, U64 u64))
{
   set(Rcpp::XPtr< Rcpp::CppFunction65<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35, U36, U37, U38, U39, U40, U41, U42, U43, U44, U45, U46, U47, U48, U49, U50, U51, U52, U53, U54, U55, U56, U57, U58, U59, U60, U61, U62, U63, U64> >(new Rcpp::CppFunction65<OUT, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35, U36, U37, U38, U39, U40, U41, U42, U43, U44, U45, U46, U47, U48, U49, U50, U51, U52, U53, U54, U55, U56, U57, U58, U59, U60, U61, U62, U63, U64>(fun), false)) ;
}



#endif

