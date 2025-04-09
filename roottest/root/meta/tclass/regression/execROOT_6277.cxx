#if 0
// Original Problem
// See https://sft.its.cern.ch/jira/browse/ROOT-6277
//
#include <boost/operators.hpp>
class T: public booist::equality_comparable<T> {
   bool operator==(const T&) { return false; }
};


/*
 dump of decl:

 ClassTemplateSpecializationDecl 0x5b0f470 </usr/include/boost/operators.hpp:798:68, line:801:29> struct equality_comparable definition
 |-public 'equality_comparable1<class T, class boost::detail::empty_base<class T> >':'struct boost::equality_comparable1<class T, class boost::detail::empty_base<class T> >'
 |-TemplateArgument type 'class T'
 |-TemplateArgument type 'class T'
 |-TemplateArgument type 'class boost::detail::empty_base<class T>'
 |-TemplateArgument type 'struct boost::detail::false_t'
 `-CXXRecordDecl 0x5b0ffa0 prev 0x5b0f470 <line:799:68, line:834:25> struct equality_comparable
 */

#endif


namespace boo {

   namespace detail {
      template <typename A> struct empty_base {};
      struct true_t;
      struct false_t;
      struct fromIsChain_t;

   }

   template <typename A, typename B, typename C>
   struct equality_comparable2 {};

   template <class T1, class B1 = detail::empty_base<T1> >
   struct equality_comparable1 : B1
   {
      friend bool operator!=(const T1& x, const T1& y) { return !static_cast<bool>(x == y); }
   };

   template<class T2> struct is_chained_base {
      typedef ::boo::detail::false_t value;
   };

   template <class T3 ,class U3 = T3 ,class B3 = ::boo::detail::empty_base<T3> ,class O = typename boo::is_chained_base<U3>::value >
   struct equality_comparable : boo::equality_comparable2<T3, U3, B3> {};

   template<class T4, class U4, class B4>
   struct equality_comparable<T4, U4, B4, ::boo::detail::true_t> : boo::equality_comparable1<T4, U4> {};

   template <class T5, class B5>
   struct equality_comparable<T5, T5, B5, ::boo::detail::false_t> : boo::equality_comparable1<T5, B5> {};

   /*
    * template<class T, class U, class B, class O> struct is_chained_base< ::boost::equality_comparable<T, U, B, O> > { typedef ::boost::detail::true_t value; };
    template<class T, class U, class B> struct is_chained_base< ::boost::equality_comparable2<T, U, B> > { typedef ::boost::detail::true_t value; };
    template<class T, class B> struct is_chained_base< ::boost::equality_comparable1<T, B> > { typedef ::boost::detail::true_t value; };
    */


}

class UserClass: public boo::equality_comparable<UserClass> {
   bool operator==(const UserClass&) { return false; }
};

#include "TClass.h"
#include "TList.h"

int execROOT_6277()
{
   TClass *cl = TClass::GetClass("UserClass");
   if (!cl) return 1;
   cl->GetListOfBases()->ls("noaddr");
   return 0;
}


#ifdef __MAKECINT__
#pragma link C++ class UserClass+;
#endif
