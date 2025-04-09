#ifndef DataFormats_Common_traits_h
#define DataFormats_Common_traits_h

/*----------------------------------------------------------------------

Definition of traits templates used in the EDM.  


----------------------------------------------------------------------*/

#include <deque>
#include <limits>
#include <list>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace edm
{
  //------------------------------------------------------------
  //
  // The trait struct template key_traits<K> is used to carry
  // information relevant to the type K when used as a 'key' in
  // RefVector and its related classes and templates.
  //
  // The general case works only for integral types K; for more
  // 'esoteric' types, one must introduce an explicit specialization.
  // That specialization must initialize the static data member
  // 'value'.

  template <class K>
  struct key_traits
  {
    typedef K key_type;
    static const key_type value;
  };

  template <class K> 
  typename  key_traits<K>::key_type const
  key_traits<K>::value =
    std::numeric_limits<typename key_traits<K>::key_type>::max();

  // Partial specialization for std::pair

  template <class U, class V>
  struct key_traits<std::pair<U,V> >
  {
    typedef std::pair<U,V>  key_type;
    static const key_type value;
  };

  template <class U, class V>
  typename key_traits<std::pair<U,V> >::key_type const
  key_traits<std::pair<U,V> >::value = 
    std::make_pair(key_traits<U>::value, key_traits<V>::value);

  // If we ever need to support instantiations of std::basic_string
  // other than std::string, this is the place to do it.

  // For value, we make a 1-character long string that contains an
  // unprintable character; we are hoping nobody ever uses such a
  // string as a legal key.
  template <> 
  struct key_traits<std::string>
  {
    typedef std::string key_type;
    static const key_type value;
  };


  //------------------------------------------------------------
  //
  // DoNotSortUponInsertion is a base class. Derive your own class X
  // from DoNotSortUponInsertion when: 
  //
  // 1. You want to use DetSetVector<X> as an EDProduct, but
  //
  // 2. You do *not* want the Event::put member template to cause the
  // DetSet<X> instances within the DetSetVector<X> to be sorted.
  //
  // DoNotSortUponInsertion has no behavior; it is used at compile
  // time to influence the behavior of Event::put.
  //
  // Usage:
  //    class MyClass : public edm::DoNotSortUponInsertion { ... }
  //
  struct DoNotSortUponInsertion { };

  //------------------------------------------------------------
  //
  // DoNotRecordParents is a base class. Derive your own (EDProduct)
  // class X from DoNotRecordParents when your class already keeps all
  // data that are relevant to parentage internally, and the
  // information kept by the event model would thus be redundant.
  //
  // DoNotRecordParents has no behavior; it is used at compile time to
  // influence the behavior of Event::put.
  //
  // Usage:
  //    class MyClass : public edm::DoNotRecordParents { ... }
  struct DoNotRecordParents { };

  // Other is a base class. NEVER USE IT. It is for the
  // core of the event model only.
  struct Other { };

  //------------------------------------------------------------
  //
  // The trait struct template has_fillView<T> is used to
  // indicate whether or not the type T has a member function
  //
  //      void T::fillView(std::vector<void const*>&) const
  //
  // We assume the 'general case' for T is to not support fillView.
  // Classes which do support fillView must specialize this trait.
  //
  //------------------------------------------------------------

  template <class T>
  struct has_fillView
  {
    static bool const value = false;
  };

  template <class T, class A>
  struct has_fillView<std::vector<T,A> >
  {
    static bool const value = true;
  };

  template <class A>
  struct has_fillView<std::vector<bool,A> >
  {
    static bool const value = false;
  };

  template <class T, class A>
  struct has_fillView<std::list<T,A> >
  {
    static bool const value = true;
  };

  template <class T, class A>
  struct has_fillView<std::deque<T,A> >
  {
    static bool const value = true;
  };

  template <class T, class A>
  struct has_fillView<std::set<T,A> >
  {
    static bool const value = true;
  };


  //------------------------------------------------------------
  //
  // The trait struct template has_setPtr<T> is used to
  // indicate whether or not the type T has a member function
  //
  //      void T::setPtr(const std::type_info&, void const*&) const
  //
  // We assume the 'general case' for T is to not support setPtr.
  // Classes which do support setPtr must specialize this trait.
  //
  //------------------------------------------------------------
  
  template <class T>
    struct has_setPtr
  {
    static bool const value = false;
  };
  
  template <class T, class A>
    struct has_setPtr<std::vector<T,A> >
  {
    static bool const value = true;
  };
  
  template <class A>
    struct has_setPtr<std::vector<bool,A> >
  {
    static bool const value = false;
  };
  
  template <class T, class A>
    struct has_setPtr<std::list<T,A> >
  {
    static bool const value = true;
  };
  
  template <class T, class A>
    struct has_setPtr<std::deque<T,A> >
  {
    static bool const value = true;
  };
  
  template <class T, class A>
    struct has_setPtr<std::set<T,A> >
  {
    static bool const value = true;
  };
}

#endif
