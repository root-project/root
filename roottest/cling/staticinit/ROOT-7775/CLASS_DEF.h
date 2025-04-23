#ifndef SGTOOLS_CLASS_DEF_H
#define SGTOOLS_CLASS_DEF_H
/** @file CLASS_DEF.h
 *  @brief macros to associate a CLID to a type
 *
 *  @author Paolo Calafiura <pcalafiura@lbl.gov>
 *  $Id: CLASS_DEF.h,v 1.3 2009-01-15 19:07:29 binet Exp $
 */
#include "ROOT-7775/CLIDRegistry.h"
template <class T> struct ClassID_traits;

//#include "t01/ClassID_traits.h"
//#include "CxxUtils/unused.h"
//#include <boost/preprocessor/stringize.hpp>

/** @def CLASS_DEF(NAME, CID , VERSION) 
 *  @brief associate a clid and a version to a type
 *  eg 
 *  @code 
 *  CLASS_DEF(std::vector<Track*>,8901, 1)
 *  @endcode 
 *  @param NAME 	type name
 *  @param CID 		clid
 *  @param VERSION 	not yet used
 */
#define CLASS_DEF(NAME, CID , VERSION)		\
  template <>					\
  struct ClassID_traits< NAME > {				 \
    static const std::string& typeName() {				\
      static const std::string s_name = #NAME;				\
      return s_name;							\
    }									\
    static const std::type_info& typeInfo() {				\
      return typeid (NAME);						\
    }									\
  };									\
  namespace detail {							\
    const bool t01 =                              \
      CLIDRegistry::addEntry<CID>(typeid(NAME),                         \
                                  ClassID_traits< NAME >::typeName(),   \
				  "arbitrary"); \
  } 


#endif // not SGTOOLS_CLASS_DEF_H
