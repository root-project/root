#ifndef SGTOOLS_CLIDREGISTRY_H
# define SGTOOLS_CLIDREGISTRY_H
/** @file CLIDRegistry.h
 * @brief  a static registry of CLID->typeName entries. NOT for general use.
 * Use ClassIDSvc instead.
 *
 * @author Paolo Calafiura <pcalafiura@lbl.gov> - ATLAS Collaboration
 *$Id: CLIDRegistry.h,v 1.2 2009-01-15 19:07:29 binet Exp $
 */

#include <string>

/** @class CLIDRegistry
 * @brief  a static registry of CLID->typeName entries. NOT for general use.
 * Use ClassIDSvc instead.
 */
class CLIDRegistry {
public:

  ///to be called by the CLASS_DEFS
  template <unsigned long CLID>
  static bool addEntry(const std::type_info& ti,
                       const std::string& typeName, 
		       const std::string& typeInfoName); 

};


//<<<<<< INLINE MEMBER FUNCTIONS                                        >>>>>>
template <unsigned long CLID>
bool CLIDRegistry::addEntry(const std::type_info& ti,
                            const std::string& typeName, 
			    const std::string& typeInfoName) {
  //more drudgery
  return true;
}


#endif // SGTOOLS_CLIDREGISTRY_H
