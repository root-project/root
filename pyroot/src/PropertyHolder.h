// Author: Wim Lavrijsen, Jun 2004

#ifndef PYROOT_PROPERTYHOLDER_H
#define PYROOT_PROPERTYHOLDER_H

// Bindings
#include "Utility.h"

// ROOT
class TDataMember;

// Standard
#include <string>


namespace PyROOT {

/** Python side ROOT data member
      @author  WLAV
      @date    06/04/2004
      @version 1.0
 */

   class PropertyHolder {
   public:
   // add the given property to the class, takes ownership the property
      static bool addToClass( PropertyHolder*, PyObject* aClass );

   public:
      PropertyHolder( TDataMember* );
      virtual ~PropertyHolder() {}

      const std::string& getName() const {
         return m_name;
      }

      virtual PyObject* get( PyObject* aTuple, PyObject* aDict );
      virtual void set( PyObject* aTuple, PyObject* aDict );

   protected:
      static void destroy( void* );
      static PyObject* invoke_get( PyObject*, PyObject*, PyObject* );
      static PyObject* invoke_set( PyObject*, PyObject*, PyObject* );

   private:
      std::string  m_name;
      TDataMember* m_dataMember;
      Utility::EDataType m_dataType;
   };

} // namespace PyROOT

#endif // !PYROOT_PROPERTYHOLDER_H
