// @(#)root/pyroot:$Name:  $:$Id: MethodDispatcher.h,v 1.4 2004/08/04 04:45:21 brun Exp $
// Author: Wim Lavrijsen, Apr 2004

#ifndef PYROOT_METHODDISPATCHER_H
#define PYROOT_METHODDISPATCHER_H

// Bindings
#include "PyCallable.h"

// Standard
#include <map>
#include <string>
#include <vector>


namespace PyROOT {

/** Python side ROOT method
      @author  WLAV
      @date    09/29/2003
      @version 1.4
 */

   class MethodDispatcher {
      typedef std::map< unsigned long, int > DispatchMap_t;
      typedef std::vector< PyCallable* > Methods_t;

   public:
   // add the given dispatcher to the class, takes ownership of dispatcher
      static bool addToClass( MethodDispatcher*, PyObject* aClass );

   public:
      MethodDispatcher( const std::string& name, bool isStatic ) :
         m_name( name ), m_isStatic( isStatic ) {}
      virtual ~MethodDispatcher() {}

      const std::string& getName() const {
         return m_name;
      }

      virtual PyObject* operator()( PyObject* aTuple, PyObject* aDict );
      virtual void addMethod( PyCallable* );

   protected:
      static void destroy( void* );
      static PyObject* invoke( PyObject*, PyObject*, PyObject* );

   protected:
      unsigned long hashSignature( PyObject* aTuple );

   private:
      std::string m_name;
      bool m_isStatic;

      DispatchMap_t m_dispatchMap;
      Methods_t     m_methods;
   };

} // namespace PyROOT

#endif // !PYROOT_METHODDISPATCHER_H
