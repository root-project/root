// @(#)root/pyroot:$Name:  $:$Id: MethodHolder.h,v 1.6 2004/07/29 04:41:38 brun Exp $
// Author: Wim Lavrijsen, Apr 2004

#ifndef PYROOT_METHODHOLDER_H
#define PYROOT_METHODHOLDER_H

// Bindings
#include "Utility.h"

// ROOT
class TClass;
class TMethod;

// CINT
class G__CallFunc;

// Standard
#include <string>
#include <vector>


namespace PyROOT {

/** Python side ROOT method
      @author  WLAV
      @date    05/06/2004
      @version 1.7
 */

   class MethodHolder {
   public:
      MethodHolder( TClass*, TMethod* tm );
      MethodHolder( const MethodHolder& );
      MethodHolder& operator=( const MethodHolder& );
      virtual ~MethodHolder();

   public:
      typedef bool (*cnvfct_t)( PyObject*, G__CallFunc*, void*& );

   public:
      virtual PyObject* operator()( PyObject* aTuple, PyObject* aDict );

      virtual bool initialize();
      virtual bool setMethodArgs( PyObject* aTuple, int offset = 0 );
      virtual PyObject* callMethod( void* self );

   protected:
      virtual bool execute( void* self );
      virtual bool execute( void* self, long& retVal );
      virtual bool execute( void* self, double& retVal );

   // read-only access for subclasses
      const TClass* getClass() const {
         return m_class;
      }
      
      TClass* getClass() {
         return m_class;
      }

      const TMethod* getMethod() const {
         return m_method;
      }
      
      TMethod* getMethod() {
         return m_method;
      }

   private:
      void copy_( const MethodHolder& );
      void destroy_() const;

      bool initDispatch_();
      void calcOffset_( void* self, TClass* cls );

   private:
   // representation
      std::string  m_name;

      TClass*      m_class;
      TMethod*     m_method;
      G__CallFunc* m_methodCall;
      Utility::EDataType m_returnType;
      std::string  m_rtShortName;
      long         m_offset;
      long         m_tagnum;

   // call dispatch buffers and cache
      std::vector< void* >    m_argsBuffer;
      std::vector< cnvfct_t > m_argsConverters;
      std::string             m_callString;
      void*                   m_lastObject;

   // admin
      bool m_isInitialized;
   };

} // namespace PyROOT

#endif // !PYROOT_METHODHOLDER_H
