// @(#)root/pyroot:$Name:  $:$Id: ObjectHolder.h,v 1.6 2004/08/04 20:46:10 brun Exp $
// Author: Wim Lavrijsen, Apr 2004

#ifndef PYROOT_OBJECTHOLDER_H
#define PYROOT_OBJECTHOLDER_H

// ROOT
#include "TObject.h"

// Standard
#include <string>


namespace PyROOT {

/** ROOT instance holder
      @author  WLAV
      @date    02/25/2003
      @version 1.3
 */

   class ObjectHolder {
   public:
      ObjectHolder( void* obj, TClass* cls, bool own = true ) :
            m_object( obj ), m_class( cls ), m_ref( 0 ) {
         if ( own == true )
            m_ref = new int( 1 );
      }

      template< class RootType >
      ObjectHolder( RootType* obj, bool own = true ) : m_object( obj ), m_ref( 0 ) {
         if ( own == true )
            m_ref = new int( 1 );

         if ( m_object != 0 )
            m_class = obj->IsA();
      }

      ObjectHolder( const ObjectHolder& );
      ObjectHolder& operator=( const ObjectHolder& );
      virtual ~ObjectHolder();

      void release();

      virtual std::string repr() const;
      virtual void* getObject() const;
      virtual TClass* objectIsA() const;

   private:
      void copy_( const ObjectHolder& );
      void destroy_() const;

   private:
      void* m_object;
      TClass* m_class;
      int* m_ref;
   };

   class AddressHolder : public ObjectHolder {
   public:
      AddressHolder( void** address, TClass* cls, bool own = false ) :
         ObjectHolder( (void*) address, cls, own ) {}

   public:
      virtual void* getObject() const;
   };

} // namespace PyROOT

#endif // !PYROOT_OBJECTHOLDER_H
