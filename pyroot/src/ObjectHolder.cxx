// @(#)root/pyroot:$Name:  $:$Id: ObjectHolder.cxx,v 1.6 2004/08/12 20:55:10 brun Exp $
// Author: Wim Lavrijsen, Apr 2004

// Bindings
#include "PyROOT.h"
#include "ObjectHolder.h"

// ROOT
#include "TROOT.h"
#include "TObject.h"
#include "TClass.h"
#include "TSystem.h"

// Standard
#include <stdio.h>


//- private helpers ------------------------------------------------------------
inline void PyROOT::ObjectHolder::copy_( const ObjectHolder& ow ) {
// up ref-count, if used
   if ( ow.m_ref != 0 )
      *ow.m_ref += 1;

   m_ref = ow.m_ref;
   m_object = ow.m_object;
   m_class = ow.m_class;
}


inline void PyROOT::ObjectHolder::destroy_() const {
// down ref-count, if used
   if ( m_ref != 0 ) {
      *m_ref -= 1;
      if ( *m_ref <= 0 && m_class && m_object ) {
         m_class->Destructor( m_object );
         delete m_ref;
      }
   }
}


//- constructors/destructor ----------------------------------------------------
PyROOT::ObjectHolder::ObjectHolder( const ObjectHolder& ow ) :
      m_object( 0 ), m_class( 0 ), m_ref( 0 ) {
   copy_( ow );
}


PyROOT::ObjectHolder& PyROOT::ObjectHolder::operator=( const ObjectHolder& ow ) {
   if ( this != &ow ) {
      destroy_();
      copy_( ow );
   }

   return *this;
}


PyROOT::ObjectHolder::~ObjectHolder() {
   destroy_();
}


//- public members -------------------------------------------------------------
void PyROOT::ObjectHolder::release() {
   if ( m_ref && ( *m_ref - 1 ) <= 0 )
      delete m_ref;
   m_ref = 0;
}

std::string PyROOT::ObjectHolder::repr() const {
   char buf[256];
   sprintf( buf, "Instance of type %s at address %p", m_class->GetName(), getObject() );
   return buf;
}

void* PyROOT::ObjectHolder::getObject() const {
   return const_cast< void* >( m_object );             // may be null
}

TClass* PyROOT::ObjectHolder::objectIsA() const {
   return const_cast< TClass* >( m_class );      // may be null
}


void* PyROOT::AddressHolder::getObject() const {
   void* address = ObjectHolder::getObject();
   if ( address )
      return *(reinterpret_cast< void** >( address ));
   return 0;
}
