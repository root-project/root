// @(#)root/pyroot:$Name:  $:$Id:  $
// Author: Wim Lavrijsen, Apr 2004

// Bindings
#include "PyROOT.h"
#include "ObjectHolder.h"

// ROOT
#include "TROOT.h"
#include "TObject.h"
#include "TClass.h"

// Standard
#include <cstdio>
#include <iostream>


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
std::string PyROOT::ObjectHolder::repr() const {
   char buf[256];
   std::sprintf( buf, "Instance of type %s at address %p",
      m_class->GetName(), (void*)m_object );
   return buf;
}
