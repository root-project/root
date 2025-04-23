namespace std {}
using namespace std;
#include "ExtHitProjectHeaders.h"

struct DeleteObjectFunctor {
   template <typename T>
   void operator()(const T *ptr) const {
      delete ptr;
   }
   template <typename T, typename Q>
   void operator()(const std::pair<T,Q> &) const {
      // Do nothing
   }
   template <typename T, typename Q>
   void operator()(const std::pair<T,Q*> &ptr) const {
      delete ptr.second;
   }
   template <typename T, typename Q>
   void operator()(const std::pair<T*,Q> &ptr) const {
      delete ptr.first;
   }
   template <typename T, typename Q>
   void operator()(const std::pair<T*,Q*> &ptr) const {
      delete ptr.first;
      delete ptr.second;
   }
};

#ifndef Belle2__RelationsInterface_TObject__cxx
#define Belle2__RelationsInterface_TObject__cxx
Belle2::RelationsInterface<TObject>::RelationsInterface() {
}
Belle2::RelationsInterface<TObject>::RelationsInterface(const RelationsInterface & rhs)
   : TObject(const_cast<RelationsInterface &>( rhs ))
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!
   (void)rhs; // avoid warning about unused parameter
}
Belle2::RelationsInterface<TObject>::~RelationsInterface() {
}
#endif // Belle2__RelationsInterface_TObject__cxx

#ifndef Belle2__ExtHit_cxx
#define Belle2__ExtHit_cxx
Belle2::ExtHit::ExtHit() {
}
Belle2::ExtHit::ExtHit(const ExtHit & rhs)
   : Belle2::RelationsInterface<TObject>(const_cast<ExtHit &>( rhs ))
   , m_PdgCode(const_cast<ExtHit &>( rhs ).m_PdgCode)
   , m_DetectorID(const_cast<ExtHit &>( rhs ).m_DetectorID)
   , m_CopyID(const_cast<ExtHit &>( rhs ).m_CopyID)
   , m_Status(const_cast<ExtHit &>( rhs ).m_Status)
   , m_TOF(const_cast<ExtHit &>( rhs ).m_TOF)
   , m_Position(const_cast<ExtHit &>( rhs ).m_Position)
   , m_Momentum(const_cast<ExtHit &>( rhs ).m_Momentum)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!
   (void)rhs; // avoid warning about unused parameter
}
Belle2::ExtHit::~ExtHit() {
}
#endif // Belle2__ExtHit_cxx

