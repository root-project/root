#include "libcondProjectHeaders.h"

//#include "libcondLinkDef.h"

//#include "libcondProjectDict.cxx"

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

#ifndef DetCondKeyTrans_cxx
#define DetCondKeyTrans_cxx
DetCondKeyTrans::DetCondKeyTrans() {
}
DetCondKeyTrans::DetCondKeyTrans(const DetCondKeyTrans & rhs)
   : m_keytrans(const_cast<DetCondKeyTrans &>( rhs ).m_keytrans)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!
   // if (&rhs) {} // avoid warning about unused parameter
   DetCondKeyTrans &modrhs = const_cast<DetCondKeyTrans &>( rhs );
   modrhs.m_keytrans.clear();
}
DetCondKeyTrans::~DetCondKeyTrans() {
}
#endif // DetCondKeyTrans_cxx

#ifndef HepGeom__Transform3D_cxx
#define HepGeom__Transform3D_cxx
HepGeom::Transform3D::Transform3D() {
}
HepGeom::Transform3D::Transform3D(const Transform3D & rhs)
   : xx_(const_cast<Transform3D &>( rhs ).xx_)
   , xy_(const_cast<Transform3D &>( rhs ).xy_)
   , xz_(const_cast<Transform3D &>( rhs ).xz_)
   , dx_(const_cast<Transform3D &>( rhs ).dx_)
   , yx_(const_cast<Transform3D &>( rhs ).yx_)
   , yy_(const_cast<Transform3D &>( rhs ).yy_)
   , yz_(const_cast<Transform3D &>( rhs ).yz_)
   , dy_(const_cast<Transform3D &>( rhs ).dy_)
   , zx_(const_cast<Transform3D &>( rhs ).zx_)
   , zy_(const_cast<Transform3D &>( rhs ).zy_)
   , zz_(const_cast<Transform3D &>( rhs ).zz_)
   , dz_(const_cast<Transform3D &>( rhs ).dz_)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!
   (void) rhs; // avoid warning about unused parameter
}
HepGeom::Transform3D::~Transform3D() {
}
#endif // HepGeom__Transform3D_cxx

#ifndef DataHeaderForm_p5_cxx
#define DataHeaderForm_p5_cxx
DataHeaderForm_p5::DataHeaderForm_p5() {
}
DataHeaderForm_p5::DataHeaderForm_p5(const DataHeaderForm_p5 & rhs)
   : m_map(const_cast<DataHeaderForm_p5 &>( rhs ).m_map)
   , m_uints(const_cast<DataHeaderForm_p5 &>( rhs ).m_uints)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!
   // if (&rhs) {} // avoid warning about unused parameter
   DataHeaderForm_p5 &modrhs = const_cast<DataHeaderForm_p5 &>( rhs );
   modrhs.m_map.clear();
   modrhs.m_uints.clear();
}
DataHeaderForm_p5::~DataHeaderForm_p5() {
}
#endif // DataHeaderForm_p5_cxx

#ifndef DataHeader_p5_cxx
#define DataHeader_p5_cxx
DataHeader_p5::DataHeader_p5() {
}
DataHeader_p5::DataHeader_p5(const DataHeader_p5 & rhs)
   : m_dataHeader(const_cast<DataHeader_p5 &>( rhs ).m_dataHeader)
   , m_dhFormToken(const_cast<DataHeader_p5 &>( rhs ).m_dhFormToken)
   , m_dhFormMdx(const_cast<DataHeader_p5 &>( rhs ).m_dhFormMdx)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!
   // if (&rhs) {} // avoid warning about unused parameter
   DataHeader_p5 &modrhs = const_cast<DataHeader_p5 &>( rhs );
   modrhs.m_dataHeader.clear();
   modrhs.m_dhFormToken.clear();
   modrhs.m_dhFormMdx.clear();
}
DataHeader_p5::~DataHeader_p5() {
}
#endif // DataHeader_p5_cxx

#ifndef DataHeaderElement_p5_cxx
#define DataHeaderElement_p5_cxx
DataHeaderElement_p5::DataHeaderElement_p5() {
}
DataHeaderElement_p5::DataHeaderElement_p5(const DataHeaderElement_p5 & rhs)
   : m_token(const_cast<DataHeaderElement_p5 &>( rhs ).m_token)
   , m_oid2(const_cast<DataHeaderElement_p5 &>( rhs ).m_oid2)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!
   // if (&rhs) {} // avoid warning about unused parameter
   DataHeaderElement_p5 &modrhs = const_cast<DataHeaderElement_p5 &>( rhs );
   modrhs.m_token.clear();
}
DataHeaderElement_p5::~DataHeaderElement_p5() {
}
#endif // DataHeaderElement_p5_cxx

