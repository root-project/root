#ifndef CORALBASE_IHANDLE_H
#define CORALBASE_IHANDLE_H 1

// First of all, enable or disable the CORAL240 API extensions (bug #89707)
#include "CoralBase/VersionInfo.h"

// Include files
#ifdef CORAL240CO
#include "CoralBase/Exception.h"
#endif
#include "RefCounted.h"

namespace coral
{

  /**
   *  @class IHandle IHandle.h CoralKernel/IHandle.h
   *
   *  A smart pointer to RefCounted objects.
   *  Replaces the equivalent smart pointer from SEAL/Framework.
   *
   *  -------------------------------------------------------------------
   *
   *  Comments about Coverity FORWARD_NULL bug #95358 (AV June 2012).
   *  As in the original implementation, IHandle's can only wrap pointers
   *  to RefCounted objects, because operator=(T*) only changes m_object
   *  if the T* object argument can be dynamic cast to a RefCounted*.
   *  The changes introduced in CORAL 2.4.0 are the following:
   *  - Assigning a null object, via operator=(T*) with T*=0, will now zero
   *  the internal pointer and possibly delete the wrapped object (previously
   *  this had no effect and it was necessary to assign a null IHandle<T>).
   *  - Any attempt to construct/assign a IHandle from a pointer of a non
   *  RefCounted object will throw (previously it had silently no effect).
   *  - Although the old and new algorithms both guarantee that m_object
   *  is either null or a RefCounted*, dynamic casts to RefCounted* are
   *  explicitly checked to formally fix the Coverity analyzer.
   *  - If the wrapped m_object is null, all dereference operators
   *  will throw (previously they would cause a segmentation fault).
   *
   *  Comments about Coverity DEADCODE related to bug #95358 (AV June 2012).
   *  An alternative approach to dynamic casts could be to use implicit
   *  (not explicite!) conversions from T* to RefCounted* when addReference
   *  or removeReference is called (or even call them directly). If the upcast
   *  fails because T* is not a RefCounted* the compiler would fail to build.
   *  Note that Coverity sometimes complains about this with a DEADCODE
   *  warning (if T* is a RefCounted*, checks for dynamic cast are useless).
   *
   */
  template< typename T > class IHandle
  {
  public:

    /// Constructor
    IHandle( T* object = 0 )
      : m_object( 0 )
    {
      this->operator=( object ); // m_object set only if object is a RefCounted
    }

    /// Destructor
    ~IHandle()
    {
#ifdef CORAL240CO // Fix Coverity FORWARD_NULL bug #95358
      if ( m_object )
      {
        RefCounted* rc = dynamic_cast<RefCounted*>( m_object );
        if ( rc ) rc->removeReference();
      }
#else
      if ( m_object )
        dynamic_cast<RefCounted*>( m_object )->removeReference();
#endif
    }

    /// Copy constructor
    IHandle( const IHandle& rhs )
      : m_object( 0 )
    {
      if ( rhs.m_object )
      {
        m_object = rhs.m_object;
#ifdef CORAL240CO // Fix Coverity FORWARD_NULL bug #95358
        RefCounted* rc = dynamic_cast<RefCounted*>( m_object );
        if ( !rc )
          throw Exception( "Object to copy is not a RefCounted",
                           "IHandle::IHandle",
                           "CoralKernel" );
        rc->addReference();
#else
        dynamic_cast<RefCounted*>( m_object )->addReference();
#endif
      }
    }

    /// Assignment operator
    IHandle& operator=( const IHandle& rhs )
    {
      if ( this != &rhs )
      {
        if ( m_object )
        {
#ifdef CORAL240CO // Fix Coverity FORWARD_NULL bug #95358
          RefCounted* rc = dynamic_cast<RefCounted*>( m_object );
          if ( rc ) rc->removeReference();
#else
          dynamic_cast<RefCounted*>( m_object )->removeReference();
#endif
          m_object = 0;
        }
        if ( rhs.m_object )
        {
#ifdef CORAL240CO // Fix Coverity FORWARD_NULL bug #95358
          RefCounted* rc = dynamic_cast<RefCounted*>( rhs.m_object );
          if ( !rc )
            throw Exception( "Object to assign from is not a RefCounted",
                             "IHandle::operator=(IHandle&)",
                             "CoralKernel" );
          rc->addReference();
          m_object = rhs.m_object;
#else
          m_object = rhs.m_object;
          dynamic_cast<RefCounted*>( m_object )->addReference();
#endif
        }
      }
      return *this;
    }

    /// Assignment operator from a pointer. Steals a reference...
    IHandle operator=( T* object )
    {
#ifdef CORAL240CO // Fix Coverity FORWARD_NULL bug #95358
      if ( m_object == object ) return *this;  // Avoid self-assignment
      if ( m_object )
      {
        RefCounted* rc = dynamic_cast<RefCounted*>( m_object );
        if ( rc ) rc->removeReference();
      }
      m_object = 0;
      if ( !object ) return *this;  // Assign from null object
      RefCounted* rc = dynamic_cast<RefCounted*>( object );
      if ( !rc ) // Ignore Coverity DEADCODE
        throw Exception( "Object to assign from is not a RefCounted",
                         "IHandle::operator=(T*)",
                         "CoralKernel" );
      //rc->addReference(); // NO! stealRef!
      m_object = object;
#else
      if ( object && dynamic_cast<RefCounted*>( object ) ) // no change in m_object if object is not a RefCounted
      {
        if ( m_object )
        {
          dynamic_cast<RefCounted*>( m_object )->removeReference();
          m_object = 0;
        }
        //dynamic_cast<RefCounted*>( object )->addReference(); // NO! stealRef!
        m_object = object;
      }
#endif
      return *this;
    }

    /// Dereference operator
    T& operator*()
    {
#ifdef CORAL240CO // Do not dereference zero pointer (see bug #95358)
      if ( !m_object ) throw Exception( "Object is null",
                                        "IHandle::operator*",
                                        "CoralKernel" );
#endif
      return *m_object;
    }

    /// Dereference operator
    const T& operator*() const
    {
#ifdef CORAL240CO // Do not dereference zero pointer (see bug #95358)
      if ( !m_object ) throw Exception( "Object is null",
                                        "IHandle::operator* const",
                                        "CoralKernel" );
#endif
      return *m_object;
    }

    /// Dereference operator
    T* operator->()
    {
#ifdef CORAL240CO // Do not dereference zero pointer (see bug #95358)
      if ( !m_object ) throw Exception( "Object is null",
                                        "IHandle::operator->",
                                        "CoralKernel" );
#endif
      return m_object;
    }

    /// Dereference operator
    const T* operator->() const
    {
#ifdef CORAL240CO // Do not dereference zero pointer (see bug #95358)
      if ( !m_object ) throw Exception( "Object is null",
                                        "IHandle::operator-> const",
                                        "CoralKernel" );
#endif
      return m_object;
    }

    /// Get the pointer
    T* get()
    {
      return m_object;
    }

    /// Get the pointer
    const T* get() const
    {
      return m_object;
    }

    /// Checks the validity of the pointer
    bool isValid()
    {
      return ( m_object != 0 );
    }

  private:

    /// The object
    T* m_object;

  };

}
#endif
