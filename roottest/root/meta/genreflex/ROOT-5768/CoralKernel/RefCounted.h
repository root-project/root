#ifndef CORALKERNEL_REFCOUNTED_H
#define CORALKERNEL_REFCOUNTED_H 1

// Include files
#include "CoralBase/boost_thread_headers.h"

namespace coral
{

  // A reference counted class
  class RefCounted
  {
  public:

    /// Increments the reference counter
    void addReference()
    {
      boost::mutex::scoped_lock lock( m_mutex );
      ++m_referenceCounter;
    }

    /// Decrements the reference counter
    int removeReference()
    {
      int result = 0;
      {
        boost::mutex::scoped_lock lock( m_mutex );
        if ( m_referenceCounter == 0 ) return result;
        --m_referenceCounter;
        result = m_referenceCounter;
      }
      if ( m_referenceCounter == 0 )
        delete this;
      return result;
    }

  protected:

    /// Constructor. Initializes the reference count to 1.
    RefCounted() :
      m_mutex(),
      m_referenceCounter( 1 )
    {}

    /// Protected destructor
    virtual ~RefCounted() {}

  private:

    /// No copy constructor
    RefCounted( const RefCounted& );

    /// No assignment operator
    RefCounted& operator=( const RefCounted& );

  private:

    /// The mutex lock for the reference counter
    boost::mutex m_mutex;

    /// Reference counter
    int m_referenceCounter;

  };

}
#endif
