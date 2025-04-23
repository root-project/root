#include "CoralBase/boost_thread_headers.h"

#if BOOST_VERSION < 15000
#ifdef WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif
#endif

namespace cool
{

  inline void sleep( int sec )
  {

#if BOOST_VERSION < 15000

#ifdef WIN32
    ::Sleep( 1000 * sec );
#else
    ::sleep( sec );
#endif

#else

    boost::this_thread::sleep_for( boost::chrono::seconds( sec ) );

#endif

  }

}
