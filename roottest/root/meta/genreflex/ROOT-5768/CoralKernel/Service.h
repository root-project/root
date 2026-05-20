#ifndef CORALBASE_SERVICE_H
#define CORALBASE_SERVICE_H 1

// First of all, enable or disable the CORAL240 API extensions (bug #89707)
#include "CoralBase/VersionInfo.h"

// Include files
#include "ILoadableComponent.h"

#include <map>
#include <stdexcept>

namespace coral
{

  class MessageStream;

  class Service : public ILoadableComponent
  {
  protected:

    /// Constructor
    explicit Service( const std::string& name );

    /// Destructor
    ~Service() override;

    /// Returns the underlying message stream object
    MessageStream& log()
    {
      return *m_log;
    }

    /// Returns the underlying message stream object
    MessageStream& log() const
    {
      return *m_log;
    }

#ifdef CORAL240CO
  private:

    /// Copy constructor is private (fix Coverity MISSING_COPY bug #95359)
    Service( const Service& rhs );

    /// Assignment op. is private (fix Coverity MISSING_ASSIGN bug #95359)
    Service& operator=( const Service& rhs );
#endif

  private:

    /// The message stream
    mutable MessageStream* m_log;

  };

}
#endif
