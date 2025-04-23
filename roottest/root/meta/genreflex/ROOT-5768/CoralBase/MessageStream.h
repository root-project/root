// $Id: MessageStream.h,v 1.7 2009-07-10 11:17:41 avalassi Exp $
//====================================================================
//  CORAL Base Printer object
//--------------------------------------------------------------------
//
//  @author      M.Frank, R. Chytracek (port to CORAL)
//====================================================================
#ifndef CORAL_CORALBASE_MESSAGESTREAM_H
#define CORAL_CORALBASE_MESSAGESTREAM_H 1

// First of all, enable or disable the CORAL240 API extensions (bug #89707)
#include "CoralBase/VersionInfo.h"

// Framework customization file
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <vector>
#include <list>
#include <algorithm>

/*
 *   CORAL namespace declaration
 */
namespace coral
{

  // Forward declarations
  class IMsgReporter;

  enum  MsgLevel { Nil=0, Verbose, Debug, Info, Warning, Error, Fatal, Always, NumLevels };

  /** @class MessageStream MessageStream.h CoralBase/MessageStream.h
   *  CORAL Base Printer object
   *
   * @author   Markus Frank, Radovan Chytracek (port to CORAL)
   * @version  1.0
   * @date     15/07/2002
   */
  class MessageStream
  {
    void prtLongLong(const long long int arg);
    void prtLongLong(const unsigned long long int arg);
    void doOutput();
  public:
    typedef MessageStream _P;
    typedef std::ios_base::fmtflags flg_t;
    typedef std::ios_base::iostate state_t;

    /// String MessageStream associated to buffer
    std::ostringstream m_stream;
    /// Flag set to true if formatting engine is active
    bool m_act;
    /// Source name
    std::string m_source;
    /// Debug level of the message service
    MsgLevel m_level;
    /// Current debug level
    MsgLevel m_currLevel;

    template<class T> struct _ItemPrint
    {
      std::ostringstream& m_str;
      _ItemPrint(std::ostringstream& s) : m_str(s) {}
      bool operator()(const T& o) { m_str << o << " ";  return true; }
    };

  public:
    // Set the verbosity of the default message reporter.
    static void setMsgVerbosity(MsgLevel lvl);
    // Get the verbosity of the default message reporter.
    static MsgLevel msgVerbosity();
    // Install a new default message reporter.
    static void installMsgReporter(IMsgReporter* reporter);
    /// Standard initializing constructor
    MessageStream(const std::string& source);
    /// Standard destructor
    virtual ~MessageStream() {}
    /// Initialize report of new message: activate if print level is sufficient.
    void report(int lvl)
    {
      m_currLevel = MsgLevel( (lvl >= NumLevels)
                              ? Always
                              : (lvl<Nil) ? Nil : lvl );
      m_act = m_currLevel >= m_level;
    }
#define CALL_RETURN(x,y) { x; return y; }
#define THIS_RETURN(x) CALL_RETURN (x,*this)
#define COND_RETURN(x) THIS_RETURN (if(m_act) x)
    // iostream write emulation
    _P& write(const char* buff,int len) COND_RETURN(m_stream.write(buff, len))
    /// Accept MessageStream modifiers
    _P& operator<<(_P& (*f)(_P&)) COND_RETURN(f(*this))
    /// Accept ios base class modifiers
    _P& operator<<(std::ios_base& (*f)(std::ios_base&)) COND_RETURN(f(m_stream))
    /// Accept ios modifiers
    _P& operator<<(std::ios& (*f)(std::ios&)) COND_RETURN(f(m_stream))
    /// Accept iostream modifiers
    _P& operator<<(std::ostream& (*f)(std::ostream&)) COND_RETURN(f(m_stream))
    /// Accept MessageStream activation using MsgStreamer operator
    _P& operator<<(MsgLevel level) THIS_RETURN(report(level))
    /// General templated stream operator
    template <typename T> _P& operator<<(const T& arg) COND_RETURN(m_stream << arg)
    /// Specialization for long long integer type
    _P& operator<<(const void* arg) COND_RETURN(m_stream << (void*)arg)
    /// Specialization for long long integer type
    _P& operator<<(const long long int arg) COND_RETURN(prtLongLong(arg))
    /// Specialization for unsigned long long integer type
    _P& operator<<(const unsigned long long int arg) COND_RETURN(prtLongLong(arg))
    /// Specialization stream operator for std::vector<T>
    template <typename T> _P& operator<<(const std::vector<T>& c )
    COND_RETURN(std::for_each(c.begin(),c.end(),_ItemPrint<T>(m_stream)))
    /// Specialization stream operator for std::list<T>
    template <typename T> _P& operator<<(const std::list<T>& c )
    COND_RETURN(std::for_each(c.begin(),c.end(),_ItemPrint<T>(m_stream)))
#ifdef _WIN32
    template<typename T> _P& operator<<(const std::_Fillobj<T>& obj)
    COND_RETURN(m_stream.fill(obj._Ch))

    template<typename T> _P& operator<<(const std::_Smanip<T>& m)
#if _MSC_VER > 1300
    COND_RETURN((*m._Pfun)(m_stream,m._Manarg))
#else
    COND_RETURN((*m._Pf)(m_stream,m._Manarg))
#endif

#elif defined (__GNUC__)
    template<typename T> _P& operator <<(const std::_Setfill<T> &m) THIS_RETURN(m_stream<<m)
    _P& operator << (const std::_Setiosflags &m) COND_RETURN(m_stream<<m)
    _P& operator << (const std::_Resetiosflags &m) COND_RETURN(m_stream<<m)
    _P& operator << (const std::_Setbase &m) COND_RETURN(m_stream<<m)
    _P& operator << (const std::_Setw &m) COND_RETURN(m_stream<<m)
    _P& operator << (const std::_Setprecision &m) COND_RETURN(m_stream<<m)
#endif

    /// MsgStream Modifier: endreq. Calls the output method of the MsgStream
    static _P & endmsg(_P& s) CALL_RETURN (if(s.m_act) s.doOutput(),s)
    static _P & flush (_P& s) CALL_RETURN (if(s.m_act) s.m_stream.flush(),s)
    static _P & dec   (_P& o) CALL_RETURN (if(o.m_act) o.setf(std::ios_base::dec,std::ios_base::basefield),o)
    static _P & hex   (_P& o) CALL_RETURN (if(o.m_act) o.setf(std::ios_base::hex,std::ios_base::basefield),o)

    /// IOS emulation
    long flags() const { return m_act ? m_stream.flags() : 0; }
    long flags(flg_t v) { return m_act ? m_stream.flags(v) : 0; }
    long setf(flg_t v) { return m_act ? m_stream.setf(v) : 0; }
    int width() const { return m_act ? m_stream.width() : 0; }
    int width(int v) { return m_act ? m_stream.width(v) : 0; }
    char fill() const { return m_act ? m_stream.fill() : -1; }
    char fill(char v) { return m_act ? m_stream.fill(v) : -1; }
    int precision() const { return m_act ? m_stream.precision() : 0; }
    int precision(int v) { return m_act ? m_stream.precision(v) : 0; }
    int rdstate() const { return m_act ? m_stream.rdstate () : std::ios_base::failbit; }
    int good() const { return m_act ? m_stream.good() : 0; }
    int eof() const { return m_act ? m_stream.eof () : 0; }
    int bad() const { return m_act ? m_stream.bad() : 0; }
    long setf(flg_t f,flg_t m) { return m_act ? m_stream.setf(f,m) : 0; }
    void unsetf(flg_t l)       { if (m_act) m_stream.unsetf(l); }
    void clear(state_t s = std::ios_base::failbit)  { if ( m_act ) m_stream.clear(s); }

  };
#undef CALL_RETURN
#undef THIS_RETURN
#undef COND_RETURN

  /** @class IMsgReporter MessageStream.h CoralBase/MessageStream.h
   *
   * Interface to external message reporter
   *
   * @author   Markus Frank
   * @version  1.0
   * @date     15/07/2002
   */
  class IMsgReporter
  {
  public:
    /// Destructor (called only by sub-classes)
    virtual ~IMsgReporter() {}
    /// Release reference to reporter
    virtual void release() = 0;
    /// Access output level
    virtual coral::MsgLevel outputLevel() const = 0;
    /// Modify output level
    virtual void setOutputLevel(coral::MsgLevel lvl) = 0;
    /// Report a message
    virtual void report(int level, const std::string& src, const std::string& msg) = 0;
  };

#ifndef CORAL240MR
  /** @class MsgReporter MessageStream.h CoralBase/MessageStream.h
   *
   * Default reporter implementation
   *
   * WARNING! THIS CLASS IS NO LONGER SUPPORTED AS OF CORAL240!
   * [See https://savannah.cern.ch/bugs/index.php?53040]
   *
   * @author   Markus Frank
   * @version  1.0
   * @date     15/07/2002
   */
  class MsgReporter : public IMsgReporter
  {
  public:
    /// Default constructor
    MsgReporter();
    /// Destructor
    ~MsgReporter() override {}
    /// Release reference to reporter
    void release() override { delete this; }
    /// Access output level
    coral::MsgLevel outputLevel() const override;
    /// Modify output level
    void setOutputLevel(coral::MsgLevel lvl) override;
    /// Report a message
    void report(int lvl, const std::string& src, const std::string& msg) override;

  private:
    /// The current message level threshold
    coral::MsgLevel m_level;
  };
#endif

} // End namespace coral

#endif // CORAL_CORALBASE_MESSAGESTREAM_H
