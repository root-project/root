// Include files
#include <cstdlib> // for atol
#include <iostream> // for cout
#include <sstream> // for stringstream

class ProcMemory {

public:

  ProcMemory() {
    m_pid = getProcessId();
  }

  ~ProcMemory() {}

  inline void printVm();

  inline long getProcessId();

  inline long getVsz();
  inline long getRss();
  inline long getProcStatus( const std::string& key );

private:

  int m_pid;

};

//----------------------------------------------------------------------------

inline void ProcMemory::printVm() {
  long vsz = getVsz();
  long rss = getRss();
  std::cout << "VSZ=" << vsz << " RSS=" << rss << std::endl;
}

//----------------------------------------------------------------------------

// WINDOWS - see http://msdn2.microsoft.com/en-us/library/t2y34y40
#ifdef WIN32

#include <process.h>
inline long ProcMemory::getProcessId() {
  return _getpid();
}

// OSX and LINUX
#else

#include <sys/types.h>
#include <unistd.h>
inline long ProcMemory::getProcessId() {
  return getpid();
}

#endif

//----------------------------------------------------------------------------

// WINDOWS
#ifdef WIN32

// For ifstream
#include <fstream>

inline long ProcMemory::getVsz() {
  return getProcStatus( "VmSize" );
}

inline long ProcMemory::getRss() {
  return getProcStatus( "VmRSS" );
}

// NB To get rid of the annoying messages "UNC not supported",
// add a DWORD registry key (hex value 1) called DisableUNCCheck to
// HKCU\Software\Microsoft\Command Processor.
// See http://weblogs.asp.net/kdente/archive/2004/01/30/65232.aspx

// NB A non-cygwin alternative for Windows could be to use "tlist":
// this is only available after installing the Windows XP Support Tools.
// See http://emea.windowsitpro.com/Article/ArticleID/43569/43569.html?Ad=1
// See http://www.ss64.com/nt/tlist.html

// NB Another simple non-cygwin alternative could be to use "tasklist":
// however, this only provides the VmRSS value, which is much less
// interesting than the VmSize value (only VmSize is affected by malloc).

inline long ProcMemory::getProcStatus( const std::string& key ) {
  long value = -1;
  std::stringstream cmd;
  cmd << "ps | gawk '{if ($4 == " << m_pid
      << ") {cmd=\"cat /proc/\"$1\"/status\"; system(cmd)}}'"
      << " | grep " << key;
  const int kMaxSize = 256;
  char line[kMaxSize];
  FILE* input_stream = _popen( cmd.str().c_str(), "r" );
  if ( fgets( line, kMaxSize, input_stream ) == NULL ) {
    //std::cout << "Error while getting status" << std::endl;
  } else {
    //std::cout << "line: " << line << std::endl;
    std::string lineStr = line;
    if( lineStr.substr(0,key.size()+1) == (key+":") ) {
      lineStr = lineStr.substr(key.size()+2,lineStr.size()-key.size()-2-3);
      value = atol(lineStr.c_str());
    }
  }
  _pclose( input_stream );
  return value;
}

//----------------------------------------------------------------------------

// OSX and FreeBSD
#elif defined(__APPLE__) || defined(__FreeBSD__)

inline long ProcMemory::getVsz() {
  return getProcStatus( "vsz" );
}

inline long ProcMemory::getRss() {
  return getProcStatus( "rss" );
}

inline long ProcMemory::getProcStatus( const std::string& key ) {
  const int kMaxSize = 256;
  char line[kMaxSize];
  long value = -1;
  std::stringstream cmd;
  cmd << "ps -p " << m_pid << " -o " << key << "| grep -iv " << key;
  FILE* input_stream = popen( cmd.str().c_str(), "r" );
  if ( fgets( line, kMaxSize, input_stream ) == NULL ) {
    //std::cout << "Error while getting status" << std::endl;
  } else {
    //std::cout << "Line: " << line << std::endl;
    value = atol( line );
  }
  pclose( input_stream );
  return value;
}

//----------------------------------------------------------------------------

// LINUX
#else

// For getpid()
#include <sys/types.h>
#include <unistd.h>

// For ifstream
#include <fstream>

inline long ProcMemory::getVsz() {
  return getProcStatus( "VmSize" );
}

inline long ProcMemory::getRss() {
  return getProcStatus( "VmRSS" );
}

inline long ProcMemory::getProcStatus( const std::string& key ) {
  long value = -1;
  std::stringstream fileName;
  fileName << "/proc/" << m_pid << "/status";
  std::ifstream proc( fileName.str().c_str() );
  if( !proc ) {
    std::cout << "Error opening " << fileName.str() << std::endl;
  } else {
    char ch;
    std::string line;
    while( proc.get(ch) ) {
      if ( ch != '\n' ) {
        line += ch;
      } else {
        if( line.substr(0,key.size()+1) == (key+":") ) {
          line = line.substr(key.size()+2,line.size()-key.size()-2-3);
          //std::cout << "Line: " << line << std::endl;
          value = atol(line.c_str());
        }
        line = "";
      }
    }
    if( !proc.eof() ) {
      std::cout << "Error while reading " << fileName.str() << std::endl;
    } else {
      //std::cout << "EOF " << fileName.str() << std::endl;
    }
  }
  //std::cout << key << ": " << value << std::endl;
  return value;
}

#endif

//----------------------------------------------------------------------------
