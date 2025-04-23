// $Id: TimingReport.h,v 1.9 2012-06-29 15:27:20 avalassi Exp $
#ifndef RELATIONALCOOL_TIMINGREPORT_H
#define RELATIONALCOOL_TIMINGREPORT_H 1

#include <iostream>
#include <map>
#include <string>
#include <boost/shared_ptr.hpp>
#include "CoolKernel/VersionInfo.h"
#include "CoolChrono.h"
#include "SealUtil_TimingItem.h"

namespace cool
{

  /**
   * class managing the various timers
   * an internal class TimingItem contains the result of the timers
   *
   *
   * This class does not support copying.
   */

  class TimingReport
  {

  public:

    TimingReport() {}

    virtual ~TimingReport();

    /// Check if the timing reports are requested.
    static bool enabled();
    
#ifdef COOL_ENABLE_TIMING_REPORT

    /// Set the timing report flag.
    /// Returns the previous value.
    static bool enable(bool value);

  private:

    /// copying unimplemented in this class.
    TimingReport( const TimingReport& ) {}

    /// copying unimplemented in this class.
    TimingReport& operator=( const TimingReport& rhs )
    {
      if (this == &rhs) return *this;  // time saving self-test
      return *this;
    }

  public:

    /*
    // generic implemetation
    template<class Chrono> seal::TimingItem& item( const std::string& name )
    {
      MapItems::iterator itr = m_items.find(name);
      // item not found create a new one
      if (itr == m_items.end() ) {
        // create a new Chrono
        Chrono * c = new Chrono();
        STItem  theItem(new seal::TimingItem(*c,name) );
        m_items.insert(std::make_pair(name,theItem) );
        return *theItem;
      }
      else {
        STItem theItem = itr->second;
        return *theItem;
      }
    }
    */

    // default implementation based on  CoolChrono
    seal::TimingItem& item( const std::string& name )
    {
      MapItems::iterator itr = m_items.find(name);
      // item not found create a new one
      if (itr == m_items.end() ) {
        // create a new CoolChrono
        CoolChrono* c = new CoolChrono();
        m_chronos.push_back( c );
        STItem theItem( new seal::TimingItem( *c, name ) );
        m_items.insert(std::make_pair(name,theItem) );
        return *theItem;
      }
      else {
        STItem theItem = itr->second;
        return *theItem;
      }
    }

    // dump compact info
    void dump( std::ostream& os = std::cout );

    // dump in a more extended format
    void dumpFull( std::ostream& os = std::cout );

    // print reports
    static
    void printLast( seal::TimingItem& item, std::ostream& os = std::cout );

  private:

    std::vector<CoolChrono*> m_chronos;

    typedef boost::shared_ptr<seal::TimingItem > STItem;
    typedef std::map<std::string, STItem> MapItems;
    MapItems m_items;

    /// Variable to store the flag controlling time reports.
    static bool m_enabled;
    
#endif

  };

}

#endif
