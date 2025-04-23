#ifndef DAQEVENT_RAWEVENT_H
#define DAQEVENT_RAWEVENT_H 1

// Include files
#include "DataObject.h"
#include "RawBank.h"
#include <stddef.h>
#include <vector>
#include <map>

namespace LHCb 
{


  // Namespace for locations in TDS
  namespace RawEventLocation {
    static const std::string& Default  = "DAQ/RawEvent";
    static const std::string& Emulated = "Emu/RawEvent";
  }

  /** @class LHCb::RawEvent RawEvent.h
   *
   * Raw event 
   *
   * @author Helder Lopes
   *  @author Markus Frank
   * created Tue Oct 04 14:45:29 2005
   * 
   */

  class RawEvent   : public DataObject  {
  public:

    /** @class LHCb::RawEvent::Bank RawEvent.h Event/RawEvent.h
     *
     * Shadow class used to deal with persistency.
     * This class is entirely internal. Do not change.
     * In particular the field comments are hints to ROOT
     * to support the storage of variable size C-arrays in order
     * to avoid a copy of the data.
     *
     * Banks can be removed using the removeBank(RawBank*) member
     * function. The bank to be removed must be identified by its
     * pointer to ensure unambiguous bank identification also in the
     * event where multiple banks if the same bank type are present.
     * If no other bank of the category of the bank (Banktype)to 
     * be removed is anymore present in the raw event, also the 
     * category is removed.
     *
     * Note:
     * - The length passed to the RawEvent::createBank should NOT
     *   contain the size of the header !
     *
     * @author  M.Frank
     * @version 1.0
     */
    struct Bank  {
      int           m_len;      // Bank length
      char          m_owns;     //! transient data member: ownership flag
      unsigned int* m_buff;     //[m_len]
      /// Default constructor
      Bank() : m_len(0), m_owns(1), m_buff(0) {}
      /// Initializing constructor
      Bank(int len, char owns, unsigned int* b) : m_len(len), m_owns(owns), m_buff(b) {}
      /// Copy constructor
      Bank(const Bank& c) : m_len(c.m_len), m_owns(c.m_owns), m_buff(c.m_buff) {
      }
      /// Assignment operator
      Bank& operator=(const Bank& c)  {
        m_len  = c.m_len;
        m_owns = c.m_owns;
        m_buff = c.m_buff;
        return *this;
      }
      /// Access to memory buffer
      unsigned int* buffer()    {   return m_buff;          }
      /// Access to ownership flag.
      bool ownsMemory()  const  {   return m_owns == 1;     }
    };

    /// Default Constructor
      RawEvent() {};

    /// Default Destructor
	virtual ~RawEvent() {};

    /// accessor method to the vector of Raw banks for a given bank type


  private:
    /// Map banks on first request
    /** @param bankType        [IN]     type of banks to be returned (from RawBank::BankType enum)
     * 
     * @return vector of mapped banks corresponding to bankType
     */
    const std::vector<LHCb::RawBank*> &  mapBanks(LHCb::RawBank::BankType bankType);

    std::map<LHCb::RawBank::BankType,std::vector<LHCb::RawBank*> > m_eventMap; //! transient Map with RawBanks (values) for each bank type  
    std::vector<Bank>                                  m_banks;    // Vector with persistent bank structure
    bool                                               m_mapped;   //! transient
  }; // class RawEvent
} // namespace LHCb

#endif /// DAQEVENT_RAWEVENT_H
