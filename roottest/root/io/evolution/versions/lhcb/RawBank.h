#ifndef DAQEVENT_RAWBANK_H
#define DAQEVENT_RAWBANK_H 1

#include <stddef.h>
#include <vector>

/** @class LHCb::RawBank RawBank.h
  *
  * Raw data bank sent by the TELL1 boards of the LHCb DAQ.
  *
  * For a detailed description of the raw bank format,
  * see <a href="https://edms.cern.ch/document/565851/5">EDMS-565851/5</a>
  *
  * Note concerning the changes done 06/03/2006:
  * - The bank size is in BYTES
  * - The full size of a bank in memory is long word (32 bit) aligned
  *   ie. a bank of 17 Bytes length actually uses 20 Bytes in memory.
  * - The bank length accessors size() and setSize() do not contain
  *   the bank header, ie. size() = (total size) - (header size).
  * - The length passed to the RawEvent::createBank should NOT
  *   contain the size of the header !
  * - If the full padded bank size is required use the utility
  *   function RawBank::totalSize = size + header size + padding size.
  *
  * @author Helder Lopes
  * @author Markus Frank
  * created Tue Oct 04 14:45:20 2005
  * 
  */
namespace LHCb 
{
  class RawBankSubClass;
  
  class RawBank  {
    /// Need to declare some friend to avoid compiler warnings
    friend class RawBankSubClass;
  private:


  public:   

    /// Default Constructor
    RawBank() {}
    /// Default Destructor
    ~RawBank() {}

    // typedef for std::vector of RawBank
    typedef std::vector<RawBank*> Vector;
    typedef std::vector<const RawBank*> ConstVector;

    /// Define bank types for RawBank
    enum BankType{ L0Calo=0,
                   L0DU,
                   PrsE,
                   EcalE,
                   HcalE,
                   PrsTrig,
                   EcalTrig,
                   HcalTrig,
                   Velo,
                   Rich,
                   TT,
                   IT,
                   OT,
                   Muon,
                   L0PU,
                   DAQ,
                   ODIN,
                   HLT,
                   VeloFull,
                   TTFull,
                   ITFull,
                   EcalPacked,
                   HcalPacked,
                   PrsPacked,
                   L0Muon,
                   ITError,
                   TTError,
                   ITPedestal,
                   TTPedestal,
                   VeloError,
                   VeloPedestal,
                   VeloProcFull,
                   OTRaw,
                   OTError,
                   EcalPackedError, //34
                   HcalPackedError, //35  
                   PrsPackedError,  //36
                   L0CaloFull,      //37
                   L0CaloError,     //38
                   L0MuonCtrlAll,   //39
                   L0MuonProcCand,  //40
                   L0MuonProcData,  //41
                   L0MuonRaw,       //42
		   L0MuonError,     //43
                   GaudiSerialize,  //44
                   GaudiHeader,     //45
                   TTProcFull,      //46
                   ITProcFull,      //47
                   LastType    // LOOP Marker; add new bank types ONLY before!
    };

    /// Magic pattern for Raw bank headers
    enum RawPattern{ MagicPattern=0xCBCB };

    /// Access to magic word for integrity check
    int magic()  const              {    return m_magic;            }

    /// Set magic word
    void setMagic()                 {    m_magic = MagicPattern;    }

    /// Header size
    int hdrSize()  const            {    return sizeof(RawBank)-sizeof(m_data);}

    /// Return size of the data body part of the bank
    int size() const                {    return m_length-hdrSize();            }

    /// Set data size of the bank in bytes
    void setSize(size_t val)        {    m_length = (val&0xFFFF)+hdrSize();    }

    /// Access the full (padded) size of the bank
    int totalSize() const           { 
      typedef unsigned int T;
      return m_length%sizeof(T)==0 ? m_length : (m_length/sizeof(T)+1)*sizeof(T);
    }
    /// Return bankType of this bank 
    BankType type() const           {    return BankType(int(m_type));         }

    /// Set the bank type
    void setType(BankType val)      {    m_type = (int(val)&0xFF);  }

    /// Return version of this bank 
    int version() const             {    return m_version;          }

    /// Set the version information of this bank
    void setVersion(int val)        {    m_version = (val&0xFF);    }

    /// Return SourceID of this bank  (TELL1 board ID)
    int sourceID() const            {    return m_sourceID;         }

    /// Set the source ID of this bank (TELL1 board ID)
    void setSourceID(int val)       {    m_sourceID = (val&0xFFFF); }

    /// Return pointer to begining of data body of the bank
    unsigned int* data()            {    return &m_data[0];         }

    /// Return pointer to begining of data body of the bank (const data access)
    const unsigned int* data() const {   return &m_data[0];         }

    /// Begin iterator 
    template <typename T> T* begin(){    return (T*)m_data;         }

    /// End iterator 
    template <typename T> T* end()  {    return ((T*)m_data) + size()/sizeof(T);  }

    /// Begin iterator over const iteration
    template <typename T> const T* begin() const {return (T*)m_data;}

    /// End iterator of const iteration
    template <typename T>  const T* end() const  {return ((T*)m_data) + size()/sizeof(T);  }

  private:
    /// Magic word (by definition 0xCBCB)
    unsigned short m_magic;
    /// Bank length in bytes (must be >= 0)
    unsigned short m_length;
    /// Bank type (must be >= 0)
    unsigned char  m_type;
    /// Version identifier (must be >= 0)
    unsigned char  m_version;
    /// Source ID (valid source IDs are > 0; invalid ones 0)
    short          m_sourceID;
    /// Opaque data block
    unsigned int   m_data[1];
  }; // class RawBank

} // namespace LHCb

#endif ///DAQEvent_RawBank_H
