
//   **************************************************************************
//   *                                                                        *
//   *                      ! ! ! A T T E N T I O N ! ! !                     *
//   *                                                                        *
//   *  This file was created automatically by GaudiObjDesc, please do not    *
//   *  delete it or edit it by hand.                                         *
//   *                                                                        *
//   *  If you want to change this file, first change the corresponding       *
//   *  xml-file and rerun the tools from GaudiObjDesc (or run make if you    *
//   *  are using it from inside a Gaudi-package).                            *
//   *                                                                        *
//   **************************************************************************

#ifndef DAQEvent_ODIN_H
#define DAQEvent_ODIN_H 1

// Include files
#include "RawBank.h"
#include "DataObject.h"
#include <ostream>

// Forward declarations

#define ulonglong unsigned long long

namespace LHCb 
{

  // Forward declarations
  
  // Class ID definition
  namespace ODINLocation {
    static const std::string& Default = "DAQ/ODIN";
  }
  

  /** @class ODIN ODIN.h
   *
   * Class for the decoding of the ODIN RawBank. 
   *
   * @author Marco Clemencic
   * created Thu Nov 29 14:52:20 2007
   * 
   */

  class ODIN: public DataObject
  {
  public:

    /// Fields in the ODIN bank
    enum Data{ RunNumber = 0,
               EventType,
               OrbitNumber,
               L0EventIDHi,
               L0EventIDLo,
               GPSTimeHi,
               GPSTimeLo,
               Word7,
               Word8
      };
    /// 
    enum EventTypeBitsEnum{ EventTypeBits = 0
      };
    /// 
    enum EventTypeMasks{ EventTypeMask = 0xFF
      };
    /// 
    enum Word7Bits{ DetectorStatusBits = 0,
                    ErrorBits          = 24
      };
    /// 
    enum ErrorCodeMasks{ SynchError       = 0x1,
                         SynchErrorForced = 0x2
      };
    /// 
    enum Word7Masks{ DetectorStatusMask = 0x00FFFFFF,
                     ErrorMask          = 0xFF000000
      };
    /// 
    enum Word8Bits{ BunchIDBits      = 0,
                    TriggerTypeBits  = 16,
                    ReadoutTypeBits  = 19,
                    ForceBits        = 21,
                    BXTypeBits       = 22,
                    BunchCurrentBits = 24
      };
    /// 
    enum Word8Masks{ BunchIDMask      = 0x00000FFF,
                     TriggerTypeMask  = 0x00070000,
                     ReadoutTypeMask  = 0x00180000,
                     ForceMask        = 0x00200000,
                     BXTypeMask       = 0x00C00000,
                     BunchCurrentMask = 0xFF000000
      };
    /// 
    enum ReadoutTypes{ ZeroSuppressed    = 0,
                       NonZeroSuppressed = 1
      };
    /// 
    enum BXTypes{ NoBeam          = 0,
                  SingleBeamLeft  = 1,
                  SingleBeamRight = 2,
                  BeamCrossing    = 3
      };
  
    /// Copy constructor. Creates a new ODIN object with the same information
    ODIN(const LHCb::ODIN& odin) : DataObject(),
                                   m_runNumber(odin.runNumber()),
                                   m_eventType(odin.eventType()),
                                   m_orbitNumber(odin.orbitNumber()),
                                   m_eventNumber(odin.eventNumber()),
                                   m_gpsTime(odin.gpsTime()),
                                   m_detectorStatus(odin.detectorStatus()),
                                   m_errorBits(odin.errorBits()),
                                   m_bunchId(odin.bunchId()),
                                   m_forceBit(odin.forceBit()),
                                   m_bunchCurrent(odin.bunchCurrent()) {}
  
    /// Default Constructor
    ODIN() : m_runNumber(0),
             m_eventType(0),
             m_orbitNumber(0),
             m_eventNumber(0),
             m_gpsTime(0),
             m_detectorStatus(0),
             m_errorBits(0),
             m_bunchId(0),
             m_triggerType(0),
             m_readoutType(ZeroSuppressed),
             m_forceBit(false),
             m_bunchCrossingType(NoBeam),
             m_bunchCurrent(0) {}
  
    /// Default Destructor
    virtual ~ODIN() {}
  
    /// Fill the ASCII output stream
   virtual std::ostream& fillStream(std::ostream& s) const;
  
    /// Retrieve const  Run number
    unsigned int runNumber() const;
  
    /// Update  Run number
    void setRunNumber(unsigned int value);
  
    /// Retrieve const  Event type
    unsigned int eventType() const;
  
    /// Update  Event type
    void setEventType(unsigned int value);
  
    /// Retrieve const  Orbit ID
    unsigned int orbitNumber() const;
  
    /// Update  Orbit ID
    void setOrbitNumber(unsigned int value);
  
    /// Retrieve const  L0 Event ID
    ulonglong eventNumber() const;
  
    /// Update  L0 Event ID
    void setEventNumber(ulonglong value);
  
    /// Retrieve const  GPS Time (microseconds)
    ulonglong gpsTime() const;
  
    /// Update  GPS Time (microseconds)
    void setGpsTime(ulonglong value);
  
    /// Retrieve const  Detector Status
    ulonglong detectorStatus() const;
  
    /// Update  Detector Status
    void setDetectorStatus(ulonglong value);
  
    /// Retrieve const  Error Bits
    unsigned int errorBits() const;
  
    /// Update  Error Bits
    void setErrorBits(unsigned int value);
  
    /// Retrieve const  Bunch ID
    unsigned int bunchId() const;
  
    /// Update  Bunch ID
    void setBunchId(unsigned int value);
  
    /// Retrieve const  Trigger Type
    unsigned int triggerType() const;
  
    /// Update  Trigger Type
    void setTriggerType(unsigned int value);
  
    /// Retrieve const  Force Bit
    bool forceBit() const;
  
    /// Update  Force Bit
    void setForceBit(bool value);
  
    /// Retrieve const  Bunch Current
    unsigned int bunchCurrent() const;
  
    /// Update  Bunch Current
    void setBunchCurrent(unsigned int value);
  
  protected:

  private:

    unsigned int m_runNumber;         ///< Run number
    unsigned int m_eventType;         ///< Event type
    unsigned int m_orbitNumber;       ///< Orbit ID
    ulonglong    m_eventNumber;       ///< L0 Event ID
    ulonglong    m_gpsTime;           ///< GPS Time (microseconds)
    ulonglong    m_detectorStatus;    ///< Detector Status
    unsigned int m_errorBits;         ///< Error Bits
    unsigned int m_bunchId;           ///< Bunch ID
    unsigned int m_triggerType;       ///< Trigger Type
      //    ReadoutTypes m_readoutType;       ///< Readout Type (@see enum LHCb::ODIN::ReadoutTypes)
    int          m_readoutType;       ///< Readout Type (@see enum LHCb::ODIN::ReadoutTypes)
    bool         m_forceBit;          ///< Force Bit
      //    BXTypes      m_bunchCrossingType; ///< Bunch Crossing Type (BXType, @see enum LHCb::ODIN::BXTypes)
    int m_bunchCrossingType;          ///< Bunch Crossing Type (BXType, @see enum LHCb::ODIN::BXTypes)
    unsigned int m_bunchCurrent;      ///< Bunch Current
  
  }; // class ODIN

  inline std::ostream& operator<< (std::ostream& str, const ODIN& obj)
  {
    return obj.fillStream(str);
  }
  
  inline std::ostream & operator << (std::ostream & s, LHCb::ODIN::Data e) {
    switch (e) {
      case LHCb::ODIN::RunNumber   : return s << "RunNumber";
      case LHCb::ODIN::EventType   : return s << "EventType";
      case LHCb::ODIN::OrbitNumber : return s << "OrbitNumber";
      case LHCb::ODIN::L0EventIDHi : return s << "L0EventIDHi";
      case LHCb::ODIN::L0EventIDLo : return s << "L0EventIDLo";
      case LHCb::ODIN::GPSTimeHi   : return s << "GPSTimeHi";
      case LHCb::ODIN::GPSTimeLo   : return s << "GPSTimeLo";
      case LHCb::ODIN::Word7       : return s << "Word7";
      case LHCb::ODIN::Word8       : return s << "Word8";
      default : return s << "ERROR wrong value for enum LHCb::ODIN::Data";
    }
  }
  
  inline std::ostream & operator << (std::ostream & s, LHCb::ODIN::EventTypeBitsEnum e) {
    switch (e) {
      case LHCb::ODIN::EventTypeBits : return s << "EventTypeBits";
      default : return s << "ERROR wrong value for enum LHCb::ODIN::EventTypeBitsEnum";
    }
  }
  
  inline std::ostream & operator << (std::ostream & s, LHCb::ODIN::EventTypeMasks e) {
    switch (e) {
      case LHCb::ODIN::EventTypeMask : return s << "EventTypeMask";
      default : return s << "ERROR wrong value for enum LHCb::ODIN::EventTypeMasks";
    }
  }
  
  inline std::ostream & operator << (std::ostream & s, LHCb::ODIN::Word7Bits e) {
    switch (e) {
      case LHCb::ODIN::DetectorStatusBits : return s << "DetectorStatusBits";
      case LHCb::ODIN::ErrorBits          : return s << "ErrorBits";
      default : return s << "ERROR wrong value for enum LHCb::ODIN::Word7Bits";
    }
  }
  
  inline std::ostream & operator << (std::ostream & s, LHCb::ODIN::ErrorCodeMasks e) {
    switch (e) {
      case LHCb::ODIN::SynchError       : return s << "SynchError";
      case LHCb::ODIN::SynchErrorForced : return s << "SynchErrorForced";
      default : return s << "ERROR wrong value for enum LHCb::ODIN::ErrorCodeMasks";
    }
  }
  
  inline std::ostream & operator << (std::ostream & s, LHCb::ODIN::Word7Masks e) {
    switch (e) {
      case LHCb::ODIN::DetectorStatusMask : return s << "DetectorStatusMask";
      case LHCb::ODIN::ErrorMask          : return s << "ErrorMask";
      default : return s << "ERROR wrong value for enum LHCb::ODIN::Word7Masks";
    }
  }
  
  inline std::ostream & operator << (std::ostream & s, LHCb::ODIN::Word8Bits e) {
    switch (e) {
      case LHCb::ODIN::BunchIDBits      : return s << "BunchIDBits";
      case LHCb::ODIN::TriggerTypeBits  : return s << "TriggerTypeBits";
      case LHCb::ODIN::ReadoutTypeBits  : return s << "ReadoutTypeBits";
      case LHCb::ODIN::ForceBits        : return s << "ForceBits";
      case LHCb::ODIN::BXTypeBits       : return s << "BXTypeBits";
      case LHCb::ODIN::BunchCurrentBits : return s << "BunchCurrentBits";
      default : return s << "ERROR wrong value for enum LHCb::ODIN::Word8Bits";
    }
  }
  
  inline std::ostream & operator << (std::ostream & s, LHCb::ODIN::Word8Masks e) {
    switch (e) {
      case LHCb::ODIN::BunchIDMask      : return s << "BunchIDMask";
      case LHCb::ODIN::TriggerTypeMask  : return s << "TriggerTypeMask";
      case LHCb::ODIN::ReadoutTypeMask  : return s << "ReadoutTypeMask";
      case LHCb::ODIN::ForceMask        : return s << "ForceMask";
      case LHCb::ODIN::BXTypeMask       : return s << "BXTypeMask";
      case LHCb::ODIN::BunchCurrentMask : return s << "BunchCurrentMask";
      default : return s << "ERROR wrong value for enum LHCb::ODIN::Word8Masks";
    }
  }
  
  inline std::ostream & operator << (std::ostream & s, LHCb::ODIN::ReadoutTypes e) {
    switch (e) {
      case LHCb::ODIN::ZeroSuppressed    : return s << "ZeroSuppressed";
      case LHCb::ODIN::NonZeroSuppressed : return s << "NonZeroSuppressed";
      default : return s << "ERROR wrong value for enum LHCb::ODIN::ReadoutTypes";
    }
  }
  
  inline std::ostream & operator << (std::ostream & s, LHCb::ODIN::BXTypes e) {
    switch (e) {
      case LHCb::ODIN::NoBeam          : return s << "NoBeam";
      case LHCb::ODIN::SingleBeamLeft  : return s << "SingleBeamLeft";
      case LHCb::ODIN::SingleBeamRight : return s << "SingleBeamRight";
      case LHCb::ODIN::BeamCrossing    : return s << "BeamCrossing";
      default : return s << "ERROR wrong value for enum LHCb::ODIN::BXTypes";
    }
  }
  
  
} // namespace LHCb;

// -----------------------------------------------------------------------------
// end of class
// -----------------------------------------------------------------------------

// Including forward declarations

inline std::ostream& LHCb::ODIN::fillStream(std::ostream& s) const
{
  char l_forceBit = (m_forceBit) ? 'T' : 'F';
  s << "{ " << "runNumber :	" << m_runNumber << std::endl
            << "eventType :	" << m_eventType << std::endl
            << "orbitNumber :	" << m_orbitNumber << std::endl
            << "eventNumber :	" << m_eventNumber << std::endl
            << "gpsTime :	" << m_gpsTime << std::endl
            << "detectorStatus :	" << m_detectorStatus << std::endl
            << "errorBits :	" << m_errorBits << std::endl
            << "bunchId :	" << m_bunchId << std::endl
            << "triggerType :	" << m_triggerType << std::endl
            << "readoutType :	" << m_readoutType << std::endl
            << "forceBit :	" << l_forceBit << std::endl
            << "bunchCrossingType :	" << m_bunchCrossingType << std::endl
            << "bunchCurrent :	" << m_bunchCurrent << std::endl << " }";
  return s;
}


inline unsigned int LHCb::ODIN::runNumber() const 
{
  return m_runNumber;
}

inline void LHCb::ODIN::setRunNumber(unsigned int value) 
{
  m_runNumber = value;
}

inline unsigned int LHCb::ODIN::eventType() const 
{
  return m_eventType;
}

inline void LHCb::ODIN::setEventType(unsigned int value) 
{
  m_eventType = value;
}

inline unsigned int LHCb::ODIN::orbitNumber() const 
{
  return m_orbitNumber;
}

inline void LHCb::ODIN::setOrbitNumber(unsigned int value) 
{
  m_orbitNumber = value;
}

inline ulonglong LHCb::ODIN::eventNumber() const 
{
  return m_eventNumber;
}

inline void LHCb::ODIN::setEventNumber(ulonglong value) 
{
  m_eventNumber = value;
}

inline ulonglong LHCb::ODIN::gpsTime() const 
{
  return m_gpsTime;
}

inline void LHCb::ODIN::setGpsTime(ulonglong value) 
{
  m_gpsTime = value;
}

inline ulonglong LHCb::ODIN::detectorStatus() const 
{
  return m_detectorStatus;
}

inline void LHCb::ODIN::setDetectorStatus(ulonglong value) 
{
  m_detectorStatus = value;
}

inline unsigned int LHCb::ODIN::errorBits() const 
{
  return m_errorBits;
}

inline void LHCb::ODIN::setErrorBits(unsigned int value) 
{
  m_errorBits = value;
}

inline unsigned int LHCb::ODIN::bunchId() const 
{
  return m_bunchId;
}

inline void LHCb::ODIN::setBunchId(unsigned int value) 
{
  m_bunchId = value;
}

inline unsigned int LHCb::ODIN::triggerType() const 
{
  return m_triggerType;
}

inline void LHCb::ODIN::setTriggerType(unsigned int value) 
{
  m_triggerType = value;
}

inline bool LHCb::ODIN::forceBit() const 
{
  return m_forceBit;
}

inline void LHCb::ODIN::setForceBit(bool value) 
{
  m_forceBit = value;
}

inline unsigned int LHCb::ODIN::bunchCurrent() const 
{
  return m_bunchCurrent;
}

inline void LHCb::ODIN::setBunchCurrent(unsigned int value) 
{
  m_bunchCurrent = value;
}




#endif ///DAQEvent_ODIN_H
