//
// See implementation or Doxygen comments for more information
//
#ifndef ALIAODFORWARDHEADER_H
#define ALIAODFORWARDHEADER_H
/**
 * @file   AliAODForwardHeader.h
 * @author Christian Holm Christensen <cholm@nbi.dk>
 * @date   Mon Dec  2 09:31:05 2013
 * 
 * @brief  Header for forward data in stand-alone AOD
 * 
 * @ingroup pwglf_forward_aod 
 */
#include <TObject.h>
class TBrowser;
class TH1;
class TH1I;

class AliAODForwardHeader : public TObject
{
public:
  /** 
   * Bits of the trigger pattern
   */
  enum { 
    /** In-elastic collision */
    kInel        = 0x0001, 
    /** In-elastic collision with at least one SPD tracklet */
    kInelGt0     = 0x0002, 
    /** Non-single diffractive collision */
    kNSD         = 0x0004, 
    /** Empty bunch crossing */
    kEmpty       = 0x0008, 
    /** A-side trigger */
    kA           = 0x0010, 
    /** B(arrel) trigger */
    kB           = 0x0020, 
    /** C-side trigger */
    kC           = 0x0080,  
    /** Empty trigger */
    kE           = 0x0100,
    /** pileup from SPD */
    kPileUp      = 0x0200,    
    /** true NSD from MC */
    kMCNSD       = 0x0400,    
    /** Offline MB triggered */
    kOffline     = 0x0800,
    /** At least one SPD cluster */ 
    kNClusterGt0 = 0x1000,
    /** V0-AND trigger */
    kV0AND       = 0x2000, 
    /** Satellite event */
    kSatellite   = 0x4000
  };
  /** 
   * Bin numbers in trigger histograms 
   */
  enum { 
    kBinAll=1,
    kBinInel, 
    kBinInelGt0, 
    kBinNSD, 
    kBinV0AND,
    kBinA, 
    kBinB, 
    kBinC, 
    kBinE,
    kBinSatellite,
    kBinPileUp, 
    kBinMCNSD,
    kBinOffline,
    kBinNClusterGt0,
    kWithTrigger, 
    kWithVertex, 
    kAccepted
  };
  /**
   * Return codes of CheckEvent 
   */
  enum ECheckStatus {
    /** Event accepted by cuts */
    kGoodEvent = 0, 
    /** Event centrality not in range */
    kWrongCentrality, 
    /** Event trigger isn't in the supplied mask */
    kWrongTrigger, 
    /** Event is a pile-up event */
    kIsPileup, 
    /** Event has no interaction point information */
    kNoVertex, 
    /** Event interaction point is out of range */
    kWrongVertex
  };
  /** 
   * Constructor
   */
  AliAODForwardHeader() 
    : fTriggers(0), fIpZ(fgkInvalidIpZ), fCentrality(-1), fNClusters(0)
  {}

  /** 
   * @{ 
   * @name Setters of data 
   */
  /** 
   * Clear all data 
   * 
   * @param option  Not used
   */
  void Clear(Option_t* option="") override;
  /** 
   * Set the trigger mask 
   * 
   * @param trg Trigger mask
   */
  void SetTriggerMask(UInt_t trg) { fTriggers = trg; } // Set triggers 
  /** 
   * Set bit(s) in trigger mask 
   * 
   * @param bits bit(s) to set 
   */
  void SetTriggerBits(UInt_t bits) { fTriggers |= bits; } // Set trigger bits
  /** 
   * Set the z coordinate of the interaction point
   * 
   * @param ipZ Interaction point z coordinate
   */
  void SetIpZ(Float_t ipZ) { fIpZ = ipZ; } // Set Ip's Z coordinate
  /** 
   * Set the event centrality 
   * 
   * @param c Centrality 
   */
  void SetCentrality(Float_t c) { fCentrality = c; }
  /** 
   * Set the number of SPD clusters seen in @f$ |\eta|<1@f$ 
   * 
   * @param n Number of SPD clusters 
   */
  void SetNClusters(UShort_t n) { fNClusters = n; }
  /* @} */

  
  /** 
   * @{ 
   * @name Tests 
   */
  /** 
   * Check if all bit(s) are set in the trigger mask.  Note, this is
   * an @e and between the bits.  If you need an @e or you should use
   * the member function IsTriggerOrBits
   * 
   * @param bits Bits to test for 
   * 
   * @return true if all enabled bits in the argument is also set in
   * the trigger word
   */
  Bool_t IsTriggerBits(UInt_t bits) const;
  /** 
   * Check if any of bit(s) are enabled in the trigger word.  This is
   * an @e or between the selected bits.  If you need and @a and you
   * should use the member function IsTriggerBits;
   * 
   * @param bits Bits to check for 
   * 
   * @return true if any of the enabled bits in the arguments are also
   * enabled in the trigger mask
   */
  Bool_t IsTriggerOrBits(UInt_t bits) const;
  /** 
   * Whether we have any trigger bits 
   *
   * @return true if we have some trigger 
   */
  Bool_t HasTrigger() const { return fTriggers != 0; } // Check for triggers
  /** 
   * Check if we have a valid z coordinate of the interaction point
   *
   * @return True if we have a valid interaction point z coordinate
   */
  Bool_t HasIpZ() const;
  /** 
   * Check if the z coordinate of the interaction point is within the
   * given limits.  Note that the convention used corresponds to the
   * convention used in ROOTs TAxis.
   * 
   * @param low  Lower cut (inclusive)
   * @param high Upper cut (exclusive)
   * 
   * @return true if @f$ low \ge ipz < high@f$ 
   */
  Bool_t InRange(Float_t low, Float_t high) const;
  /** 
   * Check if we have a valid centrality 
   * 
   * @return True if the centrality is set 
   */
  Bool_t  HasCentrality() const { return !(fCentrality  < 0); }
  /** 
   * Check if event meets the passses requirements.   
   *
   * It returns true if @e all of the following is true 
   *
   * - The trigger is within the bit mask passed.
   * - The vertex is within the specified limits. 
   * - The centrality is within the specified limits, or if lower
   *   limit is equal to or larger than the upper limit.
   * 
   * Note, for data with out a centrality estimate (e.g., pp), one
   * must pass equal centrality cuts, or no data will be accepted.  In
   * other words, for pp data, always pass cMin=0, cMax=0
   *
   * If a histogram is passed in the last parameter, then that
   * histogram is filled with the trigger bits. 
   * 
   * @param triggerMask  Trigger mask
   * @param vzMin        Minimum @f$ v_z@f$ (in centimeters)
   * @param vzMax        Maximum @f$ v_z@f$ (in centimeters) 
   * @param cMin         Minimum centrality (in percent)
   * @param cMax         Maximum centrality (in percent)
   * @param hist         Histogram to fill 
   * @param status       Histogram to fill 
   * 
   * 
   * @return @c true if the event meets the requirements 
   */
  Bool_t CheckEvent(Int_t    triggerMask=kInel,
		    Double_t vzMin=-10, Double_t vzMax=10,
		    UShort_t cMin=0,    UShort_t cMax=100, 
		    TH1*     hist=0,
		    TH1*     status=0) const;
  /* @} */

  /** 
   * @{ 
   * @name Getters of data 
   */
  /** 
   * Get the trigger bits 
   * 
   * @return Trigger bits 
   */
  UInt_t GetTriggerBits() const { return fTriggers; } // Get triggers
  /** 
   * Set the z coordinate of the interaction point
   * 
   * @return Interaction point z coordinate
   */
  Float_t GetIpZ() const { return fIpZ; } // Get Ip's Z coordinate 
  /** 
   * Get the event centrality 
   * 
   * @return The event centrality or -1 if not set
   */
  Float_t GetCentrality() const { return fCentrality; }
  /** 
   * Get the number of SPD clusters seen in @f$ |\eta|<1@f$ 
   * 
   * @return Number of SPD clusters seen
   */
  UShort_t GetNClusters() const { return fNClusters; }
  /* @} */

  /** 
   * @{ 
   * @name Standard TObject member functions 
   */
  /** 
   * Print content 
   * 
   * @param option Passed verbatim to TH2::Print 
   */
  void Print(Option_t* option="") const override;
  /** 
   * Get the name of the object 
   * 
   * @return Name of object 
   */
  const Char_t* GetName() const override { return "ForwardHeader"; }
  /** 
   * browse this object 
   * 
   * @param b Browser 
   */
  void Browse(TBrowser* b) override;
  /** 
   * This is a folder 
   * 
   * @return Always true
   */
  Bool_t IsFolder() const override { return kTRUE; } // Always true 
  /* @} */
  /** 
   * @{
   * @name Utility class functions 
   */
  /** 
   * Get a string correspondig to the trigger mask
   * 
   * @param mask Trigger mask 
   * 
   * @return Static string (copy before use)
   */
  static const Char_t* GetTriggerString(UInt_t mask);
  /** 
   * Make a histogram to record triggers in. 
   *
   * The bins defined by the trigger enumeration in this class.  One
   * can use this enumeration to retrieve the number of triggers for
   * each class.
   * 
   * @param name Name of the histogram 
   * @param mask Trigger mask 
   * 
   * @return Newly allocated histogram 
   */
  static TH1I* MakeTriggerHistogram(const char* name="triggers",
				    Int_t mask=0);
  /** 
   * Make a histogram to record status in. 
   *
   * The bins defined by the status enumeration in this class.  
   * 
   * @param name Name of the histogram 
   * 
   * @return Newly allocated histogram 
   */
  static TH1I* MakeStatusHistogram(const char* name="status");
  /** 
   * Utility function to make a trigger mask from the passed string. 
   * 
   * The string is a comma or space seperated list of case-insensitive
   * strings
   * 
   * - INEL 
   * - INEL>0
   * - NSD 
   * 
   * @param what Which triggers to put in the mask. 
   * 
   * @return The generated trigger mask. 
   */
  static UInt_t MakeTriggerMask(const char* what);
  /* @} */
protected: 
  /** Trigger bits */
  UInt_t   fTriggers;   // Trigger bit mask 
  /** Interaction point @f$z@f$ coordinate */
  Float_t  fIpZ;        // Z coordinate of the interaction point
  /** Centrality */
  Float_t  fCentrality; // Event centrality 
  /** Number of clusters in @f$|\eta|<1@f$ */
  UShort_t fNClusters;  // Number of SPD clusters in |eta|<1
  /** Invalid value for interaction point @f$z@f$ coordiante */
  static const Float_t fgkInvalidIpZ; // Invalid IpZ value 
  ClassDefOverride(AliAODForwardHeader,1); // AOD forward header 
};

//____________________________________________________________________
inline Bool_t
AliAODForwardHeader::InRange(Float_t low, Float_t high) const 
{
  return HasIpZ() && fIpZ >= low && fIpZ < high;
}

//____________________________________________________________________
inline Bool_t 
AliAODForwardHeader::IsTriggerBits(UInt_t bits) const 
{ 
  return HasTrigger() && ((fTriggers & bits) == bits); 
}
//____________________________________________________________________
inline Bool_t 
AliAODForwardHeader::IsTriggerOrBits(UInt_t bits) const 
{ 
  return HasTrigger() && ((fTriggers & bits) != 0);
}


#endif
// Local Variables:
//  mode: C++
// End:

