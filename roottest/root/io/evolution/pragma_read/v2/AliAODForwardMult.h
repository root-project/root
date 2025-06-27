//
// See implementation or Doxygen comments for more information
//
#ifndef ALIAODFORWARDMULT_H
#define ALIAODFORWARDMULT_H
/**
 * @file   AliAODForwardMult.h
 * @author Christian Holm Christensen <cholm@dalsgaard.hehi.nbi.dk>
 * @date   Wed Mar 23 13:58:00 2011
 * 
 * @brief  Per-event @f$ N_{ch}@f$ per @f$(\eta,\varphi)@f$ bin 
 * 
 * @ingroup pwglf_forward_aod
 * 
 */
#include <TObject.h>
#include <TH2D.h>
#include "AliAODForwardHeader.h"
class TBrowser;
class TH1I;
/**
 * Class that contains the forward multiplicity data per event 
 *
 * This class contains a histogram of 
 * @f[
 *   \frac{d^2N_{ch}}{d\eta d\phi}\quad,
 * @f]
 * as well as a trigger mask for each analysed event.  
 * 
 * The eta acceptance of the event is stored in the underflow bins of
 * the histogram.  So to build the final histogram, one needs to
 * correct for this acceptance (properly weighted by the events), and
 * the vertex efficiency.  This simply boils down to defining a 2D
 * histogram and summing the event histograms in that histogram.  One
 * should of course also do proper book-keeping of the accepted event.
 *
 * @code 
 * TTree* GetAODTree()
 * { 
 *    TFile* file = TFile::Open("AliAODs.root","READ");
 *    TTree* tree = static_cast<TTree*>(file->Get("aodTree"));
 *    return tree;
 * }
 * 
 * void Analyse()
 * { 
 *   TH2D*              sum        = 0;                  // Summed hist
 *   TTree*             tree       = GetAODTree();       // AOD tree
 *   AliAODForwardMult* mult       = 0;                  // AOD object
 *   Int_t              nTriggered = 0;                  // # of triggered ev.
 *   Int_t              nWithVertex= 0;                  // # of ev. w/vertex
 *   Int_t              nAccepted  = 0;                  // # of ev. used
 *   Int_t              nAvailable = tree->GetEntries(); // How many entries
 *   Float_t            vzLow      = -10;                // Lower ip cut
 *   Float_t            vzHigh     =  10;                // Upper ip cut
 *   Int_t              mask       = AliAODForwardMult::kInel;// Trigger mask
 *   tree->SetBranchAddress("forward", &forward);        // Set the address
 * 
 *   for (int i = 0; i < nAvailable; i++) { 
 *     // Create sum histogram on first event - to match binning to input
 *     if (!sum) sum = static_cast<TH2D*>(mult->Clone("d2ndetadphi"));
 * 
 *     tree->GetEntry(i);
 * 
 *     // Other trigger/event requirements could be defined 
 *     if (!mult->IsTriggerBits(mask)) continue; 
 *     nTriggered++;
 *
 *     // Check if we have vertex 
 *     if (!mult->HasIpZ()) continue;
 *     nWithVertex++;
 * 
 *     // Select vertex range (in centimeters) 
 *     if (!mult->InRange(vzLow, vzHigh) continue; 
 *     nAccepted++;
 * 
 *     // Add contribution from this event
 *     sum->Add(&(mult->GetHistogram()));
 *   }
 * 
 *   // Get acceptance normalisation from underflow bins 
 *   TH1D* norm   = sum->ProjectionX("norm", 0, 1, "");
 *   // Project onto eta axis - _ignoring_underflow_bins_!
 *   TH1D* dndeta = sum->Projection("dndeta", 1, -1, "e");
 *   // Normalize to the acceptance 
 *   dndeta->Divide(norm);
 *   // Scale by the vertex efficiency 
 *   dndeta->Scale(Double_t(nWithVertex)/nTriggered, "width");
 *   // And draw the result
 *   dndeta->Draw();
 * }
 * @endcode   
 *     
 * The above code will draw the final @f$ dN_{ch}/d\eta@f$ for the
 * selected event class and vertex range
 *
 * The histogram can be used as input for other kinds of analysis too, 
 * like flow, event-plane, centrality, and so on. 
 *
 * @ingroup pwglf_forward 
 * @ingroup pwglf_forward_aod
 */
class AliAODForwardMult : public TObject
{
public:
  /** 
   * User bits of these objects (bits 14-23 can be used)
   */
  enum {
    /** Secondary correction maps where applied */
    kSecondary           = (1 << 14), 
    /** Vertex bias correction was applied */
    kVertexBias          = (1 << 15),  
    /** Acceptance correction was applied */
    kAcceptance          = (1 << 16), 
    /** Merging efficiency correction was applied */
    kMergingEfficiency   = (1 << 17),
    /** Signal in overlaps is the sum */
    kSum                 = (1 << 18), 
    /** Used eta dependent empirical correction - to be implemented */
    kEmpirical           = (1 << 19)
  };
  /** 
   * Default constructor 
   * 
   * Used by ROOT I/O sub-system - do not use
   */
  AliAODForwardMult();
  /** 
   * Constructor 
   * 
   * @param isMC Whether this was from MC or not 
   */
  AliAODForwardMult(Bool_t isMC);
  /** 
   * Destructor 
   */
  virtual ~AliAODForwardMult() {} // Destructor 
  /** 
   * Initialize 
   * 
   * @param etaAxis  Pseudo-rapidity axis
   */
  void Init(const TAxis& etaAxis);
  /** 
   * Set the center of mass energy per nucleon-pair.  This is stored 
   * in the (0,0) of the histogram 
   * 
   * @param sNN Center of mass energy per nucleon pair (GeV)
   */
  void SetSNN(UShort_t sNN); 
  /** 
   * Get the collision system number
   * - 0: Unknown 
   * - 1: pp
   * - 2: PbPb
   * 
   * @param sys Collision system number
   */
  void SetSystem(UShort_t sys);
  /** 
   * Get the @f$ d^2N_{ch}/d\eta d\phi@f$ histogram, 
   *
   * @return @f$ d^2N_{ch}/d\eta d\phi@f$ histogram, 
   */  
  const TH2D& GetHistogram() const { return fHist; } // Get histogram 
  /** 
   * Get the @f$ d^2N_{ch}/d\eta d\phi@f$ histogram, 
   *
   * @return @f$ d^2N_{ch}/d\eta d\phi@f$ histogram, 
   */  
  TH2D& GetHistogram() { return fHist; } // Get histogram 
  /** 
   * Clear all data 
   * 
   * @param option  Passed on to TH2::Reset verbatim
   */
  void Clear(Option_t* option="") override;
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
  /** 
   * @return @c true if secondary corrected
   */
  Bool_t IsSecondaryCorrected() const { return TestBit(kSecondary); }
  /** 
   * @return @c true if vertex-bias corrected
   * @deprecated 
   */
  Bool_t IsVertexBiasCorrected() const { return TestBit(kVertexBias); }
  /** 
   * @return @c true if (fixed) acceptance corrected
   * @deprecated 
   */
  Bool_t IsAcceptanceCorrected() const { return TestBit(kAcceptance); }
  /** 
   * @return @c true if corrected for merging efficiency 
   * @deprecated 
   */
  Bool_t IsMergingEfficiencyCorrected() const { 
    return TestBit(kMergingEfficiency); }
  /** 
   * @return @c true if corrected using an empirical correction for secondaries 
   * @note Not available yet 
   */
  Bool_t IsEmpiricalCorrected() const { return TestBit(kEmpirical); }
  /** 
   * @return @c true if signal in overlaps are straight sums
   */
  Bool_t IsSumSignal() const { return TestBit(kSum); }
  /** 
   * Get the center of mass energy per nucleon pair (GeV)
   * 
   * @return Center of mass energy per nucleon pair (GeV)
   */
  UShort_t GetSNN() const;
  /** 
   * Get the collision system number
   * - 0: Unknown 
   * - 1: pp
   * - 2: PbPb
   * 
   * @return Collision system number
   */
  UShort_t GetSystem() const;
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
  const Char_t* GetName() const override { return (fIsMC ? "ForwardMC" : "Forward"); }
  /** 
   * Create a backward compatiblity header 
   * 
   * @param triggers    Trigger information
   * @param ipZ         Interaction point Z coordinate 
   * @param centrality  Centrality estimate 
   * @param nClusters   Number of clusters in the SPD first layer 
   */
  void CreateHeader(UInt_t   triggers, 
		    Float_t  ipZ, 
		    Float_t  centrality, 
		    UShort_t nClusters);
protected: 
  /** From MC or not */
  Bool_t   fIsMC;       // Whether this is from MC 
  /** Histogram of @f$d^2N_{ch}/(d\eta d\phi)@f$ for this event */
  TH2D     fHist;       // Histogram of d^2N_{ch}/(deta dphi) for this event
  AliAODForwardHeader* fHeader; //! Cached header 

  ClassDefOverride(AliAODForwardMult,6); // AOD forward multiplicity 
};

#endif
// Local Variables:
//  mode: C++
// End:

