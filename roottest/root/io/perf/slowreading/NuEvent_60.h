////////// ///////// ///////// //////// ///////// ///////// ///////// 72
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

#ifndef NUEVENT_H
#define NUEVENT_H

#include <string>

#include "TObject.h"

class NuEvent : public TObject
{
 public:
  NuEvent();

  //non-const methods
  void Reset();

  //////////////////////////////////////////////////////////////////
  /// EVERYTIME A TRUTH VARIABLE IS ADDED TO THIS CLASS IT MUST
  /// ALSO BE ADDED TO NuMCEvent
  ///////////////////////////////////////////////////////////////////

  ///////////////////////////
  //book keeping  quantities
  ///////////////////////////

  Int_t index;//the n'th NuEvent object to be written to the tree
  Int_t entry;//the n'th snarl to be processed 
  
  /////////////////////////////
  //snarl/run based quantities
  /////////////////////////////

  Int_t run;//fHeader.fRun
  Int_t subRun;//fHeader.fSubRun
  Int_t runType;//fHeader.fRunType
  Int_t errorCode;//fHeader.fErrorCode
  Int_t snarl;//actual DAQ snarl number
  Int_t trigSrc;//fHeader.fTrigSrc
  Int_t timeFrame;//fHeader.fTimeFrame
  Int_t remoteSpillType;//fHeader.fRemoteSpillType
  
  Int_t detector;//fHeader.fVldContext.fDetector
  Int_t simFlag;//fHeader.fVldContext.fSimFlag
  Int_t timeSec;//fHeader.fVldContext.fTimeStamp.fSec
  Int_t timeNanoSec;//fHeader.fVldContext.fTimeStamp.fNanoSec
  Double_t timeSeconds;//VldTimeStamp.GetSeconds() sec+nanoSec/1e9

  Double_t trigtime;//evthdr.trigtime
  Double_t medianTime;//the median time in event (wrt timeFrame start)
  Double_t timeEvtMin;//earliest time in event (wrt timeNanoSec)
  Double_t timeEvtMax;//latest time in event (wrt timeNanoSec)

  Int_t nearestSpillSec;//time of nearest spill (sec)
  Int_t nearestSpillNanosec; //time of nearest spill(nsec);
  Double_t timeToNearestSpill; //time to nearest spill (uses SpillTimeFinder)

  Int_t planeEvtHdrBeg;//evthdr.plane.beg (diff. to evt.plane.beg!)
  Int_t planeEvtHdrEnd;//evthdr.plane.end (diff. to evt.plane.end!)
  Double_t snarlPulseHeight;//evthdr.ph.sigcor

  //////////////////////////
  // data quality variables
  //////////////////////////

  Bool_t isGoodDataQuality;//uses DataUtil::IsGoodData()
  Bool_t isGoodDataQualityRUN;//uses DataUtil::IsGoodDataRUN()
  Bool_t isGoodDataQualityCOIL;//uses DataUtil::IsGoodDataCOIL()
  Bool_t isGoodDataQualityHV;//uses DataUtil::IsGoodDataHV()
  Bool_t isGoodDataQualityGPS;//uses DataUtil::IsGoodDataGPS()

  Int_t numActiveCrates;// number of active readout crates ('cratemask')
  Int_t numTimeFrames;// number of timeframes, gives run duration
  Int_t numGoodSnarls;// number of 'good' snarls during run
  Float_t snarlRateMedian;// median snarl rate during run
  Float_t snarlRateMax;// maximum snarl rate during run

  Int_t deltaSecToSpillGPS;//time from trigger to nearest spill (sec)
  Int_t deltaNanoSecToSpillGPS;//time from trigger to nearest spill (nsec)
  Int_t gpsError;//GPS error (nsec)
  Int_t gpsSpillType;// type of spill 

  Bool_t coilIsOk;//CoilTools::IsOk(vldc);
  Bool_t coilIsReverse;//CoilTools::IsReverse(vldc);
  Float_t coilCurrent;//detector bfld coil current (not always filled)

  Bool_t isLI;//LISieve::IsLI()
  Int_t litag;//CC analysis definition: reco.FDRCBoundary(nu);
  Double_t litime;//evthdr.litime (==-1 if not LI)
  
  ///////////////////////////
  //reconstructed quantities
  ///////////////////////////
  
  //these variables are here to add a level of indirection, they don't store
  //information directly from the sntp files, but rather they can be set at  
  //analysis-time to what the user wants them to contain
  //e.g. shwEn could be set to the deweighted, energy corrected, CC shower energy
  //if the analysis selected the event as a CC event. Alternatively,
  //if the analysis
  //selected the event as NC then shwEn could be set to the linear NC,
  //no energy correction, NC shower energy.
  //Don't rely on the defaults, but they typically assume you have a CC event and 
  //want linear, energy corrected values of the energy

  //basic neutrino variables
  Float_t energy;//reconstructed energy
  Float_t energyCC;//reco'd energy assuming it is a CC event, default value of energy
  Float_t energyNC;//reco'd energy assuming it is a NC event (shower only)
  Float_t energyRM;//reco'd energy of rock muon event (default value: track energy only)
  Float_t trkEn;//track energy (range or curv. depending on containment)
  Float_t trkEnRange;//track energy from range
  Float_t trkEnCurv;//track energy from curvature
  Float_t shwEn;//shower energy (e.g. primary shw energy)
  Float_t shwEnCC;//shower energy of event assuming it is CC, default value of shwEn
  Float_t shwEnNC;//shower energy of event assuming it is NC
  
  //basic kinematics
  Float_t y;//reco'd y = (Enu-Emu)/Enu
  Float_t q2;//reco'd q2 = -q^{2} = 2 Enu Emu (1 - cos(theta_mu))
  Float_t x;//reco'd x = Q2 / (2M* Ehad)  {M=nucleon mass}
  Float_t w2;//reco'd w2 = M^{2} + q^{2} + 2M(Enu-Emu)
  Float_t dirCosNu;//reco'd direction cosine of track

  ////////////////////////////////////////
  //event info extracted from the ntuples
  Int_t evt;//reco'd event number, evt.index
  Int_t slc;//reco'd slice number, evt.slc
  Int_t nevt;// (N of events in snarl)
  Int_t ndigitEvt;//evt.ndigit
  Int_t nstripEvt;//evt.nstrip
  Int_t nshw;//evt.nshower
  Int_t ntrk;//evt.ntrack
  Int_t primshw;//evt.primshw
  Int_t primtrk;//evt.primtrk
  Float_t rawPhEvt;//evt.ph.raw
  Float_t evtphsigcor;//evt.ph.sigcor
  Float_t evtphsigmap;//evt.ph.sigmap
  Int_t planeEvtN;       //number of planes in event
  Int_t planeEvtNu;//evt.plane.nu
  Int_t planeEvtNv;//evt.plane.nv

  //these RO PID variables store the values obtained using the 
  //NtpSREvent. Rustem's code defines which trk to use by using  
  //the number of "active" planes
  Float_t roIDEvt;//RO's PID variable (got using the evt)
  Float_t knn01TrkActivePlanesEvt;//number of active planes in trk
  Float_t knn10TrkMeanPhEvt;//average ph per plane in trk
  Float_t knn20LowHighPhEvt;//av of low ph strips / av of high ph strips
  Float_t knn40TrkPhFracEvt;//fraction of ph in trk
  Float_t roIDNuMuBarEvt;//RO's PID NuMuBar selection (0 or 1)
  Float_t relativeAngleEvt;//RO's track angle relative to muon dir.

  //----------------------------
  //  Jasmine new variables (Event)
  //----------------------------
  Float_t JMIDEvt;
  Float_t JMntrkplEvt;
  Float_t JMendphEvt;
  Float_t JMmeanphEvt;
  Float_t JMscatuEvt;
  Float_t JMscatvEvt;
  Float_t JMscatuvEvt;
  Float_t JMtrkqpEvt;
  Float_t JMeknnIDEvt;
  Float_t JMeknn208Evt;
  Float_t JMeknn205Evt;
  Float_t JMeknn204Evt;

  //----------------------------
  Float_t xEvtVtx;//evt.vtx.x
  Float_t yEvtVtx;//evt.vtx.y
  Float_t zEvtVtx;//evt.vtx.z
  Float_t uEvtVtx;//evt.vtx.u
  Float_t vEvtVtx;//evt.vtx.v
  Int_t planeEvtVtx;//evt.vtx.plane
  Int_t planeEvtBeg;//evt.plane.beg
  Int_t planeEvtBegu;//evt.plane.begu
  Int_t planeEvtBegv;//evt.plane.begv
  
  Float_t xEvtEnd;//evt.end.x
  Float_t yEvtEnd;//evt.end.y
  Float_t zEvtEnd;//evt.end.z
  Float_t uEvtEnd;//evt.end.u
  Float_t vEvtEnd;//evt.end.v
  Int_t planeEvtEnd;//evt.plane.end
  Int_t planeEvtEndu;//evt.plane.endu
  Int_t planeEvtEndv;//evt.plane.endv

  /////////////////////////////////////////////////////////
  //these are the variables of the "best" track and shower
  Bool_t trkExists;//simply state if track exists
  Int_t trkIndex;//trk.index, position in TClonesArray in sntp file
  Int_t ndigitTrk;//trk.ndigit
  Int_t nstripTrk;//trk.nstrip
  Float_t trkEnCorRange;//trk energy from range (EnCor applied)
  Float_t trkEnCorCurv;//trk energy from curvature (EnCor applied)
  Float_t trkShwEnNear; // Shower energy 'near' the track vertex
  Float_t trkMomentumRange;//trk momentum from range (no EnCor)
  Float_t trkMomentumTransverse; // trkmom*sin(acos(dirCosNu));
  Int_t containedTrk;//trk.contained
  Int_t trkfitpass;//trk.fit.pass
  Float_t trkvtxdcosz;//trk.vtx.dcosz
  Float_t trkvtxdcosy;//trk.vtx.dcosy
  Int_t trknplane;//trk.nplane
  Int_t charge;//+1 or -1: simply derived from qp (from trk fit)
  Float_t qp;//trk Q/P from fit (no EnCor)
  Float_t qp_rangebiased;//trk Q/P from fit (with range bias)
  Float_t sigqp;//trk sigmaQ/P from fit
  Float_t qp_sigqp; // qp / sigqp
  Float_t chi2;//trk chi2 of fit
  Float_t ndof;//trk ndof of fit
  Float_t qpFraction;//trk.stpfitqp[i], npositive/nstrip
  Int_t trkVtxUVDiffPl;//trk.plane.begu-trk.plane.begv;
  Int_t trkLength;//abs(trk.plane.end-trk.plane.beg+1);
  Int_t planeTrkNu;//trk.plane.nu: number of u planes hit
  Int_t planeTrkNv;//trk.plane.nv: number of v planes hit
  Int_t ntrklike;//the number of trk-like planes
  Float_t trkphsigcor;//trk.ph.sigcor
  Float_t trkphsigmap;//trk.ph.sigmap

  Int_t trkfitpassSA;//variables from the SA track fitter
  Float_t trkvtxdcoszSA;//fitsa.fit.dcosz 
  Int_t chargeSA;//definitions same as variables without SA postfix
  Float_t qpSA;
  Float_t sigqpSA;
  Float_t chi2SA;
  Float_t ndofSA;
  Float_t probSA;
  Float_t xTrkVtxSA;//calculated from u&v
  Float_t yTrkVtxSA;//calculated from u&v
  Float_t zTrkVtxSA;//fitsa.fit.z
  Float_t uTrkVtxSA;//fitsa.fit.u
  Float_t vTrkVtxSA;//fitsa.fit.v

  Float_t jitter;//from MajCInfo
  Float_t jPID;//from MajCInfo
  Float_t majC;//from MajCInfo
  Float_t smoothMajC;//from MajCInfo
  
  //The best RO PID (roID) variables are located with the other PIDs
  //see below

  Float_t xTrkVtx;//trk.vtx.x
  Float_t yTrkVtx;//trk.vtx.y
  Float_t zTrkVtx;//trk.vtx.z
  Float_t uTrkVtx;//trk.vtx.u
  Float_t vTrkVtx;//trk.vtx.v
  Int_t planeTrkVtx;//trk.vtx.plane
  Int_t planeTrkBeg;//trk.plane.beg
  Int_t planeTrkBegu;//trk.plane.begu
  Int_t planeTrkBegv;//trk.plane.begv
  // Strip geometry information
  Int_t stripTrkBeg;  // First strip hit
  Int_t stripTrkBegu; // First strip hit in a u plane
  Int_t stripTrkBegv; // First strip hit in a v plane
  Bool_t stripTrkBegIsu; // True if the first strip hit in this track was in a u plane
  Int_t regionTrkVtx; // What region? See enum Rock_DetectorRegion
  Float_t phiTrkVtx; // Location of the track vertex in Phi = atan(vtx.x / vtx.y)

  Float_t xTrkEnd;//trk.end.x
  Float_t yTrkEnd;//trk.end.y
  Float_t zTrkEnd;//trk.end.z
  Float_t uTrkEnd;//trk.end.u
  Float_t vTrkEnd;//trk.end.v
  Int_t planeTrkEnd;//trk.plane.end
  Int_t planeTrkEndu;//trk.plane.endu
  Int_t planeTrkEndv;//trk.plane.endv

  Float_t drTrkFidall;//trk.fidall.dr
  Float_t dzTrkFidall;//trk.fidall.dz
  Float_t drTrkFidvtx;//trk.fidbeg.dr
  Float_t dzTrkFidvtx;//trk.fidbeg.dz
  Float_t drTrkFidend;//trk.fidend.dr
  Float_t dzTrkFidend;//trk.fidend.dz
  Float_t traceTrkFidall; //trk.fidall.trace
  Float_t traceTrkFidvtx; //trk.fidvtx.trace
  Float_t traceTrkFidend; //trk.fidend.trace
  
  Float_t cosPrTrkVtx; // Cos of angle between theta and radial momentum
  
  //shw variables
  Bool_t shwExists;//simply state if shw exists
  Int_t ndigitShw;//shw.ndigit
  Int_t nstripShw;//shw.nstrip
  Int_t nplaneShw;//shw.plane.n
  Float_t shwEnCor;//shower energy (EnCor applied)
  Float_t shwEnNoCor;//shower energy (no EnCor)
  Float_t shwEnMip;//shower energy (no EnCor)
  Float_t shwEnLinCCNoCor;//shw.shwph.linCCgev
  Float_t shwEnLinCCCor;//EnCor applied
  Float_t shwEnWtCCNoCor;//shw.shwph.wtCCgev
  Float_t shwEnWtCCCor;//EnCor applied
  Float_t shwEnLinNCNoCor;//shw.shwph.linNCgev
  Float_t shwEnLinNCCor;//EnCor applied
  Float_t shwEnWtNCNoCor;//shw.shwph.wtNCgev
  Float_t shwEnWtNCCor;//EnCor applied

  Int_t planeShwBeg;//shw.plane.beg
  Int_t planeShwEnd;//shw.plane.end
  Float_t xShwVtx;//shw.vtx.x
  Float_t yShwVtx;//shw.vtx.y
  Float_t zShwVtx;//shw.vtx.z


  //////////////////////////////////////////////////
  //standard ntuple variables for first trk/shw
  Bool_t trkExists1;//simply state if track exists
  Int_t trkIndex1;//trk.index, position in TClonesArray in sntp file
  Int_t ndigitTrk1;//trk.ndigit
  Int_t nstripTrk1;//trk.nstrip
  Float_t trkEnCorRange1;//trk energy from range (EnCor applied)
  Float_t trkEnCorCurv1;//trk energy from curvature (EnCor applied)
  Float_t trkShwEnNear1; // Shower energy 'near' the track vertex
  Float_t trkMomentumRange1;//trk momentum from range (no EnCor)
  Float_t trkMomentumTransverse1; // trkmom*sin(acos(dirCosNu));
  Int_t containedTrk1;//trk.contained
  Int_t trkfitpass1;//trk.fit.pass
  Float_t trkvtxdcosz1;//trk.vtx.dcosz
  Float_t trkvtxdcosy1;//trk.vtx.dcosy
  Int_t trknplane1;//trk.nplane
  Int_t charge1;//+1 for NuMuBar and -1 for NuMu
  Float_t qp1;//trk Q/P from fit (no EnCor)
  Float_t qp_rangebiased1;//trk Q/P from fit (with range bias)
  Float_t sigqp1;//trk sigmaQ/P from fit
  Float_t qp_sigqp1; // qp / sigqp
  Float_t chi21;//trk chi2 of fit
  Float_t ndof1;//trk ndof of fit
  Float_t qpFraction1;//trk.stpfitqp[i], npositive/nstrip
  Int_t trkVtxUVDiffPl1;//trk.plane.begu-trk.plane.begv;
  Int_t trkLength1;//abs(trk.plane.end-trk.plane.beg+1);
  Int_t planeTrkNu1;//trk.plane.nu: number of u planes hit
  Int_t planeTrkNv1;//trk.plane.nv: number of v planes hit
  Int_t ntrklike1;//trk.plane.ntrklike: number of trk-like planes
  Float_t trkphsigcor1;//trk.ph.sigcor
  Float_t trkphsigmap1;//trk.ph.sigmap

  Int_t trkfitpassSA1;//variables from the SA track fitter
  Float_t trkvtxdcoszSA1;//fitsa.fit.dcosz 
  Int_t chargeSA1;//definitions same as variables without SA postfix
  Float_t qpSA1;
  Float_t sigqpSA1;
  Float_t chi2SA1;
  Float_t ndofSA1;
  Float_t probSA1;
  Float_t xTrkVtxSA1;//calculated from u&v
  Float_t yTrkVtxSA1;//calculated from u&v
  Float_t zTrkVtxSA1;//fitsa.fit.z
  Float_t uTrkVtxSA1;//fitsa.fit.u
  Float_t vTrkVtxSA1;//fitsa.fit.v

  Float_t jitter1;//from MajCInfo
  Float_t jPID1;//from MajCInfo
  Float_t majC1;//from MajCInfo
  Float_t smoothMajC1;//from MajCInfo

  Float_t roID1;//RO's PID variable (got using the evt)
  Float_t knn01TrkActivePlanes1;//number of active planes in trk
  Float_t knn10TrkMeanPh1;//average ph per plane in trk
  Float_t knn20LowHighPh1;//av of low ph strips/av of high ph strips
  Float_t knn40TrkPhFrac1;//fraction of ph in trk
  Float_t roIDNuMuBar1;//RO's PID NuMuBar selection (0 or 1)
  Float_t relativeAngle1;//RO's track angle relative to muon dir.

  //---------------------------- 
  //  Jasmine new variables (track1)
  //----------------------------       
  Float_t JMID1;
  Float_t JMntrkpl1;
  Float_t JMendph1;
  Float_t JMmeanph1;
  Float_t JMscatu1;
  Float_t JMscatv1;
  Float_t JMscatuv1;
  Float_t JMtrkqp1;

  Float_t xTrkVtx1;//trk.vtx.x
  Float_t yTrkVtx1;//trk.vtx.y
  Float_t zTrkVtx1;//trk.vtx.z
  Float_t uTrkVtx1;//trk.vtx.u
  Float_t vTrkVtx1;//trk.vtx.v
  Int_t planeTrkVtx1;//trk.vtx.plane
  Int_t planeTrkBeg1;//trk.plane.beg
  Int_t planeTrkBegu1;//trk.plane.begu
  Int_t planeTrkBegv1;//trk.plane.begv
  // Strip geometry information
  Int_t stripTrkBeg1;  // First strip hit
  Int_t stripTrkBegu1; // First strip hit in a u plane
  Int_t stripTrkBegv1; // First strip hit in a v plane
  Bool_t stripTrkBegIsu1; // True if the first strip hit in this track was in a u plane
  Int_t regionTrkVtx1; // What region? See enum Rock_DetectorRegion
  Float_t phiTrkVtx1; // Location of the track vertex in Phi = atan(vtx.x / vtx.y)
  
  Float_t xTrkEnd1;//trk.end.x
  Float_t yTrkEnd1;//trk.end.y
  Float_t zTrkEnd1;//trk.end.z
  Float_t uTrkEnd1;//trk.end.u
  Float_t vTrkEnd1;//trk.end.v
  Int_t planeTrkEnd1;//trk.plane.end
  Int_t planeTrkEndu1;//trk.plane.endu
  Int_t planeTrkEndv1;//trk.plane.endv

  Float_t drTrkFidall1;//trk.fidall.dr
  Float_t dzTrkFidall1;//trk.fidall.dz
  Float_t drTrkFidvtx1;//trk.fidbeg.dr
  Float_t dzTrkFidvtx1;//trk.fidbeg.dz
  Float_t drTrkFidend1;//trk.fidend.dr
  Float_t dzTrkFidend1;//trk.fidend.dz
  Float_t traceTrkFidall1; //trk.fidall.trace
  Float_t traceTrkFidvtx1; //trk.fidvtx.trace
  Float_t traceTrkFidend1; //trk.fidend.trace
  Float_t cosPrTrkVtx1; // Cos of angle between theta and radial momentum
    
  //shw variables
  Bool_t shwExists1;//simply state if shw exists
  Int_t ndigitShw1;//shw.ndigit
  Int_t nstripShw1;//shw.nstrip
  Int_t nplaneShw1;//shw.plane.n
  Float_t shwEnCor1;//shower energy (EnCor applied)
  Float_t shwEnNoCor1;//shower energy (no EnCor)
  Float_t shwEnLinCCNoCor1;//shw.shwph.linCCgev
  Float_t shwEnLinCCCor1;//EnCor applied
  Float_t shwEnWtCCNoCor1;//shw.shwph.wtCCgev
  Float_t shwEnWtCCCor1;//EnCor applied
  Float_t shwEnLinNCNoCor1;//shw.shwph.linNCgev
  Float_t shwEnLinNCCor1;//EnCor applied
  Float_t shwEnWtNCNoCor1;//shw.shwph.wtNCgev
  Float_t shwEnWtNCCor1;//EnCor applied
  Float_t shwEnMip1;//shower energy (no EnCor)
  Int_t planeShwBeg1;//shw.plane.beg
  Int_t planeShwEnd1;//shw.plane.end
  Float_t xShwVtx1;//shw.vtx.x
  Float_t yShwVtx1;//shw.vtx.y
  Float_t zShwVtx1;//shw.vtx.z


  //////////////////////////////////////////////
  //standard ntuple variables for second trk/shw
  Bool_t trkExists2;//simply state if track exists
  Int_t trkIndex2;//trk.index, position in TClonesArray in sntp file
  Int_t ndigitTrk2;//trk.ndigit
  Int_t nstripTrk2;//trk.nstrip
  Float_t trkEnCorRange2;//trk energy from range (EnCor applied)
  Float_t trkEnCorCurv2;//trk energy from curvature (EnCor applied)
  Float_t trkShwEnNear2; // Shower energy 'near' the track vertex
  Float_t trkMomentumRange2;//trk momentum from range (no EnCor)
  Float_t trkMomentumTransverse2; // trkmom*sin(acos(dirCosNu));
  Int_t containedTrk2;//trk.contained
  Int_t trkfitpass2;//trk.fit.pass
  Float_t trkvtxdcosz2;//trk.vtx.dcosz
  Float_t trkvtxdcosy2;//trk.vtx.dcosy
  Int_t trknplane2;//trk.nplane
  Int_t charge2;//+1 for NuMuBar and -1 for NuMu
  Float_t qp2;//trk Q/P from fit (no EnCor)
  Float_t qp_rangebiased2;//trk Q/P from fit (with range bias)
  Float_t sigqp2;//trk sigmaQ/P from fit
  Float_t qp_sigqp2; // qp / sigqp
  Float_t chi22;//trk chi2 of fit
  Float_t ndof2;//trk ndof of fit
  Float_t qpFraction2;//trk.stpfitqp[i], npositive/nstrip
  Int_t trkVtxUVDiffPl2;//trk.plane.begu-trk.plane.begv;
  Int_t trkLength2;//abs(trk.plane.end-trk.plane.beg+1);
  Int_t planeTrkNu2;//trk.plane.nu: number of u planes hit
  Int_t planeTrkNv2;//trk.plane.nv: number of v planes hit
  Int_t ntrklike2;//trk.plane.ntrklike: number of trk-like planes
  Float_t trkphsigcor2;//trk.ph.sigcor
  Float_t trkphsigmap2;//trk.ph.sigmap

  Int_t trkfitpassSA2;//variables from the SA track fitter
  Float_t trkvtxdcoszSA2;//fitsa.fit.dcosz
  Int_t chargeSA2;//definitions same as variables without SA postfix
  Float_t qpSA2;
  Float_t sigqpSA2;
  Float_t chi2SA2;
  Float_t ndofSA2;
  Float_t probSA2;
  Float_t xTrkVtxSA2;//calculated from u&v
  Float_t yTrkVtxSA2;//calculated from u&v
  Float_t zTrkVtxSA2;//fitsa.fit.z
  Float_t uTrkVtxSA2;//fitsa.fit.u
  Float_t vTrkVtxSA2;//fitsa.fit.v

  Float_t jitter2;//from MajCInfo
  Float_t jPID2;//from MajCInfo
  Float_t majC2;//from MajCInfo
  Float_t smoothMajC2;//from MajCInfo

  Float_t roID2;//RO's PID variable (got using the evt)
  Float_t knn01TrkActivePlanes2;//number of active planes in trk
  Float_t knn10TrkMeanPh2;//average ph per plane in trk
  Float_t knn20LowHighPh2;//av of low ph strips/av of high ph strips
  Float_t knn40TrkPhFrac2;//fraction of ph in trk
  Float_t roIDNuMuBar2;//RO's PID NuMuBar selection (0 or 1)
  Float_t relativeAngle2;//RO's track angle relative to muon dir.

  //----------------------------
  //  Jasmine new variables (track2)
  //----------------------------          
  Float_t JMID2;
  Float_t JMntrkpl2;
  Float_t JMendph2;
  Float_t JMmeanph2;
  Float_t JMscatu2;
  Float_t JMscatv2;
  Float_t JMscatuv2;
  Float_t JMtrkqp2;

  Float_t xTrkVtx2;//trk.vtx.x
  Float_t yTrkVtx2;//trk.vtx.y
  Float_t zTrkVtx2;//trk.vtx.z
  Float_t uTrkVtx2;//trk.vtx.u
  Float_t vTrkVtx2;//trk.vtx.v
  Int_t planeTrkVtx2;//trk.vtx.plane
  Int_t planeTrkBeg2;//trk.plane.beg
  Int_t planeTrkBegu2;//trk.plane.begu
  Int_t planeTrkBegv2;//trk.plane.begv
  Int_t stripTrkBeg2;  // First strip hit
  Int_t stripTrkBegu2; // First strip hit in a u plane
  Int_t stripTrkBegv2; // First strip hit in a v plane
  Bool_t stripTrkBegIsu2; // True if the first strip hit in this track was in a u plane
  Int_t regionTrkVtx2; // What region? See enum Rock_DetectorRegion
  Float_t phiTrkVtx2; // Location of the track vertex in Phi = atan(vtx.x / vtx.y)
  
  Float_t xTrkEnd2;//trk.end.x
  Float_t yTrkEnd2;//trk.end.y
  Float_t zTrkEnd2;//trk.end.z
  Float_t uTrkEnd2;//trk.end.u
  Float_t vTrkEnd2;//trk.end.v
  Int_t planeTrkEnd2;//trk.plane.end
  Int_t planeTrkEndu2;//trk.plane.endu
  Int_t planeTrkEndv2;//trk.plane.endv

  Float_t drTrkFidall2;//trk.fidall.dr
  Float_t dzTrkFidall2;//trk.fidall.dz
  Float_t drTrkFidvtx2;//trk.fidbeg.dr
  Float_t dzTrkFidvtx2;//trk.fidbeg.dz
  Float_t drTrkFidend2;//trk.fidend.dr
  Float_t dzTrkFidend2;//trk.fidend.dz
  Float_t traceTrkFidall2; //trk.fidall.trace
  Float_t traceTrkFidvtx2; //trk.fidvtx.trace
  Float_t traceTrkFidend2; //trk.fidend.trace
  Float_t cosPrTrkVtx2; // Cos of angle between theta and radial momentum
  
  //shw variables
  Bool_t shwExists2;//simply state if shw exists
  Int_t ndigitShw2;//shw.ndigit
  Int_t nstripShw2;//shw.nstrip
  Int_t nplaneShw2;//shw.plane.n
  Float_t shwEnCor2;//shower energy (EnCor applied)
  Float_t shwEnNoCor2;//shower energy (no EnCor)
  Float_t shwEnLinCCNoCor2;//shw.shwph.linCCgev
  Float_t shwEnLinCCCor2;//EnCor applied
  Float_t shwEnWtCCNoCor2;//shw.shwph.wtCCgev
  Float_t shwEnWtCCCor2;//EnCor applied
  Float_t shwEnLinNCNoCor2;//shw.shwph.linNCgev
  Float_t shwEnLinNCCor2;//EnCor applied
  Float_t shwEnWtNCNoCor2;//shw.shwph.wtNCgev
  Float_t shwEnWtNCCor2;//EnCor applied
  Float_t shwEnMip2;//shower energy (no EnCor)
  Int_t planeShwBeg2;//shw.plane.beg
  Int_t planeShwEnd2;//shw.plane.end
  Float_t xShwVtx2;//shw.vtx.x
  Float_t yShwVtx2;//shw.vtx.y
  Float_t zShwVtx2;//shw.vtx.z


  //////////////////////////////////////////////
  //standard ntuple variables for third trk/shw
  Bool_t trkExists3;//simply state if track exists
  Int_t trkIndex3;//trk.index, position in TClonesArray in sntp file
  Int_t ndigitTrk3;//trk.ndigit
  Int_t nstripTrk3;//trk.nstrip
  Float_t trkEnCorRange3;//trk energy from range (EnCor applied)
  Float_t trkEnCorCurv3;//trk energy from curvature (EnCor applied)
  Float_t trkShwEnNear3; // Shower energy 'near' the track vertex
  Float_t trkMomentumRange3;//trk momentum from range (no EnCor)
  Float_t trkMomentumTransverse3; // trkmom*sin(acos(dirCosNu));
  Int_t containedTrk3;//trk.contained
  Int_t trkfitpass3;//trk.fit.pass
  Float_t trkvtxdcosz3;//trk.vtx.dcosz
  Float_t trkvtxdcosy3;//trk.vtx.dcosy
  Int_t trknplane3;//trk.nplane
  Int_t charge3;//+1 for NuMuBar and -1 for NuMu
  Float_t qp3;//trk Q/P from fit (no EnCor)
  Float_t qp_rangebiased3;//trk Q/P from fit (with range bias)
  Float_t sigqp3;//trk sigmaQ/P from fit
  Float_t qp_sigqp3; // qp / sigqp
  Float_t chi23;//trk chi2 of fit
  Float_t ndof3;//trk ndof of fit
  Float_t qpFraction3;//trk.stpfitqp[i], npositive/nstrip
  Int_t trkVtxUVDiffPl3;//trk.plane.begu-trk.plane.begv;
  Int_t trkLength3;//abs(trk.plane.end-trk.plane.beg+1);
  Int_t planeTrkNu3;//trk.plane.nu: number of u planes hit
  Int_t planeTrkNv3;//trk.plane.nv: number of v planes hit
  Int_t ntrklike3;//trk.plane.ntrklike: number of trk-like planes
  Float_t trkphsigcor3;//trk.ph.sigcor
  Float_t trkphsigmap3;//trk.ph.sigmap

  Int_t trkfitpassSA3;//variables from the SA track fitter
  Float_t trkvtxdcoszSA3;//fitsa.fit.dcosz 
  Int_t chargeSA3;//definitions same as variables without SA postfix
  Float_t qpSA3;
  Float_t sigqpSA3;
  Float_t chi2SA3;
  Float_t ndofSA3;
  Float_t probSA3;
  Float_t xTrkVtxSA3;//calculated from u&v
  Float_t yTrkVtxSA3;//calculated from u&v
  Float_t zTrkVtxSA3;//fitsa.fit.z
  Float_t uTrkVtxSA3;//fitsa.fit.u
  Float_t vTrkVtxSA3;//fitsa.fit.v

  Float_t jitter3;//from MajCInfo
  Float_t jPID3;//from MajCInfo
  Float_t majC3;//from MajCInfo
  Float_t smoothMajC3;//from MajCInfo

  Float_t roID3;//RO's PID variable (got using the evt)
  Float_t knn01TrkActivePlanes3;//number of active planes in trk
  Float_t knn10TrkMeanPh3;//average ph per plane in trk
  Float_t knn20LowHighPh3;//av of low ph strips/av of high ph strips
  Float_t knn40TrkPhFrac3;//fraction of ph in trk
  Float_t roIDNuMuBar3;//RO's PID NuMuBar selection (0 or 1)
  Float_t relativeAngle3;//RO's track angle relative to muon dir.

  //----------------------------
  //  Jasmine new variables (track3)
  //----------------------------   
  Float_t JMID3;
  Float_t JMntrkpl3;
  Float_t JMendph3;
  Float_t JMmeanph3;
  Float_t JMscatu3;
  Float_t JMscatv3;
  Float_t JMscatuv3;
  Float_t JMtrkqp3;

  Float_t xTrkVtx3;//trk.vtx.x
  Float_t yTrkVtx3;//trk.vtx.y
  Float_t zTrkVtx3;//trk.vtx.z
  Float_t uTrkVtx3;//trk.vtx.u
  Float_t vTrkVtx3;//trk.vtx.v
  Int_t planeTrkVtx3;//trk.vtx.plane
  Int_t planeTrkBeg3;//trk.plane.beg
  Int_t planeTrkBegu3;//trk.plane.begu
  Int_t planeTrkBegv3;//trk.plane.begv
  // Strip geometry
  Int_t stripTrkBeg3;  // First strip hit
  Int_t stripTrkBegu3; // First strip hit in a u plane
  Int_t stripTrkBegv3; // First strip hit in a v plane
  Bool_t stripTrkBegIsu3; // True if the first strip hit in this track was in a u plane
  Int_t regionTrkVtx3;// What region? See enum Rock_DetectorRegion
  Float_t phiTrkVtx3;//Location of the track vertex in Phi = atan(vtx.x / vtx.y)
  
  Float_t xTrkEnd3;//trk.end.x
  Float_t yTrkEnd3;//trk.end.y
  Float_t zTrkEnd3;//trk.end.z
  Float_t uTrkEnd3;//trk.end.u
  Float_t vTrkEnd3;//trk.end.v
  Int_t planeTrkEnd3;//trk.plane.end
  Int_t planeTrkEndu3;//trk.plane.endu
  Int_t planeTrkEndv3;//trk.plane.endv

  Float_t drTrkFidall3;//trk.fidall.dr
  Float_t dzTrkFidall3;//trk.fidall.dz
  Float_t drTrkFidvtx3;//trk.fidbeg.dr
  Float_t dzTrkFidvtx3;//trk.fidbeg.dz
  Float_t drTrkFidend3;//trk.fidend.dr
  Float_t dzTrkFidend3;//trk.fidend.dz
  Float_t traceTrkFidall3; //trk.fidall.trace
  Float_t traceTrkFidvtx3; //trk.fidvtx.trace
  Float_t traceTrkFidend3; //trk.fidend.trace
  Float_t cosPrTrkVtx3; // Cos of angle between theta and radial momentum
  
  //shw variables
  Bool_t shwExists3;//simply state if shw exists
  Int_t ndigitShw3;//shw.ndigit
  Int_t nstripShw3;//shw.nstrip
  Int_t nplaneShw3;//shw.plane.n
  Float_t shwEnCor3;//shower energy (EnCor applied)
  Float_t shwEnNoCor3;//shower energy (no EnCor)
  Float_t shwEnLinCCNoCor3;//shw.shwph.linCCgev
  Float_t shwEnLinCCCor3;//EnCor applied
  Float_t shwEnWtCCNoCor3;//shw.shwph.wtCCgev
  Float_t shwEnWtCCCor3;//EnCor applied
  Float_t shwEnLinNCNoCor3;//shw.shwph.linNCgev
  Float_t shwEnLinNCCor3;//EnCor applied
  Float_t shwEnWtNCNoCor3;//shw.shwph.wtNCgev
  Float_t shwEnWtNCCor3;//EnCor applied
  Float_t shwEnMip3;//shower energy (no EnCor)
  Int_t planeShwBeg3;//shw.plane.beg
  Int_t planeShwEnd3;//shw.plane.end
  Float_t xShwVtx3;//shw.vtx.x
  Float_t yShwVtx3;//shw.vtx.y
  Float_t zShwVtx3;//shw.vtx.z


  //////////////////////////////////////////////
  //standard ntuple variables for fourth shw
  Bool_t shwExists4;//simply state if shw exists
  Int_t ndigitShw4;//shw.ndigit
  Int_t nstripShw4;//shw.nstrip
  Int_t nplaneShw4;//shw.plane.n
  Float_t shwEnCor4;//shower energy (EnCor applied)
  Float_t shwEnNoCor4;//shower energy (no EnCor)
  Float_t shwEnLinCCNoCor4;//shw.shwph.linCCgev
  Float_t shwEnLinCCCor4;//EnCor applied
  Float_t shwEnWtCCNoCor4;//shw.shwph.wtCCgev
  Float_t shwEnWtCCCor4;//EnCor applied
  Float_t shwEnLinNCNoCor4;//shw.shwph.linNCgev
  Float_t shwEnLinNCCor4;//EnCor applied
  Float_t shwEnWtNCNoCor4;//shw.shwph.wtNCgev
  Float_t shwEnWtNCCor4;//EnCor applied
  Float_t shwEnMip4;//shower energy (no EnCor)
  Int_t planeShwBeg4;//shw.plane.beg
  Int_t planeShwEnd4;//shw.plane.end
  Float_t xShwVtx4;//shw.vtx.x
  Float_t yShwVtx4;//shw.vtx.y
  Float_t zShwVtx4;//shw.vtx.z


  //////////////////////////////////////////////
  //standard ntuple variables for fifth shw
  Bool_t shwExists5;//simply state if shw exists
  Int_t ndigitShw5;//shw.ndigit
  Int_t nstripShw5;//shw.nstrip
  Int_t nplaneShw5;//shw.plane.n
  Float_t shwEnCor5;//shower energy (EnCor applied)
  Float_t shwEnNoCor5;//shower energy (no EnCor)
  Float_t shwEnLinCCNoCor5;//shw.shwph.linCCgev
  Float_t shwEnLinCCCor5;//EnCor applied
  Float_t shwEnWtCCNoCor5;//shw.shwph.wtCCgev
  Float_t shwEnWtCCCor5;//EnCor applied
  Float_t shwEnLinNCNoCor5;//shw.shwph.linNCgev
  Float_t shwEnLinNCCor5;//EnCor applied
  Float_t shwEnWtNCNoCor5;//shw.shwph.wtNCgev
  Float_t shwEnWtNCCor5;//EnCor applied
  Float_t shwEnMip5;//shower energy (no EnCor)
  Int_t planeShwBeg5;//shw.plane.beg
  Int_t planeShwEnd5;//shw.plane.end
  Float_t xShwVtx5;//shw.vtx.x
  Float_t yShwVtx5;//shw.vtx.y
  Float_t zShwVtx5;//shw.vtx.z


  ////////////////////////
  //other info calculated
  Float_t rEvtVtx;//radius of evt vtx
  Float_t rEvtEnd;//radius of evt end
  Float_t distToEdgeEvtVtx;//evt vtx distance to edge of scint
  Int_t evtVtxUVDiffPl;//difference in UV of evt vtx in planes

  Float_t rTrkVtx;//radius of trk vtx
  Float_t rTrkEnd;//radius of trk end
  Float_t sigqp_qp;//best trk sigmaQ/P / Q/P from fit
  Float_t chi2PerNdof;//best-trk chi2/ndof of fit
  Float_t prob;//best trk probability of fit  

  Int_t containmentFlag;//containment flag to use
  Int_t containmentFlagCC0093Std;//CC 0.9e20 pot containment flag
  Int_t containmentFlagCC0250Std;//CC 2.5e20 pot containment flag
  Int_t containmentFlagPitt;//pitt specific flag: 1-4, up/down stop/exit
  Int_t usedRange;//1=used range, 0=not
  Int_t usedCurv;//1=used curvature, 0=not
  

  /////////
  //weights
  /////////
  Float_t rw;//the weight to use as default 
  Float_t fluxErr;//the error on the flux to use as default
  Float_t rwActual;//beam reweight factor actually used
  Float_t generatorWeight;//weight factor from generator
  Float_t detectorWeight;//SKZP detector weight to use as default

  //Daikon04 and earlier: average POT weighted weights, e.g. (1.27*RunI + 1.23*RunII)/2.50
  //Daikon07 and later: Just the correct weights for this MC event
  Float_t trkEnWeight;//SKZP weight applied to trkEn (number close to 1)
  Float_t shwEnWeight;//SKZP weight applied to shwEn (number close to 1)
  Float_t beamWeight;//SKZP weight for beam (e.g. hadron prod.)
  Float_t fluxErrHadProdAfterTune;//flux error, kHadProdAfterTune
  Float_t fluxErrTotalErrorPreTune;//flux error, kTotalErrorPreTune
  Float_t fluxErrTotalErrorAfterTune;//flux error kTotalErrorAfterTune
  Float_t detectorWeightNMB;//SKZP detector weight (e.g. xsec)
  Float_t detectorWeightNM;//NuMu rather than NuMuBar above
  
  //Daikon04 and earlier: weights for RunI (2005-2006), e.g. LE-10
  //Daikon07 and later: Only set for RunI events
  Float_t trkEnWeightRunI;//as above
  Float_t shwEnWeightRunI;//as above
  Float_t beamWeightRunI;//as above
  Float_t fluxErrHadProdAfterTuneRunI;//as above
  Float_t fluxErrTotalErrorPreTuneRunI;//as above
  Float_t fluxErrTotalErrorAfterTuneRunI;//as above
  Float_t detectorWeightNMBRunI;//as above
  Float_t detectorWeightNMRunI;//as above
  
  //Daikon04 and earlier: weights for RunII (2006-2007), e.g. LE-09
  //Daikon07 and later: Only set for RunII events
  Float_t trkEnWeightRunII;//as above
  Float_t shwEnWeightRunII;//as above
  Float_t beamWeightRunII;//as above
  Float_t fluxErrHadProdAfterTuneRunII;//as above
  Float_t fluxErrTotalErrorPreTuneRunII;//as above
  Float_t fluxErrTotalErrorAfterTuneRunII;//as above
  Float_t detectorWeightNMBRunII;//as above
  Float_t detectorWeightNMRunII;//as above

  //energies with and without weights
  Float_t energyRw;//energy after beam reweighting
  Float_t energyNoRw;//plain reco'd energy before reweighting
  Float_t trkEnRw;//trk energy after beam reweighting
  Float_t trkEnNoRw;//plain reco'd trk energy before reweighting
  Float_t shwEnRw;//shw energy after beam reweighting
  Float_t shwEnNoRw;//plain reco'd shw energy before reweighting
  
  //pids
  Float_t dpID;//DP's PID variable
  Float_t abID;//AB's PID variable
  Float_t roID;//RO's PID variable
  Float_t knn01TrkActivePlanes;//number of active planes in trk
  Float_t knn10TrkMeanPh;//average ph per plane in trk
  Float_t knn20LowHighPh;//av of low ph strips/av of high ph strips
  Float_t knn40TrkPhFrac;//fraction of ph in trk
  Float_t roIDNuMuBar;//RO's PID variable for NuMuBar selection (0 or 1)
  Float_t relativeAngle;//RO's track angle relative to muon dir.
  Float_t poID;//PO's PID variable
  Float_t poIDKin;//PO's PID variable (kinematics only)

  //-------------------------------------
  //Jasmine variables
  //-------------------------------------
  Float_t JMID;
  Float_t JMntrkpl;
  Float_t JMendph;
  Float_t JMmeanph;
  Float_t JMscatu;
  Float_t JMscatv;
  Float_t JMscatuv;
  Float_t JMtrkqp;
  Float_t JMeknnID;
  Float_t JMeknn208;
  Float_t JMeknn205;
  Float_t JMeknn204;


  ////////////////////////////
  // NC variables
  // Comments from NC codes
  ////////////////////////////
  // event
  Float_t closeTimeDeltaZ; //distance in z to event closest in time
  Int_t edgeActivityStrips;//number of strips in partially instrumented region in 40ns time window 
  Float_t edgeActivityPH;//summed PH in partially instrumented region in 40ns time window 
  Int_t oppEdgeStrips;//number of strips in opposite edge region in 40ns time window
  Float_t oppEdgePH;//summed PH in opposite edge region in 40ns time window
  Float_t vtxMetersToCoilEvt;//distance to the coil hole
  Float_t vtxMetersToCloseEdgeEvt;//distance to nearest edge (in XY)of partial plane outline in ND
  Double_t minTimeSeparation;//time difference to closest event in time 

  // shower
  Float_t transverseRMSU;//rms of transverse strip positions in U
  Float_t transverseRMSV;//rms of transverse strip positions in V

  // track
  Float_t dtdz;//gradient of t(z) fit (with c_light=1)
  Float_t endMetersToCloseEdge; //distance to nearest edge (in XY) of full plane outline in ND
  Float_t vtxMetersToCloseEdgeTrk;//distance to nearest edge (in XY) of partial plane outline in ND
  Float_t vtxMetersToCoilTrk;//distance to the center of the coil hole
  Float_t traceEndZ;//delta in z from end to projected exit location

  //beam variables
  Float_t pot;//pot in current spill
  Float_t potDB;//pot in current spill from database
  Float_t potSinceLastEvt;//includes pot count for spills w/ no evts
  Float_t potSinceLastEvtGood;//as above but also good beam+det spills
  Float_t potSinceLastEvtBad;//as above but for bad beam+det spills
  Float_t potSinceLastEvtDB;//as above but with database version of pots
  Float_t potSinceLastEvtGoodDB;//as above but with database version of pots
  Float_t potSinceLastEvtBadDB;//as above but with database version of pots

  Int_t runPeriod;
  Bool_t hornIsReverse;  
  //Conventions/BeamType.h definition
  Int_t beamTypeDB;// From BeamMonSpill
  Int_t beamType; // From NuConfig
  Float_t intensity;//Only filled for MC events
  Float_t hornCur;//BeamMonSpill::fHornCur
  Bool_t goodBeam;//BMSpillAna.SelectSpill()
  Bool_t goodBeamSntp;//uses the beam data in the sntp file
  
  
  //////////////////
  //truth variables
  //////////////////

  //////////////////////////////////////////////////////////////////
  //EVERYTIME A TRUTH VARIABLE IS ADDED TO THIS CLASS IT MUST
  //ALSO BE ADDED TO NuMCEvent
  ///////////////////////////////////////////////////////////////////

  Float_t energyMC;//what could be truely reco'd: neuEn*y for NC
  
  Float_t neuEnMC;//p4neu[3];
  Float_t neuPxMC;//p4neu[0];
  Float_t neuPyMC;//p4neu[1];  
  Float_t neuPzMC;//p4neu[2];

  Float_t mu1EnMC;//p4mu1[3];
  Float_t mu1PxMC;//p4mu1[0];
  Float_t mu1PyMC;//p4mu1[1];  
  Float_t mu1PzMC;//p4mu1[2];

  Float_t tgtEnMC;//p4tgt[3]
  Float_t tgtPxMC;//p4tgt[0];
  Float_t tgtPyMC;//p4tgt[1];
  Float_t tgtPzMC;//p4tgt[2];

  Int_t zMC;//z;
  Int_t aMC;//a;
  Int_t nucleusMC;//encoding from Mad of the nucleus type
  Int_t initialStateMC;//encoding from Mad of the initial state
  Int_t hadronicFinalStateMC;//encoding from Mad of the hfs

  Float_t yMC;//y value from sntp file
  Float_t y2MC;//p4shw[3]/(fabs(p4mu1[3])+p4shw[3])
  Float_t xMC;//x;
  Float_t q2MC;//q2;
  Float_t w2MC;//w2;

  Float_t trkEnMC;//p4mu1[3], not proper p4: muon energy (+/- !!!)
  Float_t trkEn2MC;//(1-y)*p4neu[3];
  Float_t shwEnMC;//p4shw[3]
  Float_t shwEn2MC;//y*p4neu[3];
  
  Float_t trkEndEnMC;//NtpMCStdHep.dethit[1].pE, particle En at last scint hit
  Float_t trkStartEnMC; // muon energy at first scintillator hit
  Bool_t  trkContainmentMC;// Experimental: MC Containment of primary muon
  
  Float_t sigma;//mc.sigma=cross-section
  Int_t iaction;//CC=1, NC=0
  Int_t iresonance;//QE=1001, RES=1002, DIS=1003, CPP=1004
  Int_t inu;//>0 particles, <0 anti-particles
  Int_t inunoosc;//id of neutrino at birth
  Int_t itg;//neutrino interaction target

  Float_t vtxxMC;//vtxx: x vtx of neutrino interaction
  Float_t vtxyMC;//vtxy: y vtx of neutrino interaction
  Float_t vtxzMC;//vtxz: z vtx of neutrino interaction
  Float_t vtxuMC;//vtx u, calculated from x&y
  Float_t vtxvMC;//vtx v, calculated from x&y
  Int_t planeTrkVtxMC;//calculated from z
  Float_t rTrkVtxMC;//calculated from x&y

  Int_t mc;//the index of the object to be used
  Int_t mcTrk;//track mc index
  Int_t mcShw;//shower mc index
  Int_t mcEvt;//event mc index

  Int_t mcTrk1;//1st track mc index
  Int_t mcTrk2;//2nd track mc index
  Int_t mcTrk3;//3rd track mc index

  Int_t mcShw1;//1st shower mc index
  Int_t mcShw2;//2nd shower mc index
  Int_t mcShw3;//3rd shower mc index
  Int_t mcShw4;//4th shower mc index
  Int_t mcShw5;//5th shower mc index
  
  //http://www.hep.utexas.edu/~zarko/wwwgnumi/v19/v19/output_gnumi.html
  Float_t Npz;
  Float_t NdxdzNea;
  Float_t NdydzNea;
  Float_t NenergyN;
  Float_t NWtNear;
  Float_t NdxdzFar;
  Float_t NdydzFar;
  Float_t NenergyF;
  Float_t NWtFar;
  Int_t Ndecay;
  Float_t Vx;
  Float_t Vy;
  Float_t Vz;
  Float_t pdPx;
  Float_t pdPy;
  Float_t pdPz;
  Float_t ppdxdz;
  Float_t ppdydz;
  Float_t pppz;
  Float_t ppenergy;
  Float_t ppmedium;
  Float_t ppvx;//mc.flux.ppvx = parent production vtx x, e.g. decay pipe
  Float_t ppvy;//mc.flux.ppvy
  Float_t ppvz;//mc.flux.ppvz
  Int_t ptype;//mc.flux.ptype = pion, kaon, muon, etc
  Float_t Necm;
  Float_t Nimpwt;
  Float_t tvx;
  Float_t tvy;
  Float_t tvz;
  Float_t tpx;//mc.flux.tpx = grand^n parent target momentum x (Fluka)
  Float_t tpy;//mc.flux.tpy
  Float_t tpz;//mc.flux.tpz
  Int_t tptype;//mc.flux.tptype = pion, kaon, muon, etc
  Int_t tgen;//mc.flux.tgen
                                                               
  //----------------------------
  //  Truth  Intranuke weighting (Jasmine ma)
  //---------------------------- 
  Int_t   InukeNwts;
  Float_t InukePiCExchgP;   //0
  Float_t InukePiCExchgN;   //1
  Float_t InukePiEScatP;    //2
  Float_t InukePiEScatN;    //3
  Float_t InukePiInEScatP;  //4
  Float_t InukePiInEScatN;  //5
  Float_t InukePiAbsorbP;   //6
  Float_t InukePiAbsorbN;   //7
  Float_t InukePi2PiP;      //8
  Float_t InukePi2PiN;      //9
  Float_t InukeNknockP;     //10
  Float_t InukeNknockN;     //11
  Float_t InukeNNPiP;       //12
  Float_t InukeNNPiN;       //13
  Float_t InukeFormTP;      //14
  Float_t InukeFormTN;      //15
  Float_t InukePiXsecP;     //16
  Float_t InukePiXsecN;     //17
  Float_t InukeNXsecP;      //18
  Float_t InukeNXsecN;      //19
  Float_t InukeNucrad;     
  Float_t InukeWrad;
  //-----------------------------   

  //////////////////////////////////////////////////////////////////
  //EVERYTIME A TRUTH VARIABLE IS ADDED TO THIS CLASS IT MUST
  //ALSO BE ADDED TO NuMCEvent
  ///////////////////////////////////////////////////////////////////

  ////////////////////////////
  //program control variables
  ////////////////////////////
  Int_t anaVersion;//different cuts etc
  Int_t releaseType;//Conventions/ReleaseType.h definition
  Int_t recoVersion;//Birch/Cedar
  Int_t mcVersion;//Carrot/Daikon, used to select pdfs in data case
  Int_t reweightVersion;//the beam reweighting to use

  Bool_t useGeneratorReweight;//switch to turn off generator reweighting
  std::string sGeneratorConfigName;//name of generator rw configuration
  Int_t generatorConfigNo;//number of generator rw configuration

  Bool_t useDBForDataQuality;//flag to use DB for data quality
  Bool_t useDBForSpillTiming;//flag to use DB for spill timing
  Bool_t useDBForBeamInfo;//flag to use DB for beam info
  
  Bool_t cutOnDataQuality;//flag to enable cut on data quality
  Bool_t cutOnSpillTiming;//flag to enable cut on spill timing
  Bool_t cutOnBeamInfo;//flag to enable cut on beam info
  
  Bool_t applyEnergyShifts;//flag to use energy shifts
  Bool_t applyBeamWeight;//flag to use beam weight
  Bool_t apply1SigmaWeight;//flag to use +1-sigma SKZP error shift 
  Bool_t applyDetectorWeight;//flag to use detector weight, e.g. xsec
  Bool_t applyGeneratorWeight;//flag to use generator weight
  
  Bool_t calcMajCurv;//flag to run majorityCurv or not
  Bool_t calcRoID;//flag to run RoID or not
  Bool_t calcJmID;// flag to run JmID or not

  ClassDef(NuEvent,60);
};

#endif //NUEVENT_H

