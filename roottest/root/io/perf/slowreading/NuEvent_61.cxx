#include "Rtypes.h"
#include "NuEvent_61.h"

//......................................................................

NuEvent::NuEvent()
{
  this->Reset();
}

//......................................................................

void NuEvent::Reset()
{
  //////////////////////////////////////////////////////////////////
  /// EVERYTIME A TRUTH VARIABLE IS ADDED TO THIS CLASS IT MUST
  /// ALSO BE ADDED TO NuMCEvent
  ///////////////////////////////////////////////////////////////////

  ///See header file for a description of the variables

  ///////////////////////////
  //book keeping  quantities
  ///////////////////////////
  index=-1;
  entry=-1;

  /////////////////////////////
  //snarl/run based quantities
  /////////////////////////////
  run=-1;
  subRun=-1;
  runType=-1;
  errorCode=-1;
  snarl=-1;
  trigSrc=-1;
  timeFrame=-1;
  remoteSpillType=-1;

  detector=-1;
  simFlag=-1;
  timeSec=-1;
  timeNanoSec=-1;
  timeSeconds=-1;  

  trigtime=-1;
  medianTime=-1;
  timeEvtMin=-999999;
  timeEvtMax=-999999;

  nearestSpillSec=-1;
  nearestSpillNanosec=0;
  timeToNearestSpill=-999999;

  planeEvtHdrBeg=-1;
  planeEvtHdrEnd=-1;
  snarlPulseHeight=-1;

  //////////////////////////
  // data quality variables
  //////////////////////////
  isGoodDataQuality=false;
  isGoodDataQualityRUN=false;
  isGoodDataQualityCOIL=false;
  isGoodDataQualityHV=false;
  isGoodDataQualityGPS=false;

  numActiveCrates=0;
  numTimeFrames=0;
  numGoodSnarls=0;
  snarlRateMedian=0.0;
  snarlRateMax=0.0;

  coilIsOk=true;//false is safer, but old ntuples don't have it
  coilIsReverse=false;//assume forward (unlikely to happen upon reverse)
  coilCurrent=0;//will never want 0 most likely

  deltaSecToSpillGPS=-999999;
  deltaNanoSecToSpillGPS=0;
  gpsError=-1;
  gpsSpillType=-1;

  isLI=false;
  litag=0;
  litime=-1;
    
  ///////////////////////////
  //reconstructed quantities
  ///////////////////////////
  energy=-1;
  energyCC=-1;
  energyNC=-1;
  energyRM=-1;
  trkEn=-1;
  trkEnRange=-1;
  trkEnCurv=-1;
  shwEn=-1;
  shwEnCC=-1;
  shwEnNC=-1;

  y=-1;
  q2=-1;
  x=-1;
  w2=-1;
  dirCosNu=-999;

  //fvnmb=false;//DEPRECATED
  //fvpitt=false;//DEPRECATED
  //fvcc=false;//DEPRECATED

  ////////////////////////////////////////
  //event info extracted from the ntuples
  evt=-1;
  slc=-1;  
  nevt=-1;
  ndigitEvt=-1;
  nstripEvt=-1;
  nshw=-1;
  ntrk=-1;
  primshw=-1;
  primtrk=-1;
  rawPhEvt=-1;
  evtphsigcor=-1;
  evtphsigmap=-1;
  planeEvtN=-1;
  planeEvtNu=-1;
  planeEvtNv=-1;

  roIDEvt=-999;
  knn01TrkActivePlanesEvt=-999;
  knn10TrkMeanPhEvt=-999;
  knn20LowHighPhEvt=-999;
  knn40TrkPhFracEvt=-999;
  roIDNuMuBarEvt=-999;
  relativeAngleEvt=-999;
  // jasmine's test variables
  jmIDEvt=-1;
  jmTrackPlaneEvt= -999;
  jmMeanPhEvt = -999;
  jmEndPhEvt = -999;
  jmScatteringUEvt =-999;
  jmScatteringVEvt =-999;
  jmScatteringUVEvt =-999;
  jmEventknnIDEvt = -999;
  jmEventknn208Evt =-999;
  jmEventknn207Evt =-999;
  jmEventknn206Evt =-999;
  jmEventknn205Evt =-999;
  jmEventknn204Evt =-999;


  xEvtVtx=-999;
  yEvtVtx=-999;
  zEvtVtx=-999;
  uEvtVtx=-999;
  vEvtVtx=-999;
  planeEvtVtx=-1;
  planeEvtBeg=-1;
  planeEvtBegu=-1;
  planeEvtBegv=-1;
  
  xEvtEnd=-999;
  yEvtEnd=-999;
  zEvtEnd=-999;
  uEvtEnd=-999;
  vEvtEnd=-999;
  planeEvtEnd=-1;
  planeEvtEndu=-1;
  planeEvtEndv=-1;
  
  /////////////////////////////////////////////////////////
  //these are the variables of the "best" track and shower
  trkExists=false;
  trkIndex=-1;
  ndigitTrk=-1;
  nstripTrk=-1;
  trkEnCorRange=-1;
  trkEnCorCurv=-1;
  trkShwEnNear=-1;
  trkMomentumRange=-1;
  trkMomentumTransverse=-1;
  containedTrk=0;
  trkfitpass=-1;
  trkvtxdcosz=-999;
  trkvtxdcosy=-999;
  trknplane=-999;
  charge=0;
  qp=-999;
  qp_rangebiased=-999;
  sigqp=-1;
  qp_sigqp=-999;
  chi2=-1;
  ndof=0;
  qpFraction=-1;
  trkVtxUVDiffPl=-999;
  trkLength=-1;
  planeTrkNu=-1;
  planeTrkNv=-1;
  ntrklike=-1;
  trkphsigcor=-1;
  trkphsigmap=-1;

  trkfitpassSA=-1;
  trkvtxdcoszSA=-999;
  chargeSA=0;
  qpSA=-999;
  sigqpSA=-1;
  chi2SA=-1;
  ndofSA=0;
  probSA=-1;
  xTrkVtxSA=-999;
  yTrkVtxSA=-999;
  zTrkVtxSA=-999;
  uTrkVtxSA=-999;
  vTrkVtxSA=-999;

  jitter=-1;
  jPID=-999;
  majC=0;
  //majCRatio=-999;
  //rms=-1;
  //simpleMajC=-999;
  smoothMajC=-999;
  //sqJitter=-1;
  //totWidth=-999;
  
  xTrkVtx=-999;
  yTrkVtx=-999;
  zTrkVtx=-999;
  uTrkVtx=-999;
  vTrkVtx=-999;
  planeTrkVtx=-1;
  planeTrkBeg=-1;
  planeTrkBegu=-1;
  planeTrkBegv=-1;
  stripTrkBeg=-1;
  stripTrkBegu=-1;
  stripTrkBegv=-1;
  stripTrkBegIsu=false;
  regionTrkVtx=-1;
  phiTrkVtx=-999;
  
  xTrkEnd=-999;
  yTrkEnd=-999;
  zTrkEnd=-999;
  uTrkEnd=-999;
  vTrkEnd=-999;
  planeTrkEnd=-1;
  planeTrkEndu=-1;
  planeTrkEndv=-1;

  drTrkFidall=-999;
  dzTrkFidall=-999;
  drTrkFidvtx=-999;
  dzTrkFidvtx=-999;
  drTrkFidend=-999;
  dzTrkFidend=-999;
  traceTrkFidall = -999;
  traceTrkFidvtx = -999;
  traceTrkFidend = -999;
  
  cosPrTrkVtx = -999;
  
  //shw variables
  shwExists=false;
  ndigitShw=-1;
  nstripShw=-1;
  nplaneShw=-1;
  shwEnCor=-1;
  shwEnNoCor=-1;
  shwEnMip=-1;
  shwEnLinCCNoCor=-1;
  shwEnLinCCCor=-1;
  shwEnWtCCNoCor=-1;
  shwEnWtCCCor=-1;
  shwEnLinNCNoCor=-1;
  shwEnLinNCCor=-1;
  shwEnWtNCNoCor=-1;
  shwEnWtNCCor=-1;

  planeShwBeg=-1;
  planeShwEnd=-1;
  xShwVtx=-999;
  yShwVtx=-999;
  zShwVtx=-999;


  ///////////////////////////////////////////////
  //standard ntuple variables for primary trk/shw
  trkExists1=false;
  trkIndex1=-1;
  ndigitTrk1=-1;
  nstripTrk1=-1;
  trkEnCorRange1=-1;
  trkEnCorCurv1=-1;
  trkShwEnNear1=-1;
  trkMomentumRange1=-1;
  trkMomentumTransverse1=-1;
  containedTrk1=0;
  trkfitpass1=-1;
  trkvtxdcosz1=-999;
  trkvtxdcosy1=-999;
  trknplane1=-999;
  charge1=0;
  qp1=-999;
  qp_rangebiased1=-999;
  sigqp1=-1;
  qp_sigqp1=-999;
  chi21=-1;
  ndof1=0;
  qpFraction1=-1;
  trkVtxUVDiffPl1=-999;
  trkLength1=-1;
  planeTrkNu1=-1;
  planeTrkNv1=-1;
  ntrklike1=-1;
  trkphsigcor1=-1;
  trkphsigmap1=-1;

  trkfitpassSA1=-1;
  trkvtxdcoszSA1=-999;
  chargeSA1=0;
  qpSA1=-999;
  sigqpSA1=-1;
  chi2SA1=-1;
  ndofSA1=0;
  probSA1=-1;
  xTrkVtxSA1=-999;
  yTrkVtxSA1=-999;
  zTrkVtxSA1=-999;
  uTrkVtxSA1=-999;
  vTrkVtxSA1=-999;

  jitter1=-1;
  jPID1=-999;
  majC1=0;
  //majCRatio1=-999;
  //rms1=-1;
  //simpleMajC1=-999;
  smoothMajC1=-999;
  //sqJitter1=-1;
  //totWidth1=-999;

  roID1=-999;
  knn01TrkActivePlanes1=-999;
  knn10TrkMeanPh1=-999;
  knn20LowHighPh1=-999;
  knn40TrkPhFrac1=-999;
  roIDNuMuBar1=-999;
  relativeAngle1=-999;

  // jasmine's test variables 
  jmID1 =-1;
  jmTrackPlane1 =-999;
  jmMeanPh1 = -999;
  jmEndPh1 = -999;
  jmScatteringU1 =-999;
  jmScatteringV1 =-999;
  jmScatteringUV1 =-999;

  xTrkVtx1=-999;
  yTrkVtx1=-999;
  zTrkVtx1=-999;
  uTrkVtx1=-999;
  vTrkVtx1=-999;
  planeTrkVtx1=-1;
  planeTrkBeg1=-1;
  planeTrkBegu1=-1;
  planeTrkBegv1=-1;
  stripTrkBeg1=-1;
  stripTrkBegu1=-1;
  stripTrkBegv1=-1;
  stripTrkBegIsu1=false;
  regionTrkVtx1=-1;
  phiTrkVtx1=-999;
  
  xTrkEnd1=-999;
  yTrkEnd1=-999;
  zTrkEnd1=-999;
  uTrkEnd1=-999;
  vTrkEnd1=-999;
  planeTrkEnd1=-1;
  planeTrkEndu1=-1;
  planeTrkEndv1=-1;

  drTrkFidall1=-999;
  dzTrkFidall1=-999;
  drTrkFidvtx1=-999;
  dzTrkFidvtx1=-999;
  drTrkFidend1=-999;
  dzTrkFidend1=-999;
  traceTrkFidall1 = -999;
  traceTrkFidvtx1 = -999;
  traceTrkFidend1 = -999;
  
  cosPrTrkVtx1 = -999;
  
  //shw variables
  shwExists1=false;
  ndigitShw1=-1;
  nstripShw1=-1;
  nplaneShw1=-1;
  shwEnCor1=-1;
  shwEnNoCor1=-1;
  shwEnLinCCNoCor1=-1;
  shwEnLinCCCor1=-1;
  shwEnWtCCNoCor1=-1;
  shwEnWtCCCor1=-1;
  shwEnLinNCNoCor1=-1;
  shwEnLinNCCor1=-1;
  shwEnWtNCNoCor1=-1;
  shwEnWtNCCor1=-1;
  shwEnMip1=-1;
  planeShwBeg1=-1;
  planeShwEnd1=-1;
  xShwVtx1=-999;
  yShwVtx1=-999;
  zShwVtx1=-999;


  //////////////////////////////////////////////
  //standard ntuple variables for second trk/shw
  trkExists2=false;
  trkIndex2=-1;
  ndigitTrk2=-1;
  nstripTrk2=-1;
  trkEnCorRange2=-1;
  trkEnCorCurv2=-1;
  trkShwEnNear2=-1;
  trkMomentumRange2=-1;
  trkMomentumTransverse2=-1;
  containedTrk2=0;
  trkfitpass2=-1;
  trkvtxdcosz2=-999;
  trkvtxdcosy2=-999;
  trknplane2=-999;
  charge2=0;
  qp2=-999;
  qp_rangebiased2=-999;
  sigqp2=-1;
  qp_sigqp2=-999;
  chi22=-1;
  ndof2=0;
  qpFraction2=-1;
  trkVtxUVDiffPl2=-999;
  trkLength2=-1;
  planeTrkNu2=-1;
  planeTrkNv2=-1;
  ntrklike2=-1;
  trkphsigcor2=-1;
  trkphsigmap2=-1;

  trkfitpassSA2=-1;
  trkvtxdcoszSA2=-999;
  chargeSA2=0;
  qpSA2=-999;
  sigqpSA2=-1;
  chi2SA2=-1;
  ndofSA2=0;
  probSA2=-1;
  xTrkVtxSA2=-999;
  yTrkVtxSA2=-999;
  zTrkVtxSA2=-999;
  uTrkVtxSA2=-999;
  vTrkVtxSA2=-999;

  jitter2=-1;
  jPID2=-999;
  majC2=0;
  //majCRatio2=-999;
  //rms2=-1;
  //simpleMajC2=-999;
  smoothMajC2=-999;
  //sqJitter2=-1;
  //totWidth2=-999;

  roID2=-999;
  knn01TrkActivePlanes2=-999;
  knn10TrkMeanPh2=-999;
  knn20LowHighPh2=-999;
  knn40TrkPhFrac2=-999;
  roIDNuMuBar2=-999;
  relativeAngle2=-999;
  
  // jasmine's test variables
  jmID2=-1;
  jmTrackPlane2 =-999;
  jmMeanPh2 = -999;
  jmEndPh2 = -999;
  jmScatteringU2 =-999;
  jmScatteringV2 =-999;
  jmScatteringUV2 =-999;

  xTrkVtx2=-999;
  yTrkVtx2=-999;
  zTrkVtx2=-999;
  uTrkVtx2=-999;
  vTrkVtx2=-999;
  planeTrkVtx2=-1;
  planeTrkBeg2=-1;
  planeTrkBegu2=-1;
  planeTrkBegv2=-1;
  stripTrkBeg2=-1;
  stripTrkBegu2=-1;
  stripTrkBegv2=-1;
  stripTrkBegIsu2=false;
  regionTrkVtx2=-1;
  phiTrkVtx2=-999;
  
  xTrkEnd2=-999;
  yTrkEnd2=-999;
  zTrkEnd2=-999;
  uTrkEnd2=-999;
  vTrkEnd2=-999;
  planeTrkEnd2=-1;
  planeTrkEndu2=-1;
  planeTrkEndv2=-1;

  drTrkFidall2=-999;
  dzTrkFidall2=-999;
  drTrkFidvtx2=-999;
  dzTrkFidvtx2=-999;
  drTrkFidend2=-999;
  dzTrkFidend2=-999;
  traceTrkFidall2 = -999;
  traceTrkFidvtx2 = -999;
  traceTrkFidend2 = -999;
  
  cosPrTrkVtx2 = -999;
  
  //shw variables
  shwExists2=false;
  ndigitShw2=-1;
  nstripShw2=-1;
  nplaneShw2=-1;
  shwEnCor2=-1;
  shwEnNoCor2=-1;
  shwEnLinCCNoCor2=-1;
  shwEnLinCCCor2=-1;
  shwEnWtCCNoCor2=-1;
  shwEnWtCCCor2=-1;
  shwEnLinNCNoCor2=-1;
  shwEnLinNCCor2=-1;
  shwEnWtNCNoCor2=-1;
  shwEnWtNCCor2=-1;
  shwEnMip2=-1;
  planeShwBeg2=-1;
  planeShwEnd2=-1;
  xShwVtx2=-999;
  yShwVtx2=-999;
  zShwVtx2=-999;


  //////////////////////////////////////////////
  //standard ntuple variables for third trk/shw
  trkExists3=false;
  trkIndex3=-1;
  ndigitTrk3=-1;
  nstripTrk3=-1;
  trkEnCorRange3=-1;
  trkEnCorCurv3=-1;
  trkShwEnNear3 = -1;
  trkMomentumRange3=-1;
  trkMomentumTransverse3=-1;
  containedTrk3=0;
  trkfitpass3=-1;
  trkvtxdcosz3=-999;
  trkvtxdcosy3=-999;
  trknplane3=-999;
  charge3=0;
  qp3=-999;
  qp_rangebiased3=-999;
  sigqp3=-1;
  qp_sigqp3=-999;
  chi23=-1;
  ndof3=0;
  qpFraction3=-1;
  trkVtxUVDiffPl3=-999;
  trkLength3=-1;
  planeTrkNu3=-1;
  planeTrkNv3=-1;
  ntrklike3=-1;
  trkphsigcor3=-1;
  trkphsigmap3=-1;

  trkfitpassSA3=-1;
  trkvtxdcoszSA3=-999;
  chargeSA3=0;
  qpSA3=-999;
  sigqpSA3=-1;
  chi2SA3=-1;
  ndofSA3=0;
  probSA3=-1;
  xTrkVtxSA3=-999;
  yTrkVtxSA3=-999;
  zTrkVtxSA3=-999;
  uTrkVtxSA3=-999;
  vTrkVtxSA3=-999;

  jitter3=-1;
  jPID3=-999;
  majC3=0;
  //majCRatio3=-999;
  //rms3=-1;
  //simpleMajC3=-999;
  smoothMajC3=-999;
  //sqJitter3=-1;
  //totWidth3=-999;

  roID3=-999;
  knn01TrkActivePlanes3=-999;
  knn10TrkMeanPh3=-999;
  knn20LowHighPh3=-999;
  knn40TrkPhFrac3=-999;
  roIDNuMuBar3=-999;
  relativeAngle3=-999;
  // jasmine's test variables
  jmID3 = -1;
  jmTrackPlane3 =-999;
  jmMeanPh3 = -999;
  jmEndPh3 = -999;
  jmScatteringU3 =-999;
  jmScatteringV3 =-999;
  jmScatteringUV3 =-999;

  xTrkVtx3=-999;
  yTrkVtx3=-999;
  zTrkVtx3=-999;
  uTrkVtx3=-999;
  vTrkVtx3=-999;
  planeTrkVtx3=-1;
  planeTrkBeg3=-1;
  planeTrkBegu3=-1;
  planeTrkBegv3=-1;
  stripTrkBeg3=-1;
  stripTrkBegu3=-1;
  stripTrkBegv3=-1;
  stripTrkBegIsu3=false;
  
  xTrkEnd3=-999;
  yTrkEnd3=-999;
  zTrkEnd3=-999;
  uTrkEnd3=-999;
  vTrkEnd3=-999;
  planeTrkEnd3=-1;
  planeTrkEndu3=-1;
  planeTrkEndv3=-1;
  regionTrkVtx3=-1;
  phiTrkVtx3=-999;
  
  drTrkFidall3=-999;
  dzTrkFidall3=-999;
  drTrkFidvtx3=-999;
  dzTrkFidvtx3=-999;
  drTrkFidend3=-999;
  dzTrkFidend3=-999;
  traceTrkFidall3 = -999;
  traceTrkFidvtx3 = -999;
  traceTrkFidend3 = -999;
  
  cosPrTrkVtx3 = -999;
  
  //shw variables
  shwExists3=false;
  ndigitShw3=-1;
  nstripShw3=-1;
  nplaneShw3=-1;
  shwEnCor3=-1;
  shwEnNoCor3=-1;
  shwEnLinCCNoCor3=-1;
  shwEnLinCCCor3=-1;
  shwEnWtCCNoCor3=-1;
  shwEnWtCCCor3=-1;
  shwEnLinNCNoCor3=-1;
  shwEnLinNCCor3=-1;
  shwEnWtNCNoCor3=-1;
  shwEnWtNCCor3=-1;
  shwEnMip3=-1;
  planeShwBeg3=-1;
  planeShwEnd3=-1;
  xShwVtx3=-999;
  yShwVtx3=-999;
  zShwVtx3=-999;


  //////////////////////////////////////////////
  //standard ntuple variables for fourth shw
  //shw variables
  shwExists4=false;
  ndigitShw4=-1;
  nstripShw4=-1;
  nplaneShw4=-1;
  shwEnCor4=-1;
  shwEnNoCor4=-1;
  shwEnLinCCNoCor4=-1;
  shwEnLinCCCor4=-1;
  shwEnWtCCNoCor4=-1;
  shwEnWtCCCor4=-1;
  shwEnLinNCNoCor4=-1;
  shwEnLinNCCor4=-1;
  shwEnWtNCNoCor4=-1;
  shwEnWtNCCor4=-1;
  shwEnMip4=-1;
  planeShwBeg4=-1;
  planeShwEnd4=-1;
  xShwVtx4=-999;
  yShwVtx4=-999;
  zShwVtx4=-999;


  //////////////////////////////////////////////
  //standard ntuple variables for fifth shw
  //shw variables
  shwExists5=false;
  ndigitShw5=-1;
  nstripShw5=-1;
  nplaneShw5=-1;
  shwEnCor5=-1;
  shwEnNoCor5=-1;
  shwEnLinCCNoCor5=-1;
  shwEnLinCCCor5=-1;
  shwEnWtCCNoCor5=-1;
  shwEnWtCCCor5=-1;
  shwEnLinNCNoCor5=-1;
  shwEnLinNCCor5=-1;
  shwEnWtNCNoCor5=-1;
  shwEnWtNCCor5=-1;
  shwEnMip5=-1;
  planeShwBeg5=-1;
  planeShwEnd5=-1;
  xShwVtx5=-999;
  yShwVtx5=-999;
  zShwVtx5=-999;


  ////////////////////////
  //other info calculated
  rEvtVtx=-999;
  rEvtEnd=-999;
  distToEdgeEvtVtx=-999;
  evtVtxUVDiffPl=-999;

  rTrkVtx=-999;
  rTrkEnd=-999;
  sigqp_qp=-999;
  chi2PerNdof=999;
  prob=-1;

  containmentFlag=-1;
  containmentFlagCC0093Std=-1;
  containmentFlagCC0250Std=-1;
  containmentFlagPitt=-1;
  usedRange=0;
  usedCurv=0;


  /////////
  //weights
  /////////
  rw=1;//default of no reweight
  fluxErr=-1;
  rwActual=1;//default of no reweight
  generatorWeight=1;//no weight
  detectorWeight=1;//no weight

  trkEnWeight=1;//no weight
  shwEnWeight=1;//no weight
  beamWeight=1;//no weight
  fluxErrHadProdAfterTune=-1;
  fluxErrTotalErrorPreTune=-1;
  fluxErrTotalErrorAfterTune=-1;
  detectorWeightNMB=1;//no weight
  detectorWeightNM=1;//no weight

  trkEnWeightRunI=1;//no weight
  shwEnWeightRunI=1;//no weight
  beamWeightRunI=1;//no weight
  fluxErrHadProdAfterTuneRunI=-1;
  fluxErrTotalErrorPreTuneRunI=-1;
  fluxErrTotalErrorAfterTuneRunI=-1;
  detectorWeightNMBRunI=1;//no weight
  detectorWeightNMRunI=1;//no weight
  
  trkEnWeightRunII=1;//no weight
  shwEnWeightRunII=1;//no weight
  beamWeightRunII=1;//no weight
  fluxErrHadProdAfterTuneRunII=-1;
  fluxErrTotalErrorPreTuneRunII=-1;
  fluxErrTotalErrorAfterTuneRunII=-1;
  detectorWeightNMBRunII=1;//no weight
  detectorWeightNMRunII=1;//no weight

  //energies with and without weights
  energyRw=-1;
  energyNoRw=-1;
  trkEnRw=-1;
  trkEnNoRw=-1;
  shwEnRw=-1;
  shwEnNoRw=-1;

  //pids
  dpID=-999;
  abID=-999;
  roID=-999;
  knn01TrkActivePlanes=-999;
  knn10TrkMeanPh=-999;
  knn20LowHighPh=-999;
  knn40TrkPhFrac=-999;
  roIDNuMuBar=-999;
  relativeAngle=-999;
  poID=-999;
  poIDKin=-999;
  // jasmine ID
  jmID = -1;
  jmTrackPlane =-999;
  jmMeanPh = -999;
  jmEndPh = -999;
  jmScatteringU =-999;
  jmScatteringV =-999;
  jmScatteringUV =-999;
  jmEventknnID=-999;
  jmEventknn208=-999;
  jmEventknn207=-999;
  jmEventknn206=-999;
  jmEventknn205=-999;
  jmEventknn204=-999;


  ////////////////////////////
  // NC variables
  ////////////////////////////
  // event
  closeTimeDeltaZ=-1;
  edgeActivityStrips=-1;
  edgeActivityPH=-1;
  oppEdgeStrips=-1;
  oppEdgePH=-1;
  vtxMetersToCoilEvt=-1;
  vtxMetersToCloseEdgeEvt=-1;
  minTimeSeparation=-1;
  // shower
  transverseRMSU=-1;
  transverseRMSV=-1;
  // track
  dtdz=-1;
  endMetersToCloseEdge=-1;
  vtxMetersToCloseEdgeTrk=-1;
  vtxMetersToCoilTrk=-1;
  traceEndZ=-1;
    
  //beam variables
  pot=-1;
  potDB=-1;
  potSinceLastEvt=0;    //set to zero in case it's used to count
  potSinceLastEvtGood=0;//rather than being set to a value
  potSinceLastEvtBad=0;
  potSinceLastEvtDB=0;
  potSinceLastEvtGoodDB=0;
  potSinceLastEvtBadDB=0;

  runPeriod=-1;
  hornIsReverse=false;  
  beamTypeDB=0;
  beamType=0;
  intensity=-1;
  hornCur=-999999;//might want 0 horn current
  goodBeam=false;
  goodBeamSntp=true;//false is safer, but old ntuples don't have it
  
  //////////////////
  //truth variables
  //////////////////

  //////////////////////////////////////////////////////////////////
  //EVERYTIME A TRUTH VARIABLE IS ADDED TO THIS CLASS IT MUST
  //ALSO BE ADDED TO NuMCEvent
  ///////////////////////////////////////////////////////////////////

  energyMC=-1;

  neuEnMC=-1;
  neuPxMC=-1;
  neuPyMC=-1;
  neuPzMC=-1;  

  mu1EnMC=-1;
  mu1PxMC=-1;
  mu1PyMC=-1;
  mu1PzMC=-1;

  tgtEnMC=-1;
  tgtPxMC=-1;
  tgtPyMC=-1;
  tgtPzMC=-1;

  zMC=-1;
  aMC=-1;
  nucleusMC=-1;
  initialStateMC=-1;
  hadronicFinalStateMC=-1;

  yMC=-1;
  y2MC=-1;
  xMC=-1;
  q2MC=-1;
  w2MC=-1;
  
  trkEnMC=-1;
  trkEn2MC=-1;
  shwEnMC=-1;
  shwEn2MC=-1;
  
  trkEndEnMC = -1;
  trkStartEnMC = -1;
  trkContainmentMC = false;
  
  sigma=999999;
  iaction=-1;
  iresonance=-1;
  inu=0;
  inunoosc=0;
  itg=0;

  vtxxMC=-999;
  vtxyMC=-999;
  vtxzMC=-999;
  vtxuMC=-999;
  vtxvMC=-999;
  planeTrkVtxMC=-999;
  rTrkVtxMC=-999;

  mc=-1;
  mcTrk=-1;
  mcShw=-1;
  mcEvt=-1;

  mcTrk1 = -1;
  mcTrk2 = -1;
  mcTrk3 = -1;

  mcShw1 = -1;
  mcShw2 = -1;
  mcShw3 = -1;
  mcShw4 = -1;
  mcShw5 = -1;
  
  Npz=-1;
  NdxdzNea=-1;
  NdydzNea=-1;
  NenergyN=-1;
  NWtNear=-1;
  NdxdzFar=-1;
  NdydzFar=-1;
  NenergyF=-1;
  NWtFar=-1;
  Ndecay=-1;
  Vx=-1;
  Vy=-1;
  Vz=-1;
  pdPx=-1;
  pdPy=-1;
  pdPz=-1;
  ppdxdz=-1;
  ppdydz=-1;
  pppz=-1;
  ppenergy=-1;
  ppmedium=-1;
  ppvx=-1;
  ppvy=-1;
  ppvz=-1;
  ptype=-1;
  Necm=-1;
  Nimpwt=-1;
  tvx=-1;
  tvy=-1;
  tvz=-1;
  tpx=-1;
  tpy=-1;
  tpz=-1;
  tptype=-1;
  tgen=-1;

  InukeNwts  =  -999 ;
  InukePiCExchgP  =  -999 ;   //0
  InukePiCExchgN  =  -999 ;   //1
  InukePiEScatP  =  -999 ;  //2
  InukePiEScatN  =  -999 ;  //3
  InukePiInEScatP  =  -999 ;  //4
  InukePiInEScatN  =  -999 ;  //5
  InukePiAbsorbP  =  -999 ;   //6 
  InukePiAbsorbN  =  -999 ;   //7  
  InukePi2PiP  =  -999 ;      //8   
  InukePi2PiN  =  -999 ;      //9   
  InukeNknockP  =  -999 ;     //10
  InukeNknockN  =  -999 ;     //11
  InukeNNPiP  =  -999 ;       //12
  InukeNNPiN  =  -999 ;       //13
  InukeFormTP  =  -999 ;      //14
  InukeFormTN  =  -999 ;      //15
  InukePiXsecP  =  -999 ;     //16
  InukePiXsecN  =  -999 ;     //17
  InukeNXsecP  =  -999 ;      //18
  InukeNXsecN  =  -999 ;      //19
  InukeNucrad  =  -999 ;
  InukeWrad  =  -999 ;

  //////////////////////////////////////////////////////////////////
  //EVERYTIME A TRUTH VARIABLE IS ADDED TO THIS CLASS IT MUST
  //ALSO BE ADDED TO NuMCEvent
  ///////////////////////////////////////////////////////////////////

  ////////////////////////////
  //program control variables
  ////////////////////////////
  anaVersion=0;
  releaseType=-1;//the value of Conventions/ReleaseType::kUnknown
  recoVersion=-1;//the value of Conventions/ReleaseType::kUnknown
  mcVersion=-1;//the value of Conventions/ReleaseType::kUnknown
  reweightVersion=0;
  //beamType=0;

  useGeneratorReweight=1;//default is to do the generatorReweighting
  sGeneratorConfigName="Unknown";
  generatorConfigNo=-1;
  
  //set all these to true by default, so they are used and cut on
  useDBForDataQuality=true;
  useDBForSpillTiming=true;
  useDBForBeamInfo=true;
  
  cutOnDataQuality=true;
  cutOnSpillTiming=true;
  cutOnBeamInfo=true;
  
  applyEnergyShifts=false;
  applyBeamWeight=true;
  apply1SigmaWeight=false;
  applyDetectorWeight=false;
  applyGeneratorWeight=false;

  calcMajCurv=true;
  calcRoID=true;
  calcJmID=true;
}

//......................................................................
