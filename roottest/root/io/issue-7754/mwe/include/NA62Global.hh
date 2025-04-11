// ---------------------------------------------------------------
// History:
//
// Created by Evgueni Goudzovski (eg@hep.ph.bham.ac.uk) 2016-01-21
//
// ---------------------------------------------------------------

#ifndef NA62Global_H
#define NA62Global_H

#define ClockPeriod 24.951059536        // nominal clock period in ns [= 24/(961.883 MHz)]
#define TdcCalib    (ClockPeriod/256.0) // nominal HPTDC LSB in ns

// An enumerated type used for detector ID by the Reconstruction and Analysis
// Detectors are numbered following the raw data ID (DetectorID = DAQ_ID/4)
// *** DO NOT CHANGE THE NUMERICAL VALUES! ***

namespace NA62 {

enum DetectorID {
  kDummy=0,
  kCedar=1,
  kGigaTracker=2,
  kCHANTI=3,
  kLAV=4,
  kSpectrometer=5,
  kCHOD=6,
  kRICH=7,
  kIRC=8,
  kLKr=9,
  kMUV1=10,
  kMUV2=11,
  kMUV3=12,
  kSAC=13,
  kNewCHOD=14,  // used to be MUV0, provisional (fake)
  kHAC=15,
  kL0TP=16,
  kL1TP=17,
  kL2EB=18,
  kDIM=19,
  kMUV0=20,     // provisional (fake)
  kSAV=21,      // provisional (fake)
  kNULL=22,
  kProcessing=30
};

// LAV station indices
enum LAVStationID {
  kLAV1=0,
  kLAV2=1,
  kLAV3=2,
  kLAV4=3,
  kLAV5=4,
  kLAV6=5,
  kLAV7=6,
  kLAV8=7,
  kLAV9=8,
  kLAV10=9,
  kLAV11=10,
  kLAV12=11
};

// Cedar PMT types
enum CedarPMTType {
  kPMT_EMI_9820_QB = 1,
  kPMT_R7400U_03   = 2,
  kPMT_R9880U_110  = 3,
  kPMT_R9880U_210  = 4,
  kPMT_XP85012_S   = 5
};

// An enumerated type used for L0 detector ID by the Reconstruction and Analysis
// Detectors are numbered following the L0TP data ID
// *** DO NOT CHANGE THE NUMERICAL VALUES! ***

enum L0DetectorID {
  kL0CHOD=0,
  kL0RICH=1,
  kL0LAV=2,
  kL0MUV3=3,
  kL0NewCHOD=4,
  kL0TALK=5,
  kL0Calo=6
};

// An enumerated type used for L1 algorithm ID by the Reconstruction and Analysis
// Detectors are numbered following the L1TP data ID
// *** DO NOT CHANGE THE NUMERICAL VALUES! ***

enum L1AlgoID {
  kL1CHOD=0,
  kL1RICH=1,
  kL1KTAG=2,
  kL1LAV=3,
  kL1IRCSAC=4,
  kL1Spectrometer=5,
  kL1MUV3=6,
  kL1NewCHOD=7
};

// An enumerated type used for L2 algorithm ID by the Reconstruction and Analysis
// Detectors are numbered following the L2EB data ID
// *** DO NOT CHANGE THE NUMERICAL VALUES! ***

enum L2AlgoID {
  kL2LKr=0
};

enum {kTightCandidate, kLooseCandidate, kLooseMaskedCandidate, kUndefinedCandidate};
enum {kAdjacentCells, kTracking, kNotPass, kUnknown, kMIP, kShower};
enum {kJura=0, kSaleve=1, kTop=2, kBottom=3, kUp=0, kDown=1, kPMT=0, kOr=1};
enum {kX=0, kY=1, kPositive=0, kNegative=1};
enum {kVerticalPlane=0, kHorizontalPlane=1};
enum {kLeadingEdge=1, kTrailingEdge=2, kBothEdges=3};
enum {kL0TriggerSlot=0, kL0PreviousSlot=1, kL0NextSlot=2}; // This order follows the L0TP data one *** DO NOT CHANGE THE NUMERICAL VALUES! ***
enum {kCREAMSaturationBit=1, kCREAMUnderflowBit=4, kCREAMNoCalibrationBit=5, kCREAML1ErrorBit=12};

enum {kCedarWest, kCedarNorth, kCedarH};    // Cedar internal optics types
enum {kNitrogen, kHelium, kHydrogen, kAir}; // Cedar radiator gas options

enum {kShifter=0, kExpert=1};
enum {kHistoT0=0, kHistoNoT0=1, kHistoNoT0Prim=2,kHistoFineTimeDriftT0=3};

// L1Counters, defined in NA62Reconstruction
#define NMAXL0MASKS 16

enum {
  kL1APTotal = 1,
  kL1APControl = 2,
  kL1APPeriodics = 3,
  kL1APPhysicsPassed = 4,
  kL1APPhysics = 5,
  kL1APPhysicsPassedPerMask = 6,
  kL1APPhysicsPerMask = kL1APPhysicsPassedPerMask+NMAXL0MASKS,
  kL1PhysicsPassedPerMask = kL1APPhysicsPassedPerMask+2*NMAXL0MASKS
};

// NA62FW return codes
enum {
  kSuccess=0,
  kGenericError=1,
  kConditionOffset=1, // this should always point to the beginning of the NA62ConditionService return codes
  kConditionFileNotFound=kConditionOffset+1,
  kConditionFileIsEmpty=kConditionOffset+2,
  kConditionFileIsNotReadable=kConditionOffset+3,
  kWrongConfiguration=5,
  kWriteError=6,
  kReadError=7,
  kBadBurstHeader=8,
  kBadHeader=9,
  kBadEventIndex=10,
  kAllBadBursts=11
};

namespace Verbosity{
// Always is always printed, whatever the verbosity level
// Normal/UserNormal is the default output (same as standard, but standard suppresses the prefix with time and module name):
//   Should be printing everything rhe framework does rarely (such as initialization), and non-recurrent error/warnings
//   messages (arising typically when user does something wrong).
// Extended/User:
//   Print all the other occasional messages that users might want to see, but not by default. Definitely no information
//   that will be printed for all event
// Debug:
//   Print detailed debugging information that user do not usually want to see when running normally. Will be used
//   when it is needed to understand roughly what the framework is doing (typically debug but exclude messages from within loop).
//   Also include messages printed for all events
// Trace:
//   Highest level of printing. Print everything, especially content of loops and things that gets printed many times/event
//
// Core levels should be used in the framework core.
// Analyzer levels should be used in all analyzers.
enum CoreVerbosityLevel {kAlways, kNormal, kExtended, kDebug, kTrace, kCDisable};
enum AnalyzerVerbosityLevel {kUserAlways, kUserNormal, kUser, kUDisable};
} /* namespace Verbosity */

} //namespace NA62
#endif
