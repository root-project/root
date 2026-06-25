/*
 * Project: RooFit
 * Author:
 *   Stephan Hageboeck, CERN 2019
 *
 * Copyright (c) 2023, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef RooFit_RooHelpers_h
#define RooFit_RooHelpers_h

#include <RooMsgService.h>
#include <RooAbsArg.h>
#include <RooAbsReal.h>

#include <ROOT/RSpan.hxx>

#include <sstream>
#include <list>
#include <vector>
#include <string>
#include <utility>

class RooAbsPdf;
class RooAbsData;
class RooAbsRealLValue;

namespace RooHelpers {

/// Switches the message service to a different level while the instance is alive.
/// Can also temporarily activate / deactivate message topics.
/// Use as
/// ~~~{.cpp}
/// RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);
/// [ statements that normally generate a lot of output ]
/// ~~~
class LocalChangeMsgLevel {
public:
   /// Change message level (and topics) while this object is alive, reset when it goes out of scope.
   /// \param[in] lvl The desired message level. Defaults to verbose.
   /// \param[in] extraTopics Extra topics to be switched on. These will only switched on in the last stream to prevent
   /// all streams are printing. \param[in] removeTopics Message topics to be switched off \param[in]
   /// overrideExternalLevel Override the user message level.
   LocalChangeMsgLevel(RooFit::MsgLevel lvl = RooFit::DEBUG, unsigned int extraTopics = 0u,
                       unsigned int removeTopics = 0u, bool overrideExternalLevel = true);

   ~LocalChangeMsgLevel();

private:
   RooFit::MsgLevel fOldKillBelow;
   std::vector<RooMsgService::StreamConfig> fOldConf;
   int fExtraStream{-1};
};

/// Wrap an object into a TObject. Sometimes needed to avoid reinterpret_cast or enable RTTI.
template <typename T>
struct WrapIntoTObject : public TObject {
   WrapIntoTObject(T &obj) : _payload(&obj) {}
   T *_payload;
};

/// Hijacks all messages with given level and topic (and optionally object name) while alive.
/// Use this like an ostringstream afterwards. The messages can e.g. be retrieved using `str()`.
/// Useful for unit tests / debugging.
class HijackMessageStream {
public:
   HijackMessageStream(RooFit::MsgLevel level, RooFit::MsgTopic topics, const char *objectName = nullptr);
   template <typename T>
   const HijackMessageStream &operator<<(const T &v) const
   {
      _str << v;
      return *this;
   }
   std::string str() { return _str.str(); }
   std::ostringstream &stream() { return _str; };
   ~HijackMessageStream();

private:
   std::ostringstream _str;
   RooFit::MsgLevel _oldKillBelow;
   std::vector<RooMsgService::StreamConfig> _oldConf;
   Int_t _thisStream;
};

/// Check if the parameters have a range, and warn if the range extends below / above the set limits.
void checkRangeOfParameters(const RooAbsReal *callingClass, std::initializer_list<const RooAbsReal *> pars,
                            double min = -std::numeric_limits<double>::max(),
                            double max = std::numeric_limits<double>::max(), bool limitsInAllowedRange = false,
                            std::string const &extraMessage = "");

/// set all RooRealVars to constants. return true if at least one changed status
bool setAllConstant(const RooAbsCollection &coll, bool constant = true);

/// Check that `function` is constant (flat) inside each bin defined by the
/// sorted `boundaries` when scanning the observable `obs`. Several interior
/// points are sampled per bin and compared to the bin's first sample; if any
/// of them deviates by more than `relTol` (relative to the value scale), the
/// function is not flat and false is returned. The value of `obs` is restored
/// on return.
bool isFunctionFlatInBins(const RooAbsReal &function, RooAbsRealLValue &obs, std::span<const double> boundaries,
                          double relTol = 1e-9);

/// Return a newly allocated list with the subset of `boundaries` that lies
/// strictly inside [`xlo`, `xhi`], with `xlo` and `xhi` added as the first and
/// last entries. This is the form expected by RooFit's binBoundaries()
/// interface, so the bin integrator covers exactly the integration range.
/// The caller takes ownership of the returned list.
std::list<double> *binBoundariesInRange(std::span<const double> boundaries, double xlo, double xhi);

} // namespace RooHelpers

#endif
