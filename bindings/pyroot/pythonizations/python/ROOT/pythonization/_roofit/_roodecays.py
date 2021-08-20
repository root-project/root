# Authors:
# * Jonas Rembser 06/2021
# * Harshal Shende 06/2021

################################################################################
# Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################


from ._utils import _decaytype_string_to_enum, cpp_signature


class RooDecay(object):
    """Some constructors of classes like RooDecay, RooBDecay, RooBCPGenDecay, RooBCPEffDecay and RooBMixDecay that take an enum
    DecayType as argument also support keyword arguments.
    For example, the following code is equivalent in PyROOT:
    \code{.py}
    # Directly passing keyword argument with string corresponding to enum value name:
    decay_tm = ROOT.RooDecay("decay_tm", "decay", dt, tau, tm, ROOT.RooDecay.DoubleSided)

    # With enum value:
    decay_tm = ROOT.RooDecay("decay_tm", "decay", dt, tau, tm, type="DoubleSided")
    \endcode
    """

    @cpp_signature(
        "RooDecay(const char *name, const char *title, RooRealVar& t, RooAbsReal& tau, const RooResolutionModel& model, DecayType type) ;"
    )
    def __init__(self, *args, **kwargs):
        """The RooDecay constructor is pythonized with enum values."""
        kwargs = _decaytype_string_to_enum(self, kwargs)
        self._init(*args, **kwargs)


class RooBDecay(object):
    @cpp_signature(
        "RooBDecay(const char *name, const char *title, RooRealVar& t,"
        "    RooAbsReal& tau, RooAbsReal& dgamma,    RooAbsReal& f0,"
        "    RooAbsReal& f1, RooAbsReal& f2,    RooAbsReal& f3, RooAbsReal& dm,"
        "    const RooResolutionModel& model,   DecayType type);"
    )
    def __init__(self, *args, **kwargs):
        """The RooBDecay constructor is pythonized with enum values."""
        kwargs = _decaytype_string_to_enum(self, kwargs)
        self._init(*args, **kwargs)


class RooBCPGenDecay(object):
    @cpp_signature(
        "RooBCPGenDecay(const char *name, const char *title, RooRealVar& t, RooAbsCategory& tag,"
        "    RooAbsReal& tau, RooAbsReal& dm, RooAbsReal& avgMistag, RooAbsReal& a, RooAbsReal& b,"
        "    RooAbsReal& delMistag, RooAbsReal& mu, const RooResolutionModel& model, DecayType type=DoubleSided) ;"
    )
    def __init__(self, *args, **kwargs):
        """The RooBCPGenDecay constructor is pythonized with enum values."""
        kwargs = _decaytype_string_to_enum(self, kwargs)
        self._init(*args, **kwargs)


class RooBCPEffDecay(object):
    @cpp_signature(
        "RooBCPEffDecay(const char *name, const char *title, RooRealVar& t, RooAbsCategory& tag,"
        "    RooAbsReal& tau, RooAbsReal& dm, RooAbsReal& avgMistag, RooAbsReal& CPeigenval,"
        "    RooAbsReal& a, RooAbsReal& b, RooAbsReal& effRatio, RooAbsReal& delMistag,"
        "    const RooResolutionModel& model, DecayType type=DoubleSided) ;"
    )
    def __init__(self, *args, **kwargs):
        """The RooBCPEffDecay constructor is pythonized with enum values."""
        kwargs = _decaytype_string_to_enum(self, kwargs)
        self._init(*args, **kwargs)


class RooBMixDecay(object):
    @cpp_signature(
        "RooBMixDecay(const char *name, const char *title, RooRealVar& t, RooAbsCategory& mixState,"
        "    RooAbsCategory& tagFlav, RooAbsReal& tau, RooAbsReal& dm, RooAbsReal& mistag, "
        "    RooAbsReal& delMistag, const RooResolutionModel& model, DecayType type=DoubleSided) ;"
    )
    def __init__(self, *args, **kwargs):
        """The RooBMixDecay constructor is pythonized with enum values."""
        kwargs = _decaytype_string_to_enum(self, kwargs)
        self._init(*args, **kwargs)
