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


r"""
/**
\class RooDecay
\brief \parblock \endparblock
\htmlonly
<div class="pyrootbox">
\endhtmlonly

## PyROOT

Some constructors of classes like RooDecay, RooBDecay, RooBCPGenDecay, RooBCPEffDecay and RooBMixDecay that take an enum
DecayType as argument also support keyword arguments.
For example, the following code is equivalent in PyROOT:
\code{.py}
# Directly passing keyword argument with string corresponding to enum value name:
decay_tm = ROOT.RooDecay("decay_tm", "decay", dt, tau, tm, type="DoubleSided")

# With enum value:
decay_tm = ROOT.RooDecay("decay_tm", "decay", dt, tau, tm, ROOT.RooDecay.DoubleSided)
\endcode

\htmlonly
</div>
\endhtmlonly
*/
"""

from ._utils import _decaytype_string_to_enum


class RooDecay(object):
    def __init__(self, *args, **kwargs):
        kwargs = _decaytype_string_to_enum(self, kwargs)
        self._init(*args, **kwargs)


class RooBDecay(object):
    def __init__(self, *args, **kwargs):
        kwargs = _decaytype_string_to_enum(self, kwargs)
        self._init(*args, **kwargs)


class RooBCPGenDecay(object):
    def __init__(self, *args, **kwargs):
        kwargs = _decaytype_string_to_enum(self, kwargs)
        self._init(*args, **kwargs)


class RooBCPEffDecay(object):
    def __init__(self, *args, **kwargs):
        kwargs = _decaytype_string_to_enum(self, kwargs)
        self._init(*args, **kwargs)


class RooBMixDecay(object):
    def __init__(self, *args, **kwargs):
        kwargs = _decaytype_string_to_enum(self, kwargs)
        self._init(*args, **kwargs)
