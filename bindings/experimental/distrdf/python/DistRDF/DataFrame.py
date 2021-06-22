## @author Vincenzo Eduardo Padulano
#  @author Enric Tejedor
#  @date 2021-02

################################################################################
# Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

from __future__ import print_function

import logging

from DistRDF import Proxy

logger = logging.getLogger(__name__)


class RDataFrame(object):
    """
    Interface to an RDataFrame that can run its computation graph distributedly.
    """

    def __init__(self, headnode, backend):
        """Initialization of """

        self._headnode = headnode

        self._headnode.backend = backend

        self._headproxy = Proxy.TransformationProxy(self._headnode)

    def __dir__(self):
        opdir = self._headnode.backend.supported_operations + super().__dir__()
        opdir.sort()
        return opdir

    def __getattr__(self, attr):
        """getattr"""
        return getattr(self._headproxy, attr)
