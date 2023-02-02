#  @author Vincenzo Eduardo Padulano
#  @author Enric Tejedor
#  @date 2021-02

################################################################################
# Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################
from __future__ import annotations

import itertools
import logging

from typing import Any, List, TYPE_CHECKING

from DistRDF import Proxy
from DistRDF.Operation import SUPPORTED_OPERATIONS

if TYPE_CHECKING:
    from DistRDF.HeadNode import HeadNode

logger = logging.getLogger(__name__)


class RDataFrame(object):
    """
    Interface to an RDataFrame that can run its computation graph distributedly.
    """

    def __init__(self, headnode: HeadNode) -> None:
        """Initialization of """

        self._headnode = headnode

        self._headproxy = Proxy.TransformationProxy(self._headnode)

    def __dir__(self) -> List[str]:
        opdir: List[str] = [
            el for el in itertools.chain.from_iterable((SUPPORTED_OPERATIONS.keys(), super().__dir__()))
        ]
        opdir.sort()
        return opdir

    def __getattr__(self, attr: str) -> Any:
        """getattr"""
        return getattr(self._headproxy, attr)
