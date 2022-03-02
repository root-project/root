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


from ._utils import _kwargs_to_roocmdargs, cpp_signature


class RooGenFitStudy(object):
    r"""Some member functions of RooGenFitStudy that take a RooCmdArg as argument also support keyword arguments.
    So far, this applies to RooGenFitStudy::setGenConfig.
    """

    @cpp_signature(
        [
            "RooGenFitStudy::setGenConfig(const char* pdfName, const char* obsName, const RooCmdArg& arg1=RooCmdArg(), const RooCmdArg& arg2=RooCmdArg(),const RooCmdArg& arg3=RooCmdArg()) ;",
            "RooGenFitStudy::setFitConfig(const char* pdfName, const char* obsName, const RooCmdArg& arg1=RooCmdArg(),const RooCmdArg& arg2=RooCmdArg(),const RooCmdArg& arg3=RooCmdArg()) ;",
        ]
    )
    def setGenConfig(self, *args, **kwargs):
        r"""The RooGenFitStudy::setGenConfig() function is pythonized with the command argument pythonization.
        The keywords must correspond to the CmdArg of the function.
        """
        # Redefinition of `RooGenFitStudy.setGenConfig` for keyword arguments.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._setGenConfig(*args, **kwargs)
