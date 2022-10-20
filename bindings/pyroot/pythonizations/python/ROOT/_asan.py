# Author: Jonas Hahnfeld CERN  10/2022

################################################################################
# Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

import os

# Filter out ASan related libraries that need to be preloaded into the
# Python process, but must not be propagated into instrumented executables
# such as root or rootcling.
if 'LD_PRELOAD' in os.environ:
    new_preload = []
    for lib in os.environ['LD_PRELOAD'].split(':'):
        if 'libROOTSanitizerConfig' in lib or 'libclang_rt.asan-' in lib:
            continue
        new_preload.append(lib)
    os.environ['LD_PRELOAD'] = ':'.join(new_preload)

