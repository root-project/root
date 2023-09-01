# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# Library for RooFit to interface with CUDA.

ROOT_LINKER_LIBRARY(RooFitCuda src/CudaInterface.cu src/CudaHelpers.cxx TYPE SHARED)
set_target_properties(RooFitCuda PROPERTIES CUDA_RUNTIME_LIBRARY Shared)

ROOT_INSTALL_HEADERS()
