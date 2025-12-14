# roottest/root/physics/CMakeLists.txt

# Register the FANG unit test
# dependencies: Core (for TRandom), MathCore (for Vectors), Physics (for FANG)
ROOT_ADD_GTEST(testFANG
    testFANG.cxx
    LIBRARIES 
        Core 
        MathCore 
        Physics
)