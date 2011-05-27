############################################################################
# CMakeLists.txt file for building ROOT math/minuit2 package
############################################################################

ROOT_USE_PACKAGE(math/mathcore)
ROOT_USE_PACKAGE(hist/hist)

add_definitions(-DWARNINGMSG -DUSE_ROOT_ERROR)

#---Deal with the parallel option on Minuit2. Probably it should be done using a build 'option' and not
#   using a environment variable  -- NOT TESTED --- 
if($ENV{USE_PARALLEL_MINUIT2})
  if($ENV{USE_OPENMP})
    add_definitions(-D_GLIBCXX_PARALLEL -fopenmp)
    set_target_properties(Minuit2 PROPERTIES LINK_FLAGS -fopenmp)
  elseif($ENV{USE_MPI})
    add_definitions(-DMPIPROC)
    set(CMAKE_CXX_COMPILER mpic++)
    set(CMAKE_C_COMPILER mpic++)
    set(CMAKE_CXX_LINK_EXECUTABLE mpic++)
  endif()
endif()

ROOT_GENERATE_DICTIONARY(G__Minuit2 *.h Minuit2/*.h LINKDEF LinkDef.h)
ROOT_GENERATE_ROOTMAP(Minuit2 LINKDEF LinkDef.h)
ROOT_LINKER_LIBRARY(Minuit2 *.cxx G__Minuit2.cxx LIBRARIES MathCore Hist)
ROOT_INSTALL_HEADERS()

