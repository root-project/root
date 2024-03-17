include(${CMAKE_CURRENT_LIST_DIR}/config.cmake OPTIONAL)

set(ccache ON CACHE BOOL "" FORCE)
set(cocoa ON CACHE BOOL "" FORCE)

# Enable builtins on macOS
foreach(dep cfitsio cppzmq davix fftw3 freetype ftgl gl2ps glew gsl gtest lz4 lzma
            nlohmannjson pcre openssl tbb vc vdt veccore xrootd xxhash zeromq zstd)
  set(builtin_${dep} ON CACHE BOOL "" FORCE)
endforeach()

# Disable options that should not be built on macOS
foreach(option arrow cefweb clad cuda cudnn dcache monalisa gfal jemalloc
        mysql pgsql pythia6 r test_distrdf_dask test_distrdf_pyspark tmva
        tmva-cpu tmva-gpu tmva-pymva tmva-rmva tmva-sofie x11)
  set(${option} OFF CACHE BOOL "" FORCE)
endforeach()
