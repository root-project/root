############################################################################
# CMakeLists.txt file for building ROOT core/brotli package
############################################################################
#
# inspired by @zzxuanyuan's pull request https://github.com/root-mirror/root/pull/81

#---Declare ZipBROTLI sources as part of libCore-------------------------------
set(sources ${CMAKE_CURRENT_SOURCE_DIR}/src/ZipBROTLI.cc ${CMAKE_CURRENT_SOURCE_DIR}/src/enc/backward_references.cc ${CMAKE_CURRENT_SOURCE_DIR}/src/enc/block_splitter.cc ${CMAKE_CURRENT_SOURCE_DIR}/src/enc/brotli_bit_stream.cc ${CMAKE_CURRENT_SOURCE_DIR}/src/enc/compress_fragment.cc ${CMAKE_CURRENT_SOURCE_DIR}/src/enc/compress_fragment_two_pass.cc ${CMAKE_CURRENT_SOURCE_DIR}/src/enc/encode.cc ${CMAKE_CURRENT_SOURCE_DIR}/src/enc/encode_parallel.cc ${CMAKE_CURRENT_SOURCE_DIR}/src/enc/entropy_encode.cc ${CMAKE_CURRENT_SOURCE_DIR}/src/enc/histogram.cc ${CMAKE_CURRENT_SOURCE_DIR}/src/enc/literal_cost.cc ${CMAKE_CURRENT_SOURCE_DIR}/src/enc/metablock.cc ${CMAKE_CURRENT_SOURCE_DIR}/src/enc/static_dict.cc ${CMAKE_CURRENT_SOURCE_DIR}/src/enc/streams.cc ${CMAKE_CURRENT_SOURCE_DIR}/src/enc/utf8_util.cc ${CMAKE_CURRENT_SOURCE_DIR}/src/dec/bit_reader.c ${CMAKE_CURRENT_SOURCE_DIR}/src/dec/decode.c ${CMAKE_CURRENT_SOURCE_DIR}/src/dec/dictionary.c ${CMAKE_CURRENT_SOURCE_DIR}/src/dec/huffman.c ${CMAKE_CURRENT_SOURCE_DIR}/src/dec/state.c)

ROOT_OBJECT_LIBRARY(Brotli ${sources})
