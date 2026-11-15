set(_root_classes TriggerPrimitive Decoder Puppi Muons Candidate)
# Bits 0-4 select these wrappers, in order.
set(_wrapped_classes
    "pair<edm::Ptr<l1t::io_v1::L1Candidate>,float>"
    "edm::Ptr<l1t::io_v1::PFCandidate>"
    "edm::Ref<vector<l1t::io_v1::SAMuon>,2>"
    "edm::Ptr<reco::io_v1::GsfTrack>"
    "edm::Ptr<ticl::io_v1::Trackster>")
# Bits 5-13 select these vectors, in order.
set(_vector_classes
    "vector<EcalEBTriggerPrimitiveSample>"
    "vector<pair<edm::Ptr<l1t::io_v1::L1Candidate>,float> >"
    "vector<edm::Ptr<l1t::io_v1::PFCandidate> >"
    "vector<l1ct::io_v1::JetTagClass>"
    "vector<edm::Ref<vector<l1t::io_v1::SAMuon>,2> >"
    "vector<l1t::io_v1::SAMuon>"
    "vector<edm::Ptr<reco::io_v1::GsfTrack> >"
    "vector<edm::Ptr<ticl::io_v1::Trackster> >"
    "vector<array<unsigned int,2> >")

# The top level classes are always selected. In addition, cover:
# - no wrappers or vectors, and each individual wrapper or vector;
# - all wrappers, all vectors, and all wrappers and vectors;
# - the specific nested selections below, where a dictionary for an outer
#   type can hide a missing dictionary for an inner type.
set(_masks 0)
foreach(_index RANGE 0 13)
  math(EXPR _bit "1 << ${_index}")
  list(APPEND _masks ${_bit})
endforeach()
list(APPEND _masks 31)
math(EXPR _all_vectors "((1 << 9) - 1) << 5")
list(APPEND _masks ${_all_vectors})
math(EXPR _all_selected "(1 << 14) - 1")
list(APPEND _masks ${_all_selected})

# Decoder: pair<Ptr<L1Candidate>, float> and vector<pair<Ptr<L1Candidate>, float>>.
math(EXPR _mask "(1 << 0) | (1 << 6)")
list(APPEND _masks ${_mask})
# Puppi: Ptr<PFCandidate> and vector<Ptr<PFCandidate>>.
math(EXPR _mask "(1 << 1) | (1 << 7)")
list(APPEND _masks ${_mask})
# Muons: Ref<vector<SAMuon>, 2> and vector<Ref<vector<SAMuon>, 2>>.
math(EXPR _mask "(1 << 2) | (1 << 9)")
list(APPEND _masks ${_mask})
# Muons: Ref<vector<SAMuon>, 2> and vector<SAMuon>.
math(EXPR _mask "(1 << 2) | (1 << 10)")
list(APPEND _masks ${_mask})
# Muons: all three nested types above.
math(EXPR _ref_vector_bit "1 << 9")
math(EXPR _samuon_vector_bit "1 << 10")
math(EXPR _ref_wrapper_bit "1 << 2")
math(EXPR _mask "${_ref_wrapper_bit} | ${_ref_vector_bit} | ${_samuon_vector_bit}")
list(APPEND _masks ${_mask})
# Candidate: Ptr<GsfTrack> and vector<Ptr<GsfTrack>>.
math(EXPR _mask "(1 << 3) | (1 << 11)")
list(APPEND _masks ${_mask})
# Candidate: Ptr<Trackster> and vector<Ptr<Trackster>>.
math(EXPR _mask "(1 << 4) | (1 << 12)")
list(APPEND _masks ${_mask})
list(REMOVE_DUPLICATES _masks)

foreach(_mask IN LISTS _masks)
  set(_selection "<lcgdict>\n")
  foreach(_class IN LISTS _root_classes)
    string(APPEND _selection "  <class name=\"${_class}\" />\n")
  endforeach()
  foreach(_index RANGE 0 13)
    if(_index LESS 5)
      set(_class_set _wrapped_classes)
      set(_class_index ${_index})
    else()
      set(_class_set _vector_classes)
      math(EXPR _class_index "${_index} - 5")
    endif()
    math(EXPR _selected "(${_mask} >> ${_index}) & 1")
    if(_selected)
      list(GET ${_class_set} ${_class_index} _class)
      string(APPEND _selection "  <class name=\"${_class}\" />\n")
    endif()
  endforeach()
  string(APPEND _selection "</lcgdict>\n")
  set(_selection_file "${CMAKE_CURRENT_BINARY_DIR}/partial${_mask}.xml")
  file(WRITE "${_selection_file}" "${_selection}")
  set(_fixture "root-meta-missingDictionaries-${_mask}-fixture")

  ROOTTEST_GENERATE_REFLEX_DICTIONARY(missingDictionaries${_mask}
                                      header.h
                                      SELECTION "${_selection_file}"
                                      NO_ROOTMAP
                                      FIXTURES_SETUP ${_fixture})

  ROOTTEST_ADD_TEST(missingDictionaries${_mask}
                    MACRO checkMissingDictionaries.C
                    MACROARG
                      "${_mask},\"${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}missingDictionaries${_mask}_dictrflx\""
                    FIXTURES_REQUIRED ${_fixture})
endforeach()
