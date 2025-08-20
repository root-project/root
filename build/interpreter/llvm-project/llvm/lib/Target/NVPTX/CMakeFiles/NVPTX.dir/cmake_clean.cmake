file(REMOVE_RECURSE
  "NVPTXGenAsmWriter.inc"
  "NVPTXGenDAGISel.inc"
  "NVPTXGenInstrInfo.inc"
  "NVPTXGenRegisterInfo.inc"
  "NVPTXGenSubtargetInfo.inc"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/NVPTX.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
