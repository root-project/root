/* @(#)root/base:$Id$ */

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_GLConstants
#define ROOT_GLConstants

//*-*   This contains the map for those OpenGL package constant the ROOT uses
//*-*   We need the second table since CINT doesn't understand the "define" pre-processor command


#ifndef __CINT__
#ifndef GLConstants

#define GLConstants1(abc_prefix) \
                   abc_prefix##QUAD_STRIP,  abc_prefix##QUADS, abc_prefix##COMPILE, abc_prefix##COMPILE_AND_EXECUTE  \
                  ,abc_prefix##LIGHT0    ,  abc_prefix##LIGHT1    ,  abc_prefix##LIGHT2    ,  abc_prefix##LIGHT3     \
                  ,abc_prefix##LIGHT4    ,  abc_prefix##LIGHT5    ,  abc_prefix##LIGHT6    ,  abc_prefix##LIGHT7     \
                  ,abc_prefix##CW        ,  abc_prefix##CCW                                                          \
                  ,abc_prefix##FRONT     ,  abc_prefix##BACK      ,  abc_prefix##FRONT_AND_BACK                      \
                  ,abc_prefix##POINT     ,  abc_prefix##LINE      ,  abc_prefix##FILL                                \
                  ,abc_prefix##ACCUM_ALPHA_BITS, abc_prefix##ACCUM_BLUE_BITS, abc_prefix##ACCUM_CLEAR_VALUE   \
                  ,abc_prefix##ACCUM_GREEN_BITS, abc_prefix##ACCUM_RED_BITS,  abc_prefix##ALPHA_BIAS          \
                  ,abc_prefix##ALPHA_BITS,       abc_prefix##ALPHA_SCALE,     abc_prefix##ALPHA_TEST          \
                  ,abc_prefix##ALPHA_TEST_FUNC,  abc_prefix##ALPHA_TEST_REF,  abc_prefix##ATTRIB_STACK_DEPTH  \
                  ,abc_prefix##AUTO_NORMAL,      abc_prefix##AUX_BUFFERS,     abc_prefix##BLEND               \
                  ,abc_prefix##BLEND_DST,        abc_prefix##BLEND_SRC,       abc_prefix##BLUE_BIAS           \
                  ,abc_prefix##BLUE_BITS,        abc_prefix##BLUE_SCALE,      abc_prefix##COLOR_CLEAR_VALUE   \
                  ,abc_prefix##COLOR_MATERIAL,   abc_prefix##COLOR_MATERIAL_FACE                              \
                  ,abc_prefix##COLOR_MATERIAL_PARAMETER,                      abc_prefix##COLOR_WRITEMASK     \
                  ,abc_prefix##CULL_FACE,        abc_prefix##CULL_FACE_MODE,  abc_prefix##CURRENT_COLOR

#define GLConstants2(abc_prefix) \
                  ,abc_prefix##CURRENT_INDEX,    abc_prefix##CURRENT_NORMAL,  abc_prefix##CURRENT_RASTER_COLOR          \
                  ,abc_prefix##CURRENT_RASTER_DISTANCE,                       abc_prefix##CURRENT_RASTER_INDEX          \
                  ,abc_prefix##CURRENT_RASTER_POSITION,                       abc_prefix##CURRENT_RASTER_TEXTURE_COORDS \
                  ,abc_prefix##CURRENT_RASTER_POSITION_VALID,                 abc_prefix##CURRENT_TEXTURE_COORDS        \
                  ,abc_prefix##DEPTH_BIAS,       abc_prefix##DEPTH_BITS,      abc_prefix##DEPTH_CLEAR_VALUE   \
                  ,abc_prefix##DEPTH_FUNC,       abc_prefix##DEPTH_RANGE,     abc_prefix##DEPTH_SCALE         \
                  ,abc_prefix##DEPTH_TEST,       abc_prefix##DEPTH_WRITEMASK, abc_prefix##DITHER              \
                  ,abc_prefix##DOUBLEBUFFER,     abc_prefix##DRAW_BUFFER,     abc_prefix##EDGE_FLAG           \
                  ,abc_prefix##FOG,              abc_prefix##FOG_COLOR,       abc_prefix##FOG_DENSITY         \
                  ,abc_prefix##FOG_END,          abc_prefix##FOG_HINT,        abc_prefix##FOG_INDEX           \
                  ,abc_prefix##FOG_MODE,         abc_prefix##FOG_START,       abc_prefix##FRONT_FACE          \
                  ,abc_prefix##GREEN_BIAS,       abc_prefix##GREEN_BITS,      abc_prefix##GREEN_SCALE         \
                  ,abc_prefix##INDEX_BITS,       abc_prefix##INDEX_CLEAR_VALUE                                \
                  ,abc_prefix##INDEX_MODE,       abc_prefix##INDEX_OFFSET,    abc_prefix##INDEX_SHIFT         \
                  ,abc_prefix##INDEX_WRITEMASK,  abc_prefix##LIGHTING,        abc_prefix##LIGHT_MODEL_AMBIENT           \
                  ,abc_prefix##LIGHT_MODEL_LOCAL_VIEWER,                      abc_prefix##LIGHT_MODEL_TWO_SIDE

#define GLConstants3(abc_prefix) \
                  ,abc_prefix##LINE_SMOOTH,      abc_prefix##LINE_SMOOTH_HINT,abc_prefix##LINE_STIPPLE        \
                  ,abc_prefix##LINE_STIPPLE_PATTERN,                          abc_prefix##LINE_STIPPLE_REPEAT           \
                  ,abc_prefix##LINE_WIDTH,       abc_prefix##LINE_WIDTH_GRANULARITY                           \
                  ,abc_prefix##LINE_WIDTH_RANGE, abc_prefix##LIST_BASE,       abc_prefix##LIST_INDEX          \
                  ,abc_prefix##LIST_MODE,        abc_prefix##LOGIC_OP,        abc_prefix##LOGIC_OP_MODE       \
                  ,abc_prefix##MAP1_COLOR_4,     abc_prefix##MAP1_GRID_DOMAIN,abc_prefix##MAP1_GRID_SEGMENTS  \
                  ,abc_prefix##MAP1_INDEX,       abc_prefix##MAP1_NORMAL,     abc_prefix##MAP1_TEXTURE_COORD_1          \
                  ,abc_prefix##MAP1_TEXTURE_COORD_2,                          abc_prefix##MAP1_TEXTURE_COORD_3          \
                  ,abc_prefix##MAP1_TEXTURE_COORD_4,                          abc_prefix##MAP1_VERTEX_3       \
                  ,abc_prefix##MAP1_VERTEX_4,    abc_prefix##MAP2_COLOR_4,    abc_prefix##MAP2_GRID_DOMAIN    \
                  ,abc_prefix##MAP2_GRID_SEGMENTS,                            abc_prefix##MAP2_INDEX          \
                  ,abc_prefix##MAP2_NORMAL,      abc_prefix##MAP2_TEXTURE_COORD_1                             \
                  ,abc_prefix##MAP2_TEXTURE_COORD_2,                          abc_prefix##MAP2_TEXTURE_COORD_3          \
                  ,abc_prefix##MAP2_TEXTURE_COORD_4,                          abc_prefix##MAP2_VERTEX_3       \
                  ,abc_prefix##MAP2_VERTEX_4,    abc_prefix##MAP_COLOR,       abc_prefix##MAP_STENCIL         \
                  ,abc_prefix##MATRIX_MODE,      abc_prefix##MAX_ATTRIB_STACK_DEPTH                           \
                  ,abc_prefix##MAX_CLIP_PLANES,  abc_prefix##MAX_EVAL_ORDER,  abc_prefix##MAX_LIGHTS          \
                  ,abc_prefix##MAX_LIST_NESTING, abc_prefix##MAX_MODELVIEW_STACK_DEPTH                        \
                  ,abc_prefix##MAX_NAME_STACK_DEPTH,                          abc_prefix##MAX_PIXEL_MAP_TABLE           \
                  ,abc_prefix##MAX_PROJECTION_STACK_DEPTH,                    abc_prefix##MAX_TEXTURE_SIZE    \
                  ,abc_prefix##MAX_TEXTURE_STACK_DEPTH,                       abc_prefix##MAX_VIEWPORT_DIMS

#define GLConstants4(abc_prefix) \
                  ,abc_prefix##MODELVIEW_MATRIX, abc_prefix##MODELVIEW_STACK_DEPTH                            \
                  ,abc_prefix##NAME_STACK_DEPTH, abc_prefix##NORMALIZE,       abc_prefix##PACK_ALIGNMENT      \
                  ,abc_prefix##PACK_LSB_FIRST,   abc_prefix##PACK_ROW_LENGTH, abc_prefix##PACK_SKIP_PIXELS    \
                  ,abc_prefix##PACK_SKIP_ROWS,   abc_prefix##PACK_SWAP_BYTES, abc_prefix##PERSPECTIVE_CORRECTION_HINT   \
                  ,abc_prefix##PIXEL_MAP_A_TO_A_SIZE,                         abc_prefix##PIXEL_MAP_B_TO_B_SIZE         \
                  ,abc_prefix##PIXEL_MAP_G_TO_G_SIZE,                         abc_prefix##PIXEL_MAP_I_TO_A_SIZE         \
                  ,abc_prefix##PIXEL_MAP_I_TO_B_SIZE,                         abc_prefix##PIXEL_MAP_I_TO_G_SIZE         \
                  ,abc_prefix##PIXEL_MAP_I_TO_I_SIZE,                         abc_prefix##PIXEL_MAP_I_TO_R_SIZE         \
                  ,abc_prefix##PIXEL_MAP_R_TO_R_SIZE,                         abc_prefix##PIXEL_MAP_S_TO_S_SIZE         \
                  ,abc_prefix##POINT_SIZE,       abc_prefix##POINT_SIZE_GRANULARITY                           \
                  ,abc_prefix##POINT_SIZE_RANGE, abc_prefix##POINT_SMOOTH,    abc_prefix##POINT_SMOOTH_HINT   \
                  ,abc_prefix##POLYGON_MODE,     abc_prefix##POLYGON_SMOOTH,  abc_prefix##POLYGON_SMOOTH_HINT           \
                  ,abc_prefix##POLYGON_STIPPLE,  abc_prefix##PROJECTION_MATRIX, abc_prefix##PROJECTION_STACK_DEPTH      \
                  ,abc_prefix##READ_BUFFER,      abc_prefix##RED_BIAS,        abc_prefix##RED_BITS            \
                  ,abc_prefix##RED_SCALE,        abc_prefix##RENDER_MODE,     abc_prefix##RGBA_MODE           \
                  ,abc_prefix##SCISSOR_BOX,      abc_prefix##SCISSOR_TEST,    abc_prefix##SHADE_MODEL         \
                  ,abc_prefix##STENCIL_BITS,     abc_prefix##STENCIL_CLEAR_VALUE                              \
                  ,abc_prefix##STENCIL_FAIL,     abc_prefix##STENCIL_FUNC,    abc_prefix##STENCIL_PASS_DEPTH_FAIL         \
                  ,abc_prefix##STENCIL_PASS_DEPTH_PASS,                       abc_prefix##STENCIL_REF         \
                  ,abc_prefix##STENCIL_TEST,     abc_prefix##STENCIL_VALUE_MASK,abc_prefix##STENCIL_WRITEMASK \
                  ,abc_prefix##STEREO,           abc_prefix##SUBPIXEL_BITS,   abc_prefix##TEXTURE_1D          \
                  ,abc_prefix##TEXTURE_2D,       abc_prefix##TEXTURE_ENV_COLOR,abc_prefix##TEXTURE_ENV_MODE   \
                  ,abc_prefix##TEXTURE_GEN_S,    abc_prefix##TEXTURE_GEN_T,   abc_prefix##TEXTURE_GEN_R       \
                  ,abc_prefix##TEXTURE_GEN_Q,    abc_prefix##TEXTURE_MATRIX,  abc_prefix##TEXTURE_STACK_DEPTH \
                  ,abc_prefix##UNPACK_ALIGNMENT, abc_prefix##UNPACK_LSB_FIRST,abc_prefix##UNPACK_ROW_LENGTH   \
                  ,abc_prefix##UNPACK_SKIP_PIXELS,abc_prefix##UNPACK_SKIP_ROWS,abc_prefix##UNPACK_SWAP_BYTES  \
                  ,abc_prefix##VIEWPORT,         abc_prefix##ZOOM_X,          abc_prefix##ZOOM_Y              \
                  ,abc_prefix##SMOOTH,           abc_prefix##FLAT

#define GLConstants5(abc_prefix) \
                  ,abc_prefix##POINTS, abc_prefix##LINES, abc_prefix##LINE_STRIP, abc_prefix##LINE_LOOP \
                  ,abc_prefix##TRIANGLES, abc_prefix##TRIANGLE_STRIP, abc_prefix##TRIANGLE_FAN \
                  ,abc_prefix##POLYGON

#define GLConstants6(abc_prefix) \
                  ,abc_prefix##DIFFUSE, abc_prefix##POSITION, abc_prefix##CLIP_PLANE0

#define GLConstants(abc_prefix) \
                             GLConstants1(abc_prefix) \
                             GLConstants2(abc_prefix) \
                             GLConstants3(abc_prefix) \
                             GLConstants4(abc_prefix) \
                             GLConstants5(abc_prefix) \
                             GLConstants6(abc_prefix)
#endif


enum EG3D2GLmode { GLConstants(k) };

#else
enum EG3D2GLmode {
                   kQUAD_STRIP, kQUADS, kCOMPILE, kCOMPILE_AND_EXECUTE
                  ,kLIGHT0    ,  kLIGHT1    ,  kLIGHT2    ,  kLIGHT3
                  ,kLIGHT4    ,  kLIGHT5    ,  kLIGHT6    ,  kLIGHT7
                  ,kCW        ,  kCCW
                  ,kFRONT     ,  kBACK      ,  kFRONT_AND_BACK
                  ,kPOINT     ,  kLINE      ,  kFILL      ,
                  ,kACCUM_ALPHA_BITS, kACCUM_BLUE_BITS, kACCUM_CLEAR_VALUE
                  ,kACCUM_GREEN_BITS, kACCUM_RED_BITS,  kALPHA_BIAS
                  ,kALPHA_BITS,       kALPHA_SCALE,     kALPHA_TEST
                  ,kALPHA_TEST_FUNC,  kALPHA_TEST_REF,  kATTRIB_STACK_DEPTH
                  ,kAUTO_NORMAL,      kAUX_BUFFERS,     kBLEND
                  ,kBLEND_DST,        kBLEND_SRC,       kBLUE_BIAS
                  ,kBLUE_BITS,        kBLUE_SCALE,      kCOLOR_CLEAR_VALUE
                  ,kCOLOR_MATERIAL,   kCOLOR_MATERIAL_FACE
                  ,kCOLOR_MATERIAL_PARAMETER,           kCOLOR_WRITEMASK
                  ,kCULL_FACE,        kCULL_FACE_MODE,  kCURRENT_COLOR
                  ,kCURRENT_INDEX,    kCURRENT_NORMAL,  kCURRENT_RASTER_COLOR
                  ,kCURRENT_RASTER_DISTANCE,            kCURRENT_RASTER_INDEX
                  ,kCURRENT_RASTER_POSITION,            kCURRENT_RASTER_TEXTURE_COORDS
                  ,kCURRENT_RASTER_POSITION_VALID,      kCURRENT_TEXTURE_COORDS
                  ,kDEPTH_BIAS,       kDEPTH_BITS,      kDEPTH_CLEAR_VALUE
                  ,kDEPTH_FUNC,       kDEPTH_RANGE,     kDEPTH_SCALE
                  ,kDEPTH_TEST,       kDEPTH_WRITEMASK, kDITHER
                  ,kDOUBLEBUFFER,     kDRAW_BUFFER,     kEDGE_FLAG
                  ,kFOG,              kFOG_COLOR,       kFOG_DENSITY
                  ,kFOG_END,          kFOG_HINT,        kFOG_INDEX
                  ,kFOG_MODE,         kFOG_START,       kFRONT_FACE
                  ,kGREEN_BIAS,       kGREEN_BITS,      kGREEN_SCALE
                  ,kINDEX_BITS,       kINDEX_CLEAR_VALUE
                  ,kINDEX_MODE,       kINDEX_OFFSET,    kINDEX_SHIFT
                  ,kINDEX_WRITEMASK,  kLIGHTING,        kLIGHT_MODEL_AMBIENT
                  ,kLIGHT_MODEL_LOCAL_VIEWER,           kLIGHT_MODEL_TWO_SIDE
                  ,kLINE_SMOOTH,      kLINE_SMOOTH_HINT,kLINE_STIPPLE
                  ,kLINE_STIPPLE_PATTERN,               kLINE_STIPPLE_REPEAT
                  ,kLINE_WIDTH,       kLINE_WIDTH_GRANULARITY
                  ,kLINE_WIDTH_RANGE, kLIST_BASE,       kLIST_INDEX
                  ,kLIST_MODE,        kLOGIC_OP,        kLOGIC_OP_MODE
                  ,kMAP1_COLOR_4,     kMAP1_GRID_DOMAIN,kMAP1_GRID_SEGMENTS
                  ,kMAP1_INDEX,       kMAP1_NORMAL,     kMAP1_TEXTURE_COORD_1
                  ,kMAP1_TEXTURE_COORD_2,               kMAP1_TEXTURE_COORD_3
                  ,kMAP1_TEXTURE_COORD_4,               kMAP1_VERTEX_3
                  ,kMAP1_VERTEX_4,    kMAP2_COLOR_4,    kMAP2_GRID_DOMAIN
                  ,kMAP2_GRID_SEGMENTS,                 kMAP2_INDEX
                  ,kMAP2_NORMAL,      kMAP2_TEXTURE_COORD_1
                  ,kMAP2_TEXTURE_COORD_2,               kMAP2_TEXTURE_COORD_3
                  ,kMAP2_TEXTURE_COORD_4,               kMAP2_VERTEX_3
                  ,kMAP2_VERTEX_4,    kMAP_COLOR,       kMAP_STENCIL
                  ,kMATRIX_MODE,      kMAX_ATTRIB_STACK_DEPTH
                  ,kMAX_CLIP_PLANES,  kMAX_EVAL_ORDER,  kMAX_LIGHTS
                  ,kMAX_LIST_NESTING, kMAX_MODELVIEW_STACK_DEPTH
                  ,kMAX_NAME_STACK_DEPTH,               kMAX_PIXEL_MAP_TABLE
                  ,kMAX_PROJECTION_STACK_DEPTH,         kMAX_TEXTURE_SIZE
                  ,kMAX_TEXTURE_STACK_DEPTH,            kMAX_VIEWPORT_DIMS
                  ,kMODELVIEW_MATRIX, kMODELVIEW_STACK_DEPTH
                  ,kNAME_STACK_DEPTH, kNORMALIZE,       kPACK_ALIGNMENT
                  ,kPACK_LSB_FIRST,   kPACK_ROW_LENGTH, kPACK_SKIP_PIXELS
                  ,kPACK_SKIP_ROWS,   kPACK_SWAP_BYTES, kPERSPECTIVE_CORRECTION_HINT
                  ,kPIXEL_MAP_A_TO_A_SIZE,              kPIXEL_MAP_B_TO_B_SIZE
                  ,kPIXEL_MAP_G_TO_G_SIZE,              kPIXEL_MAP_I_TO_A_SIZE
                  ,kPIXEL_MAP_I_TO_B_SIZE,              kPIXEL_MAP_I_TO_G_SIZE
                  ,kPIXEL_MAP_I_TO_I_SIZE,              kPIXEL_MAP_I_TO_R_SIZE
                  ,kPIXEL_MAP_R_TO_R_SIZE,              kPIXEL_MAP_S_TO_S_SIZE
                  ,kPOINT_SIZE,       kPOINT_SIZE_GRANULARITY
                  ,kPOINT_SIZE_RANGE, kPOINT_SMOOTH,    kPOINT_SMOOTH_HINT
                  ,kPOLYGON_MODE,     kPOLYGON_SMOOTH,  kPOLYGON_SMOOTH_HINT
                  ,kPOLYGON_STIPPLE,  kPROJECTION_MATRIX, kPROJECTION_STACK_DEPTH
                  ,kREAD_BUFFER,      kRED_BIAS,        kRED_BITS
                  ,kRED_SCALE,        kRENDER_MODE,     kRGBA_MODE
                  ,kSCISSOR_BOX,      kSCISSOR_TEST,    kSHADE_MODEL
                  ,kSTENCIL_BITS,     kSTENCIL_CLEAR_VALUE
                  ,kSTENCIL_FAIL,     kSTENCIL_FUNC,    kSTENCIL_PASS_DEPTH_FAIL
                  ,kSTENCIL_PASS_DEPTH_PASS,            kSTENCIL_REF
                  ,kSTENCIL_TEST,     kSTENCIL_VALUE_MASK,kSTENCIL_WRITEMASK
                  ,kSTEREO,           kSUBPIXEL_BITS,   kTEXTURE_1D
                  ,kTEXTURE_2D,       kTEXTURE_ENV_COLOR,kTEXTURE_ENV_MODE
                  ,kTEXTURE_GEN_S,    kTEXTURE_GEN_T,   kTEXTURE_GEN_R
                  ,kTEXTURE_GEN_Q,    kTEXTURE_MATRIX,  kTEXTURE_STACK_DEPTH
                  ,kUNPACK_ALIGNMENT, kUNPACK_LSB_FIRST,kUNPACK_ROW_LENGTH
                  ,kUNPACK_SKIP_PIXELS,kUNPACK_SKIP_ROWS,kUNPACK_SWAP_BYTES
                  ,kVIEWPORT,         kZOOM_X,          kZOOM_Y
                  ,kSMOOTH,           kFLAT
                 };
#endif

#endif
