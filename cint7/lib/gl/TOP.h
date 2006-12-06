/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/***********************************************************************
* lib/gl/TOP.h
*  header file to link openGL and cint.
*
*  Top level parameter information file for making cintopengl
***********************************************************************/

/***********************************************************************
* opengl include files are included in GL.h. Opengl include files
* are preprocessed by external C preprocessor. All of the macro 
* information is lost in this process. 
***********************************************************************/

#ifdef __MAKECINT__
#pragma preprocessor on
#endif

/* C preprocessor is used for GL.h before Cint reads it */
#include "cintGL.h"

#ifdef __MAKECINT__
#pragma preprocessor off
#endif

#define G__GLDLL_H

/*********************************************************************
 * undefined symbols
 *********************************************************************/
#ifdef __MAKECINT__
/* Following pragmas are needed to eliminate errors compiling dictionary.
 * They are needed to fix the mismatch between header and actual library. */
#pragma link off class GLUquadric;
#pragma link off class GLUnurbs;
#pragma link off class GLUtesselator;
#pragma link off class xmesa_context;
#pragma link off class xmesa_visual;
#pragma link off class xmesa_buffer;
#pragma link off function gluBuild3DMipmapLevels;
#pragma link off function gluBuild3DMipmaps;
#pragma link off function gluBuild2DMipmapLevels;
#pragma link off function gluBuild1DMipmapLevels;
#pragma link off function XMesaMakeCurrent2;

// Mandrake8.0 specific
#pragma link off function XMesaGetString;
#pragma link off function XMesaCreateVisual;
#pragma link off function XMesaUnbindContext;
#pragma link off function XMesaGetCurrentReadBuffer;
#pragma link off function  XMesaSwapBuffers;
#pragma link off function  XMesaDestroyContext;
#pragma link off function  XMesaGetBackBuffer;
#pragma link off function  glXGetCurrentDrawableEXT;
#pragma link off function  XMesaCreatePixmapBuffer;
#pragma link off function  XMesaGetDepthBuffer;
#pragma link off function   XMesaGarbageCollect;
#pragma link off function   XMesaDestroyBuffer;
#pragma link off function    XMesaCreateWindowBuffer;
#pragma link off function    XMesaGetCurrentBuffer;
#pragma link off function    XMesaCreateContext;
#pragma link off function    XMesaFlush;
#pragma link off function     XMesaCopySubBuffer;
#pragma link off function     XMesaDestroyVisual;
#pragma link off function     XMesaSetFXmode;
#pragma link off function     XMesaGetCurrentContext;
#pragma link off function     XMesaDitherColor;
#pragma link off function     XMesaFindBuffer ;
#pragma link off function     XMesaMakeCurrent ;

#ifdef G__WIN32
#pragma link off function  auxInitWindowW;
#pragma link off function  auxRGBImageLoadW;
#pragma link off function  auxDIBImageLoadW;
#pragma link off function  auxDrawStrW;
//#pragma link off global    auxRGBMap;
#endif

#endif


/*********************************************************************
 * define macros
 *********************************************************************/
#ifdef __MAKECINT__
/* Because preprocessor is used to process those header files,
 * macros must be defined here again for Cint. */

/* gl.h */
#define GL_VERSION_1_1   1
#define GL_VERSION_1_2   1
#define GL_FALSE				0x0
#define GL_TRUE					0x1
#define GL_BYTE					0x1400
#define GL_UNSIGNED_BYTE			0x1401
#define GL_SHORT				0x1402
#define GL_UNSIGNED_SHORT			0x1403
#define GL_INT					0x1404
#define GL_UNSIGNED_INT				0x1405
#define GL_FLOAT				0x1406
#define GL_DOUBLE				0x140A
#define GL_2_BYTES				0x1407
#define GL_3_BYTES				0x1408
#define GL_4_BYTES				0x1409
#define GL_POINTS				0x0000
#define GL_LINES				0x0001
#define GL_LINE_LOOP				0x0002
#define GL_LINE_STRIP				0x0003
#define GL_TRIANGLES				0x0004
#define GL_TRIANGLE_STRIP			0x0005
#define GL_TRIANGLE_FAN				0x0006
#define GL_QUADS				0x0007
#define GL_QUAD_STRIP				0x0008
#define GL_POLYGON				0x0009
#define GL_VERTEX_ARRAY				0x8074
#define GL_NORMAL_ARRAY				0x8075
#define GL_COLOR_ARRAY				0x8076
#define GL_INDEX_ARRAY				0x8077
#define GL_TEXTURE_COORD_ARRAY			0x8078
#define GL_EDGE_FLAG_ARRAY			0x8079
#define GL_VERTEX_ARRAY_SIZE			0x807A
#define GL_VERTEX_ARRAY_TYPE			0x807B
#define GL_VERTEX_ARRAY_STRIDE			0x807C
#define GL_NORMAL_ARRAY_TYPE			0x807E
#define GL_NORMAL_ARRAY_STRIDE			0x807F
#define GL_COLOR_ARRAY_SIZE			0x8081
#define GL_COLOR_ARRAY_TYPE			0x8082
#define GL_COLOR_ARRAY_STRIDE			0x8083
#define GL_INDEX_ARRAY_TYPE			0x8085
#define GL_INDEX_ARRAY_STRIDE			0x8086
#define GL_TEXTURE_COORD_ARRAY_SIZE		0x8088
#define GL_TEXTURE_COORD_ARRAY_TYPE		0x8089
#define GL_TEXTURE_COORD_ARRAY_STRIDE		0x808A
#define GL_EDGE_FLAG_ARRAY_STRIDE		0x808C
#define GL_VERTEX_ARRAY_POINTER			0x808E
#define GL_NORMAL_ARRAY_POINTER			0x808F
#define GL_COLOR_ARRAY_POINTER			0x8090
#define GL_INDEX_ARRAY_POINTER			0x8091
#define GL_TEXTURE_COORD_ARRAY_POINTER		0x8092
#define GL_EDGE_FLAG_ARRAY_POINTER		0x8093
#define GL_V2F					0x2A20
#define GL_V3F					0x2A21
#define GL_C4UB_V2F				0x2A22
#define GL_C4UB_V3F				0x2A23
#define GL_C3F_V3F				0x2A24
#define GL_N3F_V3F				0x2A25
#define GL_C4F_N3F_V3F				0x2A26
#define GL_T2F_V3F				0x2A27
#define GL_T4F_V4F				0x2A28
#define GL_T2F_C4UB_V3F				0x2A29
#define GL_T2F_C3F_V3F				0x2A2A
#define GL_T2F_N3F_V3F				0x2A2B
#define GL_T2F_C4F_N3F_V3F			0x2A2C
#define GL_T4F_C4F_N3F_V4F			0x2A2D
#define GL_MATRIX_MODE				0x0BA0
#define GL_MODELVIEW				0x1700
#define GL_PROJECTION				0x1701
#define GL_TEXTURE				0x1702
#define GL_POINT_SMOOTH				0x0B10
#define GL_POINT_SIZE				0x0B11
#define GL_POINT_SIZE_GRANULARITY 		0x0B13
#define GL_POINT_SIZE_RANGE			0x0B12
#define GL_LINE_SMOOTH				0x0B20
#define GL_LINE_STIPPLE				0x0B24
#define GL_LINE_STIPPLE_PATTERN			0x0B25
#define GL_LINE_STIPPLE_REPEAT			0x0B26
#define GL_LINE_WIDTH				0x0B21
#define GL_LINE_WIDTH_GRANULARITY		0x0B23
#define GL_LINE_WIDTH_RANGE			0x0B22
#define GL_POINT				0x1B00
#define GL_LINE					0x1B01
#define GL_FILL					0x1B02
#define GL_CW					0x0900
#define GL_CCW					0x0901
#define GL_FRONT				0x0404
#define GL_BACK					0x0405
#define GL_POLYGON_MODE				0x0B40
#define GL_POLYGON_SMOOTH			0x0B41
#define GL_POLYGON_STIPPLE			0x0B42
#define GL_EDGE_FLAG				0x0B43
#define GL_CULL_FACE				0x0B44
#define GL_CULL_FACE_MODE			0x0B45
#define GL_FRONT_FACE				0x0B46
#define GL_POLYGON_OFFSET_FACTOR		0x8038
#define GL_POLYGON_OFFSET_UNITS			0x2A00
#define GL_POLYGON_OFFSET_POINT			0x2A01
#define GL_POLYGON_OFFSET_LINE			0x2A02
#define GL_POLYGON_OFFSET_FILL			0x8037
#define GL_COMPILE				0x1300
#define GL_COMPILE_AND_EXECUTE			0x1301
#define GL_LIST_BASE				0x0B32
#define GL_LIST_INDEX				0x0B33
#define GL_LIST_MODE				0x0B30
#define GL_NEVER				0x0200
#define GL_LESS					0x0201
#define GL_EQUAL				0x0202
#define GL_LEQUAL				0x0203
#define GL_GREATER				0x0204
#define GL_NOTEQUAL				0x0205
#define GL_GEQUAL				0x0206
#define GL_ALWAYS				0x0207
#define GL_DEPTH_TEST				0x0B71
#define GL_DEPTH_BITS				0x0D56
#define GL_DEPTH_CLEAR_VALUE			0x0B73
#define GL_DEPTH_FUNC				0x0B74
#define GL_DEPTH_RANGE				0x0B70
#define GL_DEPTH_WRITEMASK			0x0B72
#define GL_DEPTH_COMPONENT			0x1902
#define GL_LIGHTING				0x0B50
#define GL_LIGHT0				0x4000
#define GL_LIGHT1				0x4001
#define GL_LIGHT2				0x4002
#define GL_LIGHT3				0x4003
#define GL_LIGHT4				0x4004
#define GL_LIGHT5				0x4005
#define GL_LIGHT6				0x4006
#define GL_LIGHT7				0x4007
#define GL_SPOT_EXPONENT			0x1205
#define GL_SPOT_CUTOFF				0x1206
#define GL_CONSTANT_ATTENUATION			0x1207
#define GL_LINEAR_ATTENUATION			0x1208
#define GL_QUADRATIC_ATTENUATION		0x1209
#define GL_AMBIENT				0x1200
#define GL_DIFFUSE				0x1201
#define GL_SPECULAR				0x1202
#define GL_SHININESS				0x1601
#define GL_EMISSION				0x1600
#define GL_POSITION				0x1203
#define GL_SPOT_DIRECTION			0x1204
#define GL_AMBIENT_AND_DIFFUSE			0x1602
#define GL_COLOR_INDEXES			0x1603
#define GL_LIGHT_MODEL_TWO_SIDE			0x0B52
#define GL_LIGHT_MODEL_LOCAL_VIEWER		0x0B51
#define GL_LIGHT_MODEL_AMBIENT			0x0B53
#define GL_FRONT_AND_BACK			0x0408
#define GL_SHADE_MODEL				0x0B54
#define GL_FLAT					0x1D00
#define GL_SMOOTH				0x1D01
#define GL_COLOR_MATERIAL			0x0B57
#define GL_COLOR_MATERIAL_FACE			0x0B55
#define GL_COLOR_MATERIAL_PARAMETER		0x0B56
#define GL_NORMALIZE				0x0BA1
#define GL_CLIP_PLANE0				0x3000
#define GL_CLIP_PLANE1				0x3001
#define GL_CLIP_PLANE2				0x3002
#define GL_CLIP_PLANE3				0x3003
#define GL_CLIP_PLANE4				0x3004
#define GL_CLIP_PLANE5				0x3005
#define GL_ACCUM_RED_BITS			0x0D58
#define GL_ACCUM_GREEN_BITS			0x0D59
#define GL_ACCUM_BLUE_BITS			0x0D5A
#define GL_ACCUM_ALPHA_BITS			0x0D5B
#define GL_ACCUM_CLEAR_VALUE			0x0B80
#define GL_ACCUM				0x0100
#define GL_ADD					0x0104
#define GL_LOAD					0x0101
#define GL_MULT					0x0103
#define GL_RETURN				0x0102
#define GL_ALPHA_TEST				0x0BC0
#define GL_ALPHA_TEST_REF			0x0BC2
#define GL_ALPHA_TEST_FUNC			0x0BC1
#define GL_BLEND				0x0BE2
#define GL_BLEND_SRC				0x0BE1
#define GL_BLEND_DST				0x0BE0
#define GL_ZERO					0x0
#define GL_ONE					0x1
#define GL_SRC_COLOR				0x0300
#define GL_ONE_MINUS_SRC_COLOR			0x0301
#define GL_DST_COLOR				0x0306
#define GL_ONE_MINUS_DST_COLOR			0x0307
#define GL_SRC_ALPHA				0x0302
#define GL_ONE_MINUS_SRC_ALPHA			0x0303
#define GL_DST_ALPHA				0x0304
#define GL_ONE_MINUS_DST_ALPHA			0x0305
#define GL_SRC_ALPHA_SATURATE			0x0308
#define GL_CONSTANT_COLOR			0x8001
#define GL_ONE_MINUS_CONSTANT_COLOR		0x8002
#define GL_CONSTANT_ALPHA			0x8003
#define GL_ONE_MINUS_CONSTANT_ALPHA		0x8004
#define GL_FEEDBACK				0x1C01
#define GL_RENDER				0x1C00
#define GL_SELECT				0x1C02
#define GL_2D					0x0600
#define GL_3D					0x0601
#define GL_3D_COLOR				0x0602
#define GL_3D_COLOR_TEXTURE			0x0603
#define GL_4D_COLOR_TEXTURE			0x0604
#define GL_POINT_TOKEN				0x0701
#define GL_LINE_TOKEN				0x0702
#define GL_LINE_RESET_TOKEN			0x0707
#define GL_POLYGON_TOKEN			0x0703
#define GL_BITMAP_TOKEN				0x0704
#define GL_DRAW_PIXEL_TOKEN			0x0705
#define GL_COPY_PIXEL_TOKEN			0x0706
#define GL_PASS_THROUGH_TOKEN			0x0700
#define GL_FEEDBACK_BUFFER_POINTER		0x0DF0
#define GL_FEEDBACK_BUFFER_SIZE			0x0DF1
#define GL_FEEDBACK_BUFFER_TYPE			0x0DF2
#define GL_SELECTION_BUFFER_POINTER		0x0DF3
#define GL_SELECTION_BUFFER_SIZE		0x0DF4
#define GL_FOG					0x0B60
#define GL_FOG_MODE				0x0B65
#define GL_FOG_DENSITY				0x0B62
#define GL_FOG_COLOR				0x0B66
#define GL_FOG_INDEX				0x0B61
#define GL_FOG_START				0x0B63
#define GL_FOG_END				0x0B64
#define GL_LINEAR				0x2601
#define GL_EXP					0x0800
#define GL_EXP2					0x0801
#define GL_LOGIC_OP				0x0BF1
#define GL_INDEX_LOGIC_OP			0x0BF1
#define GL_COLOR_LOGIC_OP			0x0BF2
#define GL_LOGIC_OP_MODE			0x0BF0
#define GL_CLEAR				0x1500
#define GL_SET					0x150F
#define GL_COPY					0x1503
#define GL_COPY_INVERTED			0x150C
#define GL_NOOP					0x1505
#define GL_INVERT				0x150A
#define GL_AND					0x1501
#define GL_NAND					0x150E
#define GL_OR					0x1507
#define GL_NOR					0x1508
#define GL_XOR					0x1506
#define GL_EQUIV				0x1509
#define GL_AND_REVERSE				0x1502
#define GL_AND_INVERTED				0x1504
#define GL_OR_REVERSE				0x150B
#define GL_OR_INVERTED				0x150D
#define GL_STENCIL_TEST				0x0B90
#define GL_STENCIL_WRITEMASK			0x0B98
#define GL_STENCIL_BITS				0x0D57
#define GL_STENCIL_FUNC				0x0B92
#define GL_STENCIL_VALUE_MASK			0x0B93
#define GL_STENCIL_REF				0x0B97
#define GL_STENCIL_FAIL				0x0B94
#define GL_STENCIL_PASS_DEPTH_PASS		0x0B96
#define GL_STENCIL_PASS_DEPTH_FAIL		0x0B95
#define GL_STENCIL_CLEAR_VALUE			0x0B91
#define GL_STENCIL_INDEX			0x1901
#define GL_KEEP					0x1E00
#define GL_REPLACE				0x1E01
#define GL_INCR					0x1E02
#define GL_DECR					0x1E03
#define GL_NONE					0x0
#define GL_LEFT					0x0406
#define GL_RIGHT				0x0407
#define GL_FRONT_LEFT				0x0400
#define GL_FRONT_RIGHT				0x0401
#define GL_BACK_LEFT				0x0402
#define GL_BACK_RIGHT				0x0403
#define GL_AUX0					0x0409
#define GL_AUX1					0x040A
#define GL_AUX2					0x040B
#define GL_AUX3					0x040C
#define GL_COLOR_INDEX				0x1900
#define GL_RED					0x1903
#define GL_GREEN				0x1904
#define GL_BLUE					0x1905
#define GL_ALPHA				0x1906
#define GL_LUMINANCE				0x1909
#define GL_LUMINANCE_ALPHA			0x190A
#define GL_ALPHA_BITS				0x0D55
#define GL_RED_BITS				0x0D52
#define GL_GREEN_BITS				0x0D53
#define GL_BLUE_BITS				0x0D54
#define GL_INDEX_BITS				0x0D51
#define GL_SUBPIXEL_BITS			0x0D50
#define GL_AUX_BUFFERS				0x0C00
#define GL_READ_BUFFER				0x0C02
#define GL_DRAW_BUFFER				0x0C01
#define GL_DOUBLEBUFFER				0x0C32
#define GL_STEREO				0x0C33
#define GL_BITMAP				0x1A00
#define GL_COLOR				0x1800
#define GL_DEPTH				0x1801
#define GL_STENCIL				0x1802
#define GL_DITHER				0x0BD0
#define GL_RGB					0x1907
#define GL_RGBA					0x1908
#define GL_MAX_LIST_NESTING			0x0B31
#define GL_MAX_ATTRIB_STACK_DEPTH		0x0D35
#define GL_MAX_MODELVIEW_STACK_DEPTH		0x0D36
#define GL_MAX_NAME_STACK_DEPTH			0x0D37
#define GL_MAX_PROJECTION_STACK_DEPTH		0x0D38
#define GL_MAX_TEXTURE_STACK_DEPTH		0x0D39
#define GL_MAX_EVAL_ORDER			0x0D30
#define GL_MAX_LIGHTS				0x0D31
#define GL_MAX_CLIP_PLANES			0x0D32
#define GL_MAX_TEXTURE_SIZE			0x0D33
#define GL_MAX_PIXEL_MAP_TABLE			0x0D34
#define GL_MAX_VIEWPORT_DIMS			0x0D3A
#define GL_MAX_CLIENT_ATTRIB_STACK_DEPTH	0x0D3B
#define GL_ATTRIB_STACK_DEPTH			0x0BB0
#define GL_CLIENT_ATTRIB_STACK_DEPTH		0x0BB1
#define GL_COLOR_CLEAR_VALUE			0x0C22
#define GL_COLOR_WRITEMASK			0x0C23
#define GL_CURRENT_INDEX			0x0B01
#define GL_CURRENT_COLOR			0x0B00
#define GL_CURRENT_NORMAL			0x0B02
#define GL_CURRENT_RASTER_COLOR			0x0B04
#define GL_CURRENT_RASTER_DISTANCE		0x0B09
#define GL_CURRENT_RASTER_INDEX			0x0B05
#define GL_CURRENT_RASTER_POSITION		0x0B07
#define GL_CURRENT_RASTER_TEXTURE_COORDS	0x0B06
#define GL_CURRENT_RASTER_POSITION_VALID	0x0B08
#define GL_CURRENT_TEXTURE_COORDS		0x0B03
#define GL_INDEX_CLEAR_VALUE			0x0C20
#define GL_INDEX_MODE				0x0C30
#define GL_INDEX_WRITEMASK			0x0C21
#define GL_MODELVIEW_MATRIX			0x0BA6
#define GL_MODELVIEW_STACK_DEPTH		0x0BA3
#define GL_NAME_STACK_DEPTH			0x0D70
#define GL_PROJECTION_MATRIX			0x0BA7
#define GL_PROJECTION_STACK_DEPTH		0x0BA4
#define GL_RENDER_MODE				0x0C40
#define GL_RGBA_MODE				0x0C31
#define GL_TEXTURE_MATRIX			0x0BA8
#define GL_TEXTURE_STACK_DEPTH			0x0BA5
#define GL_VIEWPORT				0x0BA2
#define GL_AUTO_NORMAL				0x0D80
#define GL_MAP1_COLOR_4				0x0D90
#define GL_MAP1_GRID_DOMAIN			0x0DD0
#define GL_MAP1_GRID_SEGMENTS			0x0DD1
#define GL_MAP1_INDEX				0x0D91
#define GL_MAP1_NORMAL				0x0D92
#define GL_MAP1_TEXTURE_COORD_1			0x0D93
#define GL_MAP1_TEXTURE_COORD_2			0x0D94
#define GL_MAP1_TEXTURE_COORD_3			0x0D95
#define GL_MAP1_TEXTURE_COORD_4			0x0D96
#define GL_MAP1_VERTEX_3			0x0D97
#define GL_MAP1_VERTEX_4			0x0D98
#define GL_MAP2_COLOR_4				0x0DB0
#define GL_MAP2_GRID_DOMAIN			0x0DD2
#define GL_MAP2_GRID_SEGMENTS			0x0DD3
#define GL_MAP2_INDEX				0x0DB1
#define GL_MAP2_NORMAL				0x0DB2
#define GL_MAP2_TEXTURE_COORD_1			0x0DB3
#define GL_MAP2_TEXTURE_COORD_2			0x0DB4
#define GL_MAP2_TEXTURE_COORD_3			0x0DB5
#define GL_MAP2_TEXTURE_COORD_4			0x0DB6
#define GL_MAP2_VERTEX_3			0x0DB7
#define GL_MAP2_VERTEX_4			0x0DB8
#define GL_COEFF				0x0A00
#define GL_DOMAIN				0x0A02
#define GL_ORDER				0x0A01
#define GL_FOG_HINT				0x0C54
#define GL_LINE_SMOOTH_HINT			0x0C52
#define GL_PERSPECTIVE_CORRECTION_HINT		0x0C50
#define GL_POINT_SMOOTH_HINT			0x0C51
#define GL_POLYGON_SMOOTH_HINT			0x0C53
#define GL_DONT_CARE				0x1100
#define GL_FASTEST				0x1101
#define GL_NICEST				0x1102
#define GL_SCISSOR_TEST				0x0C11
#define GL_SCISSOR_BOX				0x0C10
#define GL_MAP_COLOR				0x0D10
#define GL_MAP_STENCIL				0x0D11
#define GL_INDEX_SHIFT				0x0D12
#define GL_INDEX_OFFSET				0x0D13
#define GL_RED_SCALE				0x0D14
#define GL_RED_BIAS				0x0D15
#define GL_GREEN_SCALE				0x0D18
#define GL_GREEN_BIAS				0x0D19
#define GL_BLUE_SCALE				0x0D1A
#define GL_BLUE_BIAS				0x0D1B
#define GL_ALPHA_SCALE				0x0D1C
#define GL_ALPHA_BIAS				0x0D1D
#define GL_DEPTH_SCALE				0x0D1E
#define GL_DEPTH_BIAS				0x0D1F
#define GL_PIXEL_MAP_S_TO_S_SIZE		0x0CB1
#define GL_PIXEL_MAP_I_TO_I_SIZE		0x0CB0
#define GL_PIXEL_MAP_I_TO_R_SIZE		0x0CB2
#define GL_PIXEL_MAP_I_TO_G_SIZE		0x0CB3
#define GL_PIXEL_MAP_I_TO_B_SIZE		0x0CB4
#define GL_PIXEL_MAP_I_TO_A_SIZE		0x0CB5
#define GL_PIXEL_MAP_R_TO_R_SIZE		0x0CB6
#define GL_PIXEL_MAP_G_TO_G_SIZE		0x0CB7
#define GL_PIXEL_MAP_B_TO_B_SIZE		0x0CB8
#define GL_PIXEL_MAP_A_TO_A_SIZE		0x0CB9
#define GL_PIXEL_MAP_S_TO_S			0x0C71
#define GL_PIXEL_MAP_I_TO_I			0x0C70
#define GL_PIXEL_MAP_I_TO_R			0x0C72
#define GL_PIXEL_MAP_I_TO_G			0x0C73
#define GL_PIXEL_MAP_I_TO_B			0x0C74
#define GL_PIXEL_MAP_I_TO_A			0x0C75
#define GL_PIXEL_MAP_R_TO_R			0x0C76
#define GL_PIXEL_MAP_G_TO_G			0x0C77
#define GL_PIXEL_MAP_B_TO_B			0x0C78
#define GL_PIXEL_MAP_A_TO_A			0x0C79
#define GL_PACK_ALIGNMENT			0x0D05
#define GL_PACK_LSB_FIRST			0x0D01
#define GL_PACK_ROW_LENGTH			0x0D02
#define GL_PACK_SKIP_PIXELS			0x0D04
#define GL_PACK_SKIP_ROWS			0x0D03
#define GL_PACK_SWAP_BYTES			0x0D00
#define GL_UNPACK_ALIGNMENT			0x0CF5
#define GL_UNPACK_LSB_FIRST			0x0CF1
#define GL_UNPACK_ROW_LENGTH			0x0CF2
#define GL_UNPACK_SKIP_PIXELS			0x0CF4
#define GL_UNPACK_SKIP_ROWS			0x0CF3
#define GL_UNPACK_SWAP_BYTES			0x0CF0
#define GL_ZOOM_X				0x0D16
#define GL_ZOOM_Y				0x0D17
#define GL_TEXTURE_ENV				0x2300
#define GL_TEXTURE_ENV_MODE			0x2200
#define GL_TEXTURE_1D				0x0DE0
#define GL_TEXTURE_2D				0x0DE1
#define GL_TEXTURE_WRAP_S			0x2802
#define GL_TEXTURE_WRAP_T			0x2803
#define GL_TEXTURE_MAG_FILTER			0x2800
#define GL_TEXTURE_MIN_FILTER			0x2801
#define GL_TEXTURE_ENV_COLOR			0x2201
#define GL_TEXTURE_GEN_S			0x0C60
#define GL_TEXTURE_GEN_T			0x0C61
#define GL_TEXTURE_GEN_MODE			0x2500
#define GL_TEXTURE_BORDER_COLOR			0x1004
#define GL_TEXTURE_WIDTH			0x1000
#define GL_TEXTURE_HEIGHT			0x1001
#define GL_TEXTURE_BORDER			0x1005
#define GL_TEXTURE_COMPONENTS			0x1003
#define GL_TEXTURE_RED_SIZE			0x805C
#define GL_TEXTURE_GREEN_SIZE			0x805D
#define GL_TEXTURE_BLUE_SIZE			0x805E
#define GL_TEXTURE_ALPHA_SIZE			0x805F
#define GL_TEXTURE_LUMINANCE_SIZE		0x8060
#define GL_TEXTURE_INTENSITY_SIZE		0x8061
#define GL_NEAREST_MIPMAP_NEAREST		0x2700
#define GL_NEAREST_MIPMAP_LINEAR		0x2702
#define GL_LINEAR_MIPMAP_NEAREST		0x2701
#define GL_LINEAR_MIPMAP_LINEAR			0x2703
#define GL_OBJECT_LINEAR			0x2401
#define GL_OBJECT_PLANE				0x2501
#define GL_EYE_LINEAR				0x2400
#define GL_EYE_PLANE				0x2502
#define GL_SPHERE_MAP				0x2402
#define GL_DECAL				0x2101
#define GL_MODULATE				0x2100
#define GL_NEAREST				0x2600
#define GL_REPEAT				0x2901
#define GL_CLAMP				0x2900
#define GL_S					0x2000
#define GL_T					0x2001
#define GL_R					0x2002
#define GL_Q					0x2003
#define GL_TEXTURE_GEN_R			0x0C62
#define GL_TEXTURE_GEN_Q			0x0C63
#define GL_PROXY_TEXTURE_1D			0x8063
#define GL_PROXY_TEXTURE_2D			0x8064
#define GL_TEXTURE_PRIORITY			0x8066
#define GL_TEXTURE_RESIDENT			0x8067
#define GL_TEXTURE_BINDING_1D			0x8068
#define GL_TEXTURE_BINDING_2D			0x8069
#define GL_TEXTURE_INTERNAL_FORMAT		0x1003
#define GL_PACK_SKIP_IMAGES			0x806B
#define GL_PACK_IMAGE_HEIGHT			0x806C
#define GL_UNPACK_SKIP_IMAGES			0x806D
#define GL_UNPACK_IMAGE_HEIGHT			0x806E
#define GL_TEXTURE_3D				0x806F
#define GL_PROXY_TEXTURE_3D			0x8070
#define GL_TEXTURE_DEPTH			0x8071
#define GL_TEXTURE_WRAP_R			0x8072
#define GL_MAX_3D_TEXTURE_SIZE			0x8073
#define GL_TEXTURE_BINDING_3D			0x806A
#define GL_ALPHA4				0x803B
#define GL_ALPHA8				0x803C
#define GL_ALPHA12				0x803D
#define GL_ALPHA16				0x803E
#define GL_LUMINANCE4				0x803F
#define GL_LUMINANCE8				0x8040
#define GL_LUMINANCE12				0x8041
#define GL_LUMINANCE16				0x8042
#define GL_LUMINANCE4_ALPHA4			0x8043
#define GL_LUMINANCE6_ALPHA2			0x8044
#define GL_LUMINANCE8_ALPHA8			0x8045
#define GL_LUMINANCE12_ALPHA4			0x8046
#define GL_LUMINANCE12_ALPHA12			0x8047
#define GL_LUMINANCE16_ALPHA16			0x8048
#define GL_INTENSITY				0x8049
#define GL_INTENSITY4				0x804A
#define GL_INTENSITY8				0x804B
#define GL_INTENSITY12				0x804C
#define GL_INTENSITY16				0x804D
#define GL_R3_G3_B2				0x2A10
#define GL_RGB4					0x804F
#define GL_RGB5					0x8050
#define GL_RGB8					0x8051
#define GL_RGB10				0x8052
#define GL_RGB12				0x8053
#define GL_RGB16				0x8054
#define GL_RGBA2				0x8055
#define GL_RGBA4				0x8056
#define GL_RGB5_A1				0x8057
#define GL_RGBA8				0x8058
#define GL_RGB10_A2				0x8059
#define GL_RGBA12				0x805A
#define GL_RGBA16				0x805B
#define GL_VENDOR				0x1F00
#define GL_RENDERER				0x1F01
#define GL_VERSION				0x1F02
#define GL_EXTENSIONS				0x1F03
#define GL_NO_ERROR 				0x0
#define GL_INVALID_VALUE			0x0501
#define GL_INVALID_ENUM				0x0500
#define GL_INVALID_OPERATION			0x0502
#define GL_STACK_OVERFLOW			0x0503
#define GL_STACK_UNDERFLOW			0x0504
#define GL_OUT_OF_MEMORY			0x0505
#define GL_RESCALE_NORMAL			0x803A
#define GL_CLAMP_TO_EDGE			0x812F
#define GL_MAX_ELEMENTS_VERTICES		0x80E8
#define GL_MAX_ELEMENTS_INDICES			0x80E9
#define GL_BGR					0x80E0
#define GL_BGRA					0x80E1
#define GL_UNSIGNED_BYTE_3_3_2			0x8032
#define GL_UNSIGNED_BYTE_2_3_3_REV		0x8362
#define GL_UNSIGNED_SHORT_5_6_5			0x8363
#define GL_UNSIGNED_SHORT_5_6_5_REV		0x8364
#define GL_UNSIGNED_SHORT_4_4_4_4		0x8033
#define GL_UNSIGNED_SHORT_4_4_4_4_REV		0x8365
#define GL_UNSIGNED_SHORT_5_5_5_1		0x8034
#define GL_UNSIGNED_SHORT_1_5_5_5_REV		0x8366
#define GL_UNSIGNED_INT_8_8_8_8			0x8035
#define GL_UNSIGNED_INT_8_8_8_8_REV		0x8367
#define GL_UNSIGNED_INT_10_10_10_2		0x8036
#define GL_UNSIGNED_INT_2_10_10_10_REV		0x8368
#define GL_LIGHT_MODEL_COLOR_CONTROL		0x81F8
#define GL_SINGLE_COLOR				0x81F9
#define GL_SEPARATE_SPECULAR_COLOR		0x81FA
#define GL_TEXTURE_MIN_LOD			0x813A
#define GL_TEXTURE_MAX_LOD			0x813B
#define GL_TEXTURE_BASE_LEVEL			0x813C
#define GL_TEXTURE_MAX_LEVEL			0x813D
#define GL_COLOR_TABLE				0x80D0
#define GL_POST_CONVOLUTION_COLOR_TABLE		0x80D1
#define GL_POST_COLOR_MATRIX_COLOR_TABLE	0x80D2
#define GL_PROXY_COLOR_TABLE			0x80D3
#define GL_PROXY_POST_CONVOLUTION_COLOR_TABLE	0x80D4
#define GL_PROXY_POST_COLOR_MATRIX_COLOR_TABLE	0x80D5
#define GL_COLOR_TABLE_SCALE			0x80D6
#define GL_COLOR_TABLE_BIAS			0x80D7
#define GL_COLOR_TABLE_FORMAT			0x80D8
#define GL_COLOR_TABLE_WIDTH			0x80D9
#define GL_COLOR_TABLE_RED_SIZE			0x80DA
#define GL_COLOR_TABLE_GREEN_SIZE		0x80DB
#define GL_COLOR_TABLE_BLUE_SIZE		0x80DC
#define GL_COLOR_TABLE_ALPHA_SIZE		0x80DD
#define GL_COLOR_TABLE_LUMINANCE_SIZE		0x80DE
#define GL_COLOR_TABLE_INTENSITY_SIZE		0x80DF
#define GL_CONVOLUTION_1D			0x8010
#define GL_CONVOLUTION_2D			0x8011
#define GL_SEPARABLE_2D				0x8012
#define GL_CONVOLUTION_BORDER_MODE		0x8013
#define GL_CONVOLUTION_FILTER_SCALE		0x8014
#define GL_CONVOLUTION_FILTER_BIAS		0x8015
#define GL_REDUCE				0x8016
#define GL_CONVOLUTION_FORMAT			0x8017
#define GL_CONVOLUTION_WIDTH			0x8018
#define GL_CONVOLUTION_HEIGHT			0x8019
#define GL_MAX_CONVOLUTION_WIDTH		0x801A
#define GL_MAX_CONVOLUTION_HEIGHT		0x801B
#define GL_POST_CONVOLUTION_RED_SCALE		0x801C
#define GL_POST_CONVOLUTION_GREEN_SCALE		0x801D
#define GL_POST_CONVOLUTION_BLUE_SCALE		0x801E
#define GL_POST_CONVOLUTION_ALPHA_SCALE		0x801F
#define GL_POST_CONVOLUTION_RED_BIAS		0x8020
#define GL_POST_CONVOLUTION_GREEN_BIAS		0x8021
#define GL_POST_CONVOLUTION_BLUE_BIAS		0x8022
#define GL_POST_CONVOLUTION_ALPHA_BIAS		0x8023
#define GL_CONSTANT_BORDER			0x8151
#define GL_REPLICATE_BORDER			0x8153
#define GL_CONVOLUTION_BORDER_COLOR		0x8154
#define GL_COLOR_MATRIX				0x80B1
#define GL_COLOR_MATRIX_STACK_DEPTH		0x80B2
#define GL_MAX_COLOR_MATRIX_STACK_DEPTH		0x80B3
#define GL_POST_COLOR_MATRIX_RED_SCALE		0x80B4
#define GL_POST_COLOR_MATRIX_GREEN_SCALE	0x80B5
#define GL_POST_COLOR_MATRIX_BLUE_SCALE		0x80B6
#define GL_POST_COLOR_MATRIX_ALPHA_SCALE	0x80B7
#define GL_POST_COLOR_MATRIX_RED_BIAS		0x80B8
#define GL_POST_COLOR_MATRIX_GREEN_BIAS		0x80B9
#define GL_POST_COLOR_MATRIX_BLUE_BIAS		0x80BA
#define GL_POST_COLOR_MATRIX_ALPHA_BIAS		0x80BB
#define GL_HISTOGRAM				0x8024
#define GL_PROXY_HISTOGRAM			0x8025
#define GL_HISTOGRAM_WIDTH			0x8026
#define GL_HISTOGRAM_FORMAT			0x8027
#define GL_HISTOGRAM_RED_SIZE			0x8028
#define GL_HISTOGRAM_GREEN_SIZE			0x8029
#define GL_HISTOGRAM_BLUE_SIZE			0x802A
#define GL_HISTOGRAM_ALPHA_SIZE			0x802B
#define GL_HISTOGRAM_LUMINANCE_SIZE		0x802C
#define GL_HISTOGRAM_SINK			0x802D
#define GL_MINMAX				0x802E
#define GL_MINMAX_FORMAT			0x802F
#define GL_MINMAX_SINK				0x8030
#define GL_TABLE_TOO_LARGE			0x8031
#define GL_BLEND_EQUATION			0x8009
#define GL_MIN					0x8007
#define GL_MAX					0x8008
#define GL_FUNC_ADD				0x8006
#define GL_FUNC_SUBTRACT			0x800A
#define GL_FUNC_REVERSE_SUBTRACT		0x800B
#define	GL_BLEND_COLOR				0x8005
#define GL_CURRENT_BIT				0x00000001
#define GL_POINT_BIT				0x00000002
#define GL_LINE_BIT				0x00000004
#define GL_POLYGON_BIT				0x00000008
#define GL_POLYGON_STIPPLE_BIT			0x00000010
#define GL_PIXEL_MODE_BIT			0x00000020
#define GL_LIGHTING_BIT				0x00000040
#define GL_FOG_BIT				0x00000080
#define GL_DEPTH_BUFFER_BIT			0x00000100
#define GL_ACCUM_BUFFER_BIT			0x00000200
#define GL_STENCIL_BUFFER_BIT			0x00000400
#define GL_VIEWPORT_BIT				0x00000800
#define GL_TRANSFORM_BIT			0x00001000
#define GL_ENABLE_BIT				0x00002000
#define GL_COLOR_BUFFER_BIT			0x00004000
#define GL_HINT_BIT				0x00008000
#define GL_EVAL_BIT				0x00010000
#define GL_LIST_BIT				0x00020000
#define GL_TEXTURE_BIT				0x00040000
#define GL_SCISSOR_BIT				0x00080000
#define GL_ALL_ATTRIB_BITS			0x000FFFFF
#define GL_CLIENT_PIXEL_STORE_BIT		0x00000001
#define GL_CLIENT_VERTEX_ARRAY_BIT		0x00000002
#define GL_CLIENT_ALL_ATTRIB_BITS 		0xFFFFFFFF
#define GL_EXT_abgr 1
#define GL_ABGR_EXT				0x8000
#define GL_EXT_blend_color 1
#define GL_CONSTANT_COLOR_EXT			0x8001
#define GL_ONE_MINUS_CONSTANT_COLOR_EXT		0x8002
#define GL_CONSTANT_ALPHA_EXT			0x8003
#define GL_ONE_MINUS_CONSTANT_ALPHA_EXT		0x8004
#define GL_BLEND_COLOR_EXT			0x8005
#define GL_EXT_polygon_offset 1
#define GL_POLYGON_OFFSET_EXT			0x8037
#define GL_POLYGON_OFFSET_FACTOR_EXT		0x8038
#define GL_POLYGON_OFFSET_BIAS_EXT		0x8039
#define GL_EXT_texture3D 1
#define GL_PACK_SKIP_IMAGES_EXT			0x806B
#define GL_PACK_IMAGE_HEIGHT_EXT		0x806C
#define GL_UNPACK_SKIP_IMAGES_EXT		0x806D
#define GL_UNPACK_IMAGE_HEIGHT_EXT		0x806E
#define GL_TEXTURE_3D_EXT			0x806F
#define GL_PROXY_TEXTURE_3D_EXT			0x8070
#define GL_TEXTURE_DEPTH_EXT			0x8071
#define GL_TEXTURE_WRAP_R_EXT			0x8072
#define GL_MAX_3D_TEXTURE_SIZE_EXT		0x8073
#define GL_TEXTURE_3D_BINDING_EXT		0x806A
#define GL_EXT_texture_object 1
#define GL_TEXTURE_PRIORITY_EXT			0x8066
#define GL_TEXTURE_RESIDENT_EXT			0x8067
#define GL_TEXTURE_1D_BINDING_EXT		0x8068
#define GL_TEXTURE_2D_BINDING_EXT		0x8069
#define GL_EXT_rescale_normal 1
#define GL_RESCALE_NORMAL_EXT			0x803A
#define GL_EXT_vertex_array 1
#define GL_VERTEX_ARRAY_EXT			0x8074
#define GL_NORMAL_ARRAY_EXT			0x8075
#define GL_COLOR_ARRAY_EXT			0x8076
#define GL_INDEX_ARRAY_EXT			0x8077
#define GL_TEXTURE_COORD_ARRAY_EXT		0x8078
#define GL_EDGE_FLAG_ARRAY_EXT			0x8079
#define GL_VERTEX_ARRAY_SIZE_EXT		0x807A
#define GL_VERTEX_ARRAY_TYPE_EXT		0x807B
#define GL_VERTEX_ARRAY_STRIDE_EXT		0x807C
#define GL_VERTEX_ARRAY_COUNT_EXT		0x807D
#define GL_NORMAL_ARRAY_TYPE_EXT		0x807E
#define GL_NORMAL_ARRAY_STRIDE_EXT		0x807F
#define GL_NORMAL_ARRAY_COUNT_EXT		0x8080
#define GL_COLOR_ARRAY_SIZE_EXT			0x8081
#define GL_COLOR_ARRAY_TYPE_EXT			0x8082
#define GL_COLOR_ARRAY_STRIDE_EXT		0x8083
#define GL_COLOR_ARRAY_COUNT_EXT		0x8084
#define GL_INDEX_ARRAY_TYPE_EXT			0x8085
#define GL_INDEX_ARRAY_STRIDE_EXT		0x8086
#define GL_INDEX_ARRAY_COUNT_EXT		0x8087
#define GL_TEXTURE_COORD_ARRAY_SIZE_EXT		0x8088
#define GL_TEXTURE_COORD_ARRAY_TYPE_EXT		0x8089
#define GL_TEXTURE_COORD_ARRAY_STRIDE_EXT	0x808A
#define GL_TEXTURE_COORD_ARRAY_COUNT_EXT	0x808B
#define GL_EDGE_FLAG_ARRAY_STRIDE_EXT		0x808C
#define GL_EDGE_FLAG_ARRAY_COUNT_EXT		0x808D
#define GL_VERTEX_ARRAY_POINTER_EXT		0x808E
#define GL_NORMAL_ARRAY_POINTER_EXT		0x808F
#define GL_COLOR_ARRAY_POINTER_EXT		0x8090
#define GL_INDEX_ARRAY_POINTER_EXT		0x8091
#define GL_TEXTURE_COORD_ARRAY_POINTER_EXT	0x8092
#define GL_EDGE_FLAG_ARRAY_POINTER_EXT		0x8093
#define GL_SGIS_texture_edge_clamp 1
#define GL_CLAMP_TO_EDGE_SGIS			0x812F
#define GL_EXT_blend_minmax 1
#define GL_FUNC_ADD_EXT				0x8006
#define GL_MIN_EXT				0x8007
#define GL_MAX_EXT				0x8008
#define GL_BLEND_EQUATION_EXT			0x8009
#define GL_EXT_blend_subtract 1
#define GL_FUNC_SUBTRACT_EXT			0x800A
#define GL_FUNC_REVERSE_SUBTRACT_EXT		0x800B
#define GL_EXT_blend_logic_op 1
#define GL_EXT_point_parameters 1
#define GL_POINT_SIZE_MIN_EXT			0x8126
#define GL_POINT_SIZE_MAX_EXT			0x8127
#define GL_POINT_FADE_THRESHOLD_SIZE_EXT	0x8128
#define GL_DISTANCE_ATTENUATION_EXT		0x8129
#define GL_PGI_misc_hints 1
#define GL_PREFER_DOUBLEBUFFER_HINT_PGI		107000
#define GL_STRICT_DEPTHFUNC_HINT_PGI		107030
#define GL_STRICT_LIGHTING_HINT_PGI		107031
#define GL_STRICT_SCISSOR_HINT_PGI		107032
#define GL_FULL_STIPPLE_HINT_PGI		107033
#define GL_NATIVE_GRAPHICS_BEGIN_HINT_PGI	107011
#define GL_NATIVE_GRAPHICS_END_HINT_PGI		107012
#define GL_CONSERVE_MEMORY_HINT_PGI		107005
#define GL_RECLAIM_MEMORY_HINT_PGI		107006
#define GL_ALWAYS_FAST_HINT_PGI			107020
#define GL_ALWAYS_SOFT_HINT_PGI			107021
#define GL_ALLOW_DRAW_OBJ_HINT_PGI		107022
#define GL_ALLOW_DRAW_WIN_HINT_PGI		107023
#define GL_ALLOW_DRAW_FRG_HINT_PGI		107024
#define GL_ALLOW_DRAW_SPN_HINT_PGI		107024
#define GL_ALLOW_DRAW_MEM_HINT_PGI		107025
#define GL_CLIP_NEAR_HINT_PGI			107040
#define GL_CLIP_FAR_HINT_PGI			107041
#define GL_WIDE_LINE_HINT_PGI		  	107042
#define GL_BACK_NORMALS_HINT_PGI		107043
#define GL_NATIVE_GRAPHICS_HANDLE_PGI		107010
#define GL_EXT_paletted_texture 1
#define GL_TABLE_TOO_LARGE_EXT			0x8031
#define GL_COLOR_TABLE_FORMAT_EXT		0x80D8
#define GL_COLOR_TABLE_WIDTH_EXT		0x80D9
#define GL_COLOR_TABLE_RED_SIZE_EXT		0x80DA
#define GL_COLOR_TABLE_GREEN_SIZE_EXT		0x80DB
#define GL_COLOR_TABLE_BLUE_SIZE_EXT		0x80DC
#define GL_COLOR_TABLE_ALPHA_SIZE_EXT	 	0x80DD
#define GL_COLOR_TABLE_LUMINANCE_SIZE_EXT	0x80DE
#define GL_COLOR_TABLE_INTENSITY_SIZE_EXT	0x80DF
#define GL_TEXTURE_INDEX_SIZE_EXT		0x80ED
#define GL_COLOR_INDEX1_EXT			0x80E2
#define GL_COLOR_INDEX2_EXT			0x80E3
#define GL_COLOR_INDEX4_EXT			0x80E4
#define GL_COLOR_INDEX8_EXT			0x80E5
#define GL_COLOR_INDEX12_EXT			0x80E6
#define GL_COLOR_INDEX16_EXT			0x80E7
#define GL_EXT_clip_volume_hint 1
#define GL_CLIP_VOLUME_CLIPPING_HINT_EXT	0x80F
#define GL_EXT_compiled_vertex_array 1
#define GL_ARRAY_ELEMENT_LOCK_FIRST_SGI		0x81A8
#define GL_ARRAY_ELEMENT_LOCK_COUNT_SGI		0x81A9
#define GL_EXT_shared_texture_palette 1
#define GL_SHARED_TEXTURE_PALETTE_EXT		0x81FB
#define GL_EXT_stencil_wrap 1
#define GL_INCR_WRAP_EXT			0x8507
#define GL_DECR_WRAP_EXT			0x8508
#define GL_INGR_blend_func_separate 1
#define GL_BLEND_DST_RGB_INGR			0x80C8
#define GL_BLEND_SRC_RGB_INGR			0x80C9
#define GL_BLEND_DST_ALPHA_INGR			0x80CA
#define GL_BLEND_SRC_ALPHA_INGR			0x80CB
#define GL_NV_texgen_reflection 1
#define GL_NORMAL_MAP_NV			0x8511
#define GL_REFLECTION_MAP_NV			0x8512
#define GL_EXT_texture_env_add 1
#define GL_ARB_multitexture 1
#define GL_TEXTURE0_ARB				0x84C0
#define GL_TEXTURE1_ARB				0x84C1
#define GL_TEXTURE2_ARB				0x84C2
#define GL_TEXTURE3_ARB				0x84C3
#define GL_TEXTURE4_ARB				0x84C4
#define GL_TEXTURE5_ARB				0x84C5
#define GL_TEXTURE6_ARB				0x84C6
#define GL_TEXTURE7_ARB				0x84C7
#define GL_TEXTURE8_ARB				0x84C8
#define GL_TEXTURE9_ARB				0x84C9
#define GL_TEXTURE10_ARB			0x84CA
#define GL_TEXTURE11_ARB			0x84CB
#define GL_TEXTURE12_ARB			0x84CC
#define GL_TEXTURE13_ARB			0x84CD
#define GL_TEXTURE14_ARB			0x84CE
#define GL_TEXTURE15_ARB			0x84CF
#define GL_TEXTURE16_ARB			0x84D0
#define GL_TEXTURE17_ARB			0x84D1
#define GL_TEXTURE18_ARB			0x84D2
#define GL_TEXTURE19_ARB			0x84D3
#define GL_TEXTURE20_ARB			0x84D4
#define GL_TEXTURE21_ARB			0x84D5
#define GL_TEXTURE22_ARB			0x84D6
#define GL_TEXTURE23_ARB			0x84D7
#define GL_TEXTURE24_ARB			0x84D8
#define GL_TEXTURE25_ARB			0x84D9
#define GL_TEXTURE26_ARB			0x84DA
#define GL_TEXTURE27_ARB			0x84DB
#define GL_TEXTURE28_ARB			0x84DC
#define GL_TEXTURE29_ARB			0x84DD
#define GL_TEXTURE30_ARB			0x84DE
#define GL_TEXTURE31_ARB			0x84DF
#define GL_ACTIVE_TEXTURE_ARB			0x84E0
#define GL_CLIENT_ACTIVE_TEXTURE_ARB		0x84E1
#define GL_MAX_TEXTURE_UNITS_ARB		0x84E2
#define GL_MESA_window_pos 1
#define GL_MESA_resize_buffers 1
#define GL_ARB_transpose_matrix 1
#define GL_TRANSPOSE_MODELVIEW_MATRIX_ARB		0x84E3
#define GL_TRANSPOSE_PROJECTION_MATRIX_ARB		0x84E4
#define GL_TRANSPOSE_TEXTURE_MATRIX_ARB			0x84E5
#define GL_TRANSPOSE_COLOR_MATRIX_ARB			0x84E6


/* glu.h */
#define __glu_h__
#define GLUAPI
#define GLAPIENTRY
#define GLCALLBACK
#define GLU_VERSION_1_1			1
#define GLU_VERSION_1_2			1
#define GLU_TRUE			1
#define GLU_FALSE			0
#define GLU_SMOOTH			100000
#define GLU_FLAT			100001
#define GLU_NONE			100002
#define GLU_POINT			100010
#define GLU_LINE			100011
#define GLU_FILL			100012
#define GLU_SILHOUETTE			100013
#define GLU_OUTSIDE			100020
#define GLU_INSIDE			100021
#define GLU_TESS_BEGIN			100100
#define GLU_TESS_VERTEX			100101
#define GLU_TESS_END			100102
#define GLU_TESS_ERROR			100103
#define GLU_TESS_EDGE_FLAG		100104
#define GLU_TESS_COMBINE		100105
#define GLU_TESS_BEGIN_DATA		100106
#define GLU_TESS_VERTEX_DATA		100107
#define GLU_TESS_END_DATA		100108
#define GLU_TESS_ERROR_DATA		100109
#define GLU_TESS_EDGE_FLAG_DATA		100110
#define GLU_TESS_COMBINE_DATA		100111
#define GLU_TESS_WINDING_ODD		100130
#define GLU_TESS_WINDING_NONZERO	100131
#define GLU_TESS_WINDING_POSITIVE	100132
#define GLU_TESS_WINDING_NEGATIVE	100133
#define GLU_TESS_WINDING_ABS_GEQ_TWO	100134
#define GLU_TESS_WINDING_RULE		100140
#define GLU_TESS_BOUNDARY_ONLY		100141
#define GLU_TESS_TOLERANCE		100142
#define GLU_TESS_ERROR1			100151  /* Missing gluBeginPolygon */
#define GLU_TESS_ERROR2			100152  /* Missing gluBeginContour */
#define GLU_TESS_ERROR3			100153  /* Missing gluEndPolygon */
#define GLU_TESS_ERROR4			100154  /* Missing gluEndContour */
#define GLU_TESS_ERROR5			100155  /* */
#define GLU_TESS_ERROR6			100156  /* */
#define GLU_TESS_ERROR7			100157  /* */
#define GLU_TESS_ERROR8			100158  /* */
#define GLU_AUTO_LOAD_MATRIX		100200
#define GLU_CULLING			100201
#define GLU_PARAMETRIC_TOLERANCE	100202
#define GLU_SAMPLING_TOLERANCE		100203
#define GLU_DISPLAY_MODE		100204
#define GLU_SAMPLING_METHOD		100205
#define GLU_U_STEP			100206
#define GLU_V_STEP			100207
#define GLU_PATH_LENGTH			100215
#define GLU_PARAMETRIC_ERROR		100216
#define GLU_DOMAIN_DISTANCE		100217
#define GLU_MAP1_TRIM_2			100210
#define GLU_MAP1_TRIM_3			100211
#define GLU_OUTLINE_POLYGON		100240
#define GLU_OUTLINE_PATCH		100241
#define GLU_NURBS_ERROR1	100251   /* spline order un-supported */
#define GLU_NURBS_ERROR2	100252   /* too few knots */
#define GLU_NURBS_ERROR3	100253   /* valid knot range is empty */
#define GLU_NURBS_ERROR4	100254   /* decreasing knot sequence */
#define GLU_NURBS_ERROR5	100255   /* knot multiplicity > spline order */
#define GLU_NURBS_ERROR6	100256   /* endcurve() must follow bgncurve() */
#define GLU_NURBS_ERROR7	100257   /* bgncurve() must precede endcurve() */
#define GLU_NURBS_ERROR8	100258   /* ctrlarray or knot vector is NULL */
#define GLU_NURBS_ERROR9 	100259   /* can't draw pwlcurves */
#define GLU_NURBS_ERROR10	100260   /* missing gluNurbsCurve() */
#define GLU_NURBS_ERROR11	100261   /* missing gluNurbsSurface() */
#define GLU_NURBS_ERROR12	100262   /* endtrim() must precede endsurface() */
#define GLU_NURBS_ERROR13	100263   /* bgnsurface() must precede endsurface() */
#define GLU_NURBS_ERROR14	100264   /* curve of improper type passed as trim curve */
#define GLU_NURBS_ERROR15	100265   /* bgnsurface() must precede bgntrim() */
#define GLU_NURBS_ERROR16	100266   /* endtrim() must follow bgntrim() */
#define GLU_NURBS_ERROR17	100267   /* bgntrim() must precede endtrim()*/
#define GLU_NURBS_ERROR18	100268   /* invalid or missing trim curve*/
#define GLU_NURBS_ERROR19	100269   /* bgntrim() must precede pwlcurve() */
#define GLU_NURBS_ERROR20	100270   /* pwlcurve referenced twice*/
#define GLU_NURBS_ERROR21	100271   /* pwlcurve and nurbscurve mixed */
#define GLU_NURBS_ERROR22	100272   /* improper usage of trim data type */
#define GLU_NURBS_ERROR23	100273   /* nurbscurve referenced twice */
#define GLU_NURBS_ERROR24	100274   /* nurbscurve and pwlcurve mixed */
#define GLU_NURBS_ERROR25	100275   /* nurbssurface referenced twice */
#define GLU_NURBS_ERROR26	100276   /* invalid property */
#define GLU_NURBS_ERROR27	100277   /* endsurface() must follow bgnsurface() */
#define GLU_NURBS_ERROR28	100278   /* intersecting or misoriented trim curves */
#define GLU_NURBS_ERROR29	100279   /* intersecting trim curves */
#define GLU_NURBS_ERROR30	100280   /* UNUSED */
#define GLU_NURBS_ERROR31	100281   /* unconnected trim curves */
#define GLU_NURBS_ERROR32	100282   /* unknown knot error */
#define GLU_NURBS_ERROR33	100283   /* negative vertex count encountered */
#define GLU_NURBS_ERROR34	100284   /* negative byte-stride */
#define GLU_NURBS_ERROR35	100285   /* unknown type descriptor */
#define GLU_NURBS_ERROR36	100286   /* null control point reference */
#define GLU_NURBS_ERROR37	100287   /* duplicate point on pwlcurve */
#define GLU_INVALID_ENUM		100900
#define GLU_INVALID_VALUE		100901
#define GLU_OUT_OF_MEMORY		100902
#define GLU_INCOMPATIBLE_GL_VERSION	100903
#define GLU_VERSION			100800
#define GLU_EXTENSIONS			100801
#define GLU_CW				100120
#define GLU_CCW				100121
#define GLU_INTERIOR			100122
#define GLU_EXTERIOR			100123
#define GLU_UNKNOWN			100124
#define GLU_BEGIN			GLU_TESS_BEGIN
#define GLU_VERTEX			GLU_TESS_VERTEX
#define GLU_END				GLU_TESS_END
#define GLU_ERROR			GLU_TESS_ERROR
#define GLU_EDGE_FLAG			GLU_TESS_EDGE_FLAG

/* glut.h */
#define GLUT_API_VERSION		3
#define GLUT_XLIB_IMPLEMENTATION	15
#define GLUT_RGB			0
#define GLUT_RGBA			GLUT_RGB
#define GLUT_INDEX			1
#define GLUT_SINGLE			0
#define GLUT_DOUBLE			2
#define GLUT_ACCUM			4
#define GLUT_ALPHA			8
#define GLUT_DEPTH			16
#define GLUT_STENCIL			32
#define GLUT_MULTISAMPLE		128
#define GLUT_STEREO			256
#define GLUT_LUMINANCE			512
#define GLUT_LEFT_BUTTON		0
#define GLUT_MIDDLE_BUTTON		1
#define GLUT_RIGHT_BUTTON		2
#define GLUT_DOWN			0
#define GLUT_UP				1
#define GLUT_KEY_F1			1
#define GLUT_KEY_F2			2
#define GLUT_KEY_F3			3
#define GLUT_KEY_F4			4
#define GLUT_KEY_F5			5
#define GLUT_KEY_F6			6
#define GLUT_KEY_F7			7
#define GLUT_KEY_F8			8
#define GLUT_KEY_F9			9
#define GLUT_KEY_F10			10
#define GLUT_KEY_F11			11
#define GLUT_KEY_F12			12
#define GLUT_KEY_LEFT			100
#define GLUT_KEY_UP			101
#define GLUT_KEY_RIGHT			102
#define GLUT_KEY_DOWN			103
#define GLUT_KEY_PAGE_UP		104
#define GLUT_KEY_PAGE_DOWN		105
#define GLUT_KEY_HOME			106
#define GLUT_KEY_END			107
#define GLUT_KEY_INSERT			108
#define GLUT_LEFT			0
#define GLUT_ENTERED			1
#define GLUT_MENU_NOT_IN_USE		0
#define GLUT_MENU_IN_USE		1
#define GLUT_NOT_VISIBLE		0
#define GLUT_VISIBLE			1
#define GLUT_HIDDEN			0
#define GLUT_FULLY_RETAINED		1
#define GLUT_PARTIALLY_RETAINED		2
#define GLUT_FULLY_COVERED		3
#define GLUT_RED			0
#define GLUT_GREEN			1
#define GLUT_BLUE			2
#define GLUT_NORMAL			0
#define GLUT_OVERLAY			1
#define GLUT_STROKE_ROMAN		((void*)0)
#define GLUT_STROKE_MONO_ROMAN		((void*)1)
#define GLUT_BITMAP_9_BY_15		((void*)2)
#define GLUT_BITMAP_8_BY_13		((void*)3)
#define GLUT_BITMAP_TIMES_ROMAN_10	((void*)4)
#define GLUT_BITMAP_TIMES_ROMAN_24	((void*)5)
#define GLUT_BITMAP_HELVETICA_10	((void*)6)
#define GLUT_BITMAP_HELVETICA_12	((void*)7)
#define GLUT_BITMAP_HELVETICA_18	((void*)8)
#if 0
#define GLUT_STROKE_ROMAN		(&glutStrokeRoman)
#define GLUT_STROKE_MONO_ROMAN		(&glutStrokeMonoRoman)
#define GLUT_BITMAP_9_BY_15		(&glutBitmap9By15)
#define GLUT_BITMAP_8_BY_13		(&glutBitmap8By13)
#define GLUT_BITMAP_TIMES_ROMAN_10	(&glutBitmapTimesRoman10)
#define GLUT_BITMAP_TIMES_ROMAN_24	(&glutBitmapTimesRoman24)
#define GLUT_BITMAP_HELVETICA_10	(&glutBitmapHelvetica10)
#define GLUT_BITMAP_HELVETICA_12	(&glutBitmapHelvetica12)
#define GLUT_BITMAP_HELVETICA_18	(&glutBitmapHelvetica18)
#endif
#define GLUT_WINDOW_X			100
#define GLUT_WINDOW_Y			101
#define GLUT_WINDOW_WIDTH		102
#define GLUT_WINDOW_HEIGHT		103
#define GLUT_WINDOW_BUFFER_SIZE		104
#define GLUT_WINDOW_STENCIL_SIZE	105
#define GLUT_WINDOW_DEPTH_SIZE		106
#define GLUT_WINDOW_RED_SIZE		107
#define GLUT_WINDOW_GREEN_SIZE		108
#define GLUT_WINDOW_BLUE_SIZE		109
#define GLUT_WINDOW_ALPHA_SIZE		110
#define GLUT_WINDOW_ACCUM_RED_SIZE	111
#define GLUT_WINDOW_ACCUM_GREEN_SIZE	112
#define GLUT_WINDOW_ACCUM_BLUE_SIZE	113
#define GLUT_WINDOW_ACCUM_ALPHA_SIZE	114
#define GLUT_WINDOW_DOUBLEBUFFER	115
#define GLUT_WINDOW_RGBA		116
#define GLUT_WINDOW_PARENT		117
#define GLUT_WINDOW_NUM_CHILDREN	118
#define GLUT_WINDOW_COLORMAP_SIZE	119
#define GLUT_WINDOW_NUM_SAMPLES		120
#define GLUT_WINDOW_STEREO		121
#define GLUT_WINDOW_CURSOR		122
#define GLUT_SCREEN_WIDTH		200
#define GLUT_SCREEN_HEIGHT		201
#define GLUT_SCREEN_WIDTH_MM		202
#define GLUT_SCREEN_HEIGHT_MM		203
#define GLUT_MENU_NUM_ITEMS		300
#define GLUT_DISPLAY_MODE_POSSIBLE	400
#define GLUT_INIT_WINDOW_X		500
#define GLUT_INIT_WINDOW_Y		501
#define GLUT_INIT_WINDOW_WIDTH		502
#define GLUT_INIT_WINDOW_HEIGHT		503
#define GLUT_INIT_DISPLAY_MODE		504
#define GLUT_ELAPSED_TIME		700
#define GLUT_WINDOW_FORMAT_ID		123
#define GLUT_HAS_KEYBOARD		600
#define GLUT_HAS_MOUSE			601
#define GLUT_HAS_SPACEBALL		602
#define GLUT_HAS_DIAL_AND_BUTTON_BOX	603
#define GLUT_HAS_TABLET			604
#define GLUT_NUM_MOUSE_BUTTONS		605
#define GLUT_NUM_SPACEBALL_BUTTONS	606
#define GLUT_NUM_BUTTON_BOX_BUTTONS	607
#define GLUT_NUM_DIALS			608
#define GLUT_NUM_TABLET_BUTTONS		609
#define GLUT_DEVICE_IGNORE_KEY_REPEAT   610
#define GLUT_DEVICE_KEY_REPEAT          611
#define GLUT_HAS_JOYSTICK		612
#define GLUT_OWNS_JOYSTICK		613
#define GLUT_JOYSTICK_BUTTONS		614
#define GLUT_JOYSTICK_AXES		615
#define GLUT_JOYSTICK_POLL_RATE		616
#define GLUT_OVERLAY_POSSIBLE           800
#define GLUT_LAYER_IN_USE		801
#define GLUT_HAS_OVERLAY		802
#define GLUT_TRANSPARENT_INDEX		803
#define GLUT_NORMAL_DAMAGED		804
#define GLUT_OVERLAY_DAMAGED		805
#define GLUT_VIDEO_RESIZE_POSSIBLE	900
#define GLUT_VIDEO_RESIZE_IN_USE	901
#define GLUT_VIDEO_RESIZE_X_DELTA	902
#define GLUT_VIDEO_RESIZE_Y_DELTA	903
#define GLUT_VIDEO_RESIZE_WIDTH_DELTA	904
#define GLUT_VIDEO_RESIZE_HEIGHT_DELTA	905
#define GLUT_VIDEO_RESIZE_X		906
#define GLUT_VIDEO_RESIZE_Y		907
#define GLUT_VIDEO_RESIZE_WIDTH		908
#define GLUT_VIDEO_RESIZE_HEIGHT	909
#define GLUT_NORMAL			0
#define GLUT_OVERLAY			1
#define GLUT_ACTIVE_SHIFT               1
#define GLUT_ACTIVE_CTRL                2
#define GLUT_ACTIVE_ALT                 4
#define GLUT_CURSOR_RIGHT_ARROW		0
#define GLUT_CURSOR_LEFT_ARROW		1
#define GLUT_CURSOR_INFO		2
#define GLUT_CURSOR_DESTROY		3
#define GLUT_CURSOR_HELP		4
#define GLUT_CURSOR_CYCLE		5
#define GLUT_CURSOR_SPRAY		6
#define GLUT_CURSOR_WAIT		7
#define GLUT_CURSOR_TEXT		8
#define GLUT_CURSOR_CROSSHAIR		9
#define GLUT_CURSOR_UP_DOWN		10
#define GLUT_CURSOR_LEFT_RIGHT		11
#define GLUT_CURSOR_TOP_SIDE		12
#define GLUT_CURSOR_BOTTOM_SIDE		13
#define GLUT_CURSOR_LEFT_SIDE		14
#define GLUT_CURSOR_RIGHT_SIDE		15
#define GLUT_CURSOR_TOP_LEFT_CORNER	16
#define GLUT_CURSOR_TOP_RIGHT_CORNER	17
#define GLUT_CURSOR_BOTTOM_RIGHT_CORNER	18
#define GLUT_CURSOR_BOTTOM_LEFT_CORNER	19
#define GLUT_CURSOR_INHERIT		100
#define GLUT_CURSOR_NONE		101
#define GLUT_CURSOR_FULL_CROSSHAIR	102
#define GLUT_KEY_REPEAT_OFF		0
#define GLUT_KEY_REPEAT_ON		1
#define GLUT_KEY_REPEAT_DEFAULT		2
#define GLUT_JOYSTICK_BUTTON_A		1
#define GLUT_JOYSTICK_BUTTON_B		2
#define GLUT_JOYSTICK_BUTTON_C		4
#define GLUT_JOYSTICK_BUTTON_D		8
#define GLUT_GAME_MODE_ACTIVE           0
#define GLUT_GAME_MODE_POSSIBLE         1
#define GLUT_GAME_MODE_WIDTH            2
#define GLUT_GAME_MODE_HEIGHT           3
#define GLUT_GAME_MODE_PIXEL_DEPTH      4
#define GLUT_GAME_MODE_REFRESH_RATE     5
#define GLUT_GAME_MODE_DISPLAY_CHANGED  6

/* glx.h */
#define GLX_VERSION_1_1		1
#define GLX_VERSION_1_2		1
#define GLX_VERSION_1_3		1
#define GLX_EXTENSION_NAME   "GLX"
#define GLX_USE_GL		1
#define GLX_BUFFER_SIZE		2
#define GLX_LEVEL		3
#define GLX_RGBA		4
#define GLX_DOUBLEBUFFER	5
#define GLX_STEREO		6
#define GLX_AUX_BUFFERS		7
#define GLX_RED_SIZE		8
#define GLX_GREEN_SIZE		9
#define GLX_BLUE_SIZE		10
#define GLX_ALPHA_SIZE		11
#define GLX_DEPTH_SIZE		12
#define GLX_STENCIL_SIZE	13
#define GLX_ACCUM_RED_SIZE	14
#define GLX_ACCUM_GREEN_SIZE	15
#define GLX_ACCUM_BLUE_SIZE	16
#define GLX_ACCUM_ALPHA_SIZE	17
#define GLX_BAD_SCREEN		1
#define GLX_BAD_ATTRIBUTE	2
#define GLX_NO_EXTENSION	3
#define GLX_BAD_VISUAL		4
#define GLX_BAD_CONTEXT		5
#define GLX_BAD_VALUE       	6
#define GLX_BAD_ENUM		7
#define GLX_VENDOR		1
#define GLX_VERSION		2
#define GLX_EXTENSIONS 		3
#if 0
#define GLX_CONFIG_CAVEAT		?
#define GLX_DONT_CARE			?
#define GLX_SLOW_CONFIG			?
#define GLX_NON_CONFORMANT_CONFIG	?
#endif
#define GLX_X_VISUAL_TYPE		0x22
#define GLX_TRANSPARENT_TYPE		0x23
#define GLX_TRANSPARENT_INDEX_VALUE	0x24
#define GLX_TRANSPARENT_RED_VALUE	0x25
#define GLX_TRANSPARENT_GREEN_VALUE	0x26
#define GLX_TRANSPARENT_BLUE_VALUE	0x27
#define GLX_TRANSPARENT_ALPHA_VALUE	0x28
#if 0
#define GLX_MAX_PBUFFER_WIDTH		?
#define GLX_MAX_PBUFFER_HEIGHT		?
#define GLX_MAX_PBUFFER_PIXELS		?
#define GLX_PRESERVED_CONTENTS		?
#define GLX_LARGEST_BUFFER		?
#define GLX_DRAWABLE_TYPE		?
#define GLX_FBCONFIG_ID			?
#define GLX_VISUAL_ID			?
#define GLX_WINDOW_BIT			?
#define GLX_PIXMAP_BIT			?
#define GLX_PBUFFER_BIT			?
#define GLX_AUX_BUFFERS_BIT		?
#define GLX_FRONT_LEFT_BUFFER_BIT	?
#define GLX_FRONT_RIGHT_BUFFER_BIT	?
#define GLX_BACK_LEFT_BUFFER_BIT	?
#define GLX_BACK_RIGHT_BUFFER_BIT	?
#define GLX_AUX_BUFFERS_BIT		?
#define GLX_DEPTH_BUFFER_BIT		?
#define GLX_STENCIL_BUFFER_BIT		?
#define GLX_ACCUM_BUFFER_BIT		?
#define GLX_RENDER_TYPE			?
#define GLX_DRAWABLE_TYPE		?
#define GLX_X_RENDERABLE		?
#endif
#define GLX_NONE			0x8000
#define GLX_TRUE_COLOR			0x8002
#define GLX_DIRECT_COLOR		0x8003
#define GLX_PSEUDO_COLOR		0x8004
#define GLX_STATIC_COLOR		0x8005
#define GLX_GRAY_SCALE			0x8006
#define GLX_STATIC_GRAY			0x8007
#define GLX_TRANSPARENT_INDEX		0x8009
#if 0
#define GLX_COLOR_INDEX_TYPE		?
#define GLX_COLOR_INDEX_BIT		?
#define GLX_SCREEN			?
#define GLX_PBUFFER_CLOBBER_MASK	?
#define GLX_DAMAGED			?
#define GLX_SAVED			?
#define GLX_WINDOW			?
#define GLX_PBUFFER			?
#endif
#define GLX_X_VISUAL_TYPE_EXT		0x22
#define GLX_TRANSPARENT_TYPE_EXT	0x23
#define GLX_TRANSPARENT_INDEX_VALUE_EXT	0x24
#define GLX_TRANSPARENT_RED_VALUE_EXT	0x25
#define GLX_TRANSPARENT_GREEN_VALUE_EXT	0x26
#define GLX_TRANSPARENT_BLUE_VALUE_EXT	0x27
#define GLX_TRANSPARENT_ALPHA_VALUE_EXT	0x28
#define GLX_TRUE_COLOR_EXT		0x8002
#define GLX_DIRECT_COLOR_EXT		0x8003
#define GLX_PSEUDO_COLOR_EXT		0x8004
#define GLX_STATIC_COLOR_EXT		0x8005
#define GLX_GRAY_SCALE_EXT		0x8006
#define GLX_STATIC_GRAY_EXT		0x8007
#define GLX_NONE_EXT			0x8000
#define GLX_TRANSPARENT_RGB_EXT		0x8008
#define GLX_TRANSPARENT_INDEX_EXT	0x8009
#define GLX_EXT_visual_info		1
#define GLX_MESA_pixmap_colormap	1
#define GLX_MESA_release_buffers	1
#define GLX_MESA_copy_sub_buffer	1
#define GLX_MESA_set_3dfx_mode		1
#define GLX_SGI_video_sync		1
#define GLX_ARB_get_proc_address	1

/* xmesa.h */
#define XMESA_MAJOR_VERSION 3
#define XMESA_MINOR_VERSION 3
#define XMESA_VERSION 1
#define XMESA_EXTENSIONS 2
#define XMESA_FX_WINDOW       1
#define XMESA_FX_FULLSCREEN   2

#endif
