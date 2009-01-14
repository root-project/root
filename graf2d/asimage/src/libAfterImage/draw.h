#ifndef DRAW_H_HEADER_INCLUDED
#define DRAW_H_HEADER_INCLUDED

typedef struct ASDrawTool
{
	int width;
	int height;
	int center_x, center_y ;
	CARD32  *matrix ;
}ASDrawTool;

typedef struct ASDrawContext
{
#define ASDrawCTX_UsingScratch	(0x01<<0)	
#define ASDrawCTX_CanvasIsARGB	(0x01<<1)
#define ASDrawCTX_ToolIsARGB	(0x01<<2)
	ASFlagType flags ;

	ASDrawTool *tool ;
	
	int canvas_width, canvas_height ;
	CARD32 *canvas ;
	CARD32 *scratch_canvas ;

	int curr_x, curr_y ;

	void (*apply_tool_func)( struct ASDrawContext *ctx, int curr_x, int curr_y, CARD32 ratio );
	void (*fill_hline_func)( struct ASDrawContext *ctx, int x_from, int y, int x_to, CARD32 ratio );
}ASDrawContext;

#define AS_DRAW_BRUSHES	3

ASDrawContext *create_asdraw_context( unsigned int width, unsigned int height );
Bool apply_asdraw_context( ASImage *im, ASDrawContext *ctx, ASFlagType filter );
void destroy_asdraw_context( ASDrawContext *ctx );

Bool asim_set_brush( ASDrawContext *ctx, int brush );
Bool asim_set_custom_brush( ASDrawContext *ctx, ASDrawTool *brush);
Bool asim_set_custom_brush_colored( ASDrawContext *ctx, ASDrawTool *brush);

Bool asim_start_path( ASDrawContext *ctx );
Bool asim_apply_path( ASDrawContext *ctx, int start_x, int start_y, Bool fill, int fill_start_x, int fill_start_y, CARD8 fill_threshold );

void asim_move_to( ASDrawContext *ctx, int dst_x, int dst_y );
void asim_line_to( ASDrawContext *ctx, int dst_x, int dst_y );
void asim_line_to_aa( ASDrawContext *ctx, int dst_x, int dst_y );
void asim_cube_bezier( ASDrawContext *ctx, int x1, int y1, int x2, int y2, int x3, int y3 );

void asim_straight_ellips( ASDrawContext *ctx, int x, int y, int rx, int ry, Bool fill );
void asim_circle( ASDrawContext *ctx, int x, int y, int r, Bool fill );
void asim_ellips( ASDrawContext *ctx, int x, int y, int rx, int ry, int angle, Bool fill );
void asim_ellips2( ASDrawContext *ctx, int x, int y, int rx, int ry, int angle, Bool fill );
void asim_rectangle( ASDrawContext *ctx, int x, int y, int width, int height );

void asim_flood_fill( ASDrawContext *ctx, int x, int y, CARD32 min_val, CARD32 max_val );

#endif /* DRAW_H_HEADER_INCLUDED */
