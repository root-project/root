/* @(#)root/g3d:$Id$ */
/* Author: Nenad Buncic   13/12/95*/

#include "X3DBuffer.h"
#include "X3DDefs.h"
#include <stdio.h>
#include <stdlib.h>


#if defined (WIN32) || defined (__MWERKS__)
   void FillX3DBuffer (X3DBuffer *buff) { }
   int  AllocateX3DBuffer () { return 0;}
#else

int currPoint = 0;
int currSeg   = 0;
int currPoly  = 0;

Color   *colors;
point   *points;
segment *segs;
polygon *polys;


int AllocateX3DBuffer ()
{
/******************************************************************************
   Allocate memory for points, colors, segments and polygons.
   Returns 1 if OK, otherwise 0.
******************************************************************************/

    int ret = 1;

    points = NULL;
    colors = NULL;
    segs   = NULL;
    polys  = NULL;

    /*
     *  Allocate memory for points
     */

    if (gSize3D.numPoints) {
        points = (point *) calloc(gSize3D.numPoints, sizeof (point));
        if (!points) {
            puts ("Unable to allocate memory for points !");
            ret = 0;
        }
    }
    else return (0);    /* if there are no points, return back */



    /*
     *  Allocate memory for colors
     */

    colors = (Color *) calloc(28+4, sizeof (Color));
    if(!colors) {
        puts ("Unable to allocate memory for colors !");
        ret = 0;
    }
    else {
        colors[ 0].red = 92;   colors[ 0].green = 92;   colors[0].blue = 92;
        colors[ 1].red = 122;  colors[ 1].green = 122;  colors[1].blue = 122;
        colors[ 2].red = 184;  colors[ 2].green = 184;  colors[2].blue = 184;
        colors[ 3].red = 215;  colors[ 3].green = 215;  colors[3].blue = 215;
        colors[ 4].red = 138;  colors[ 4].green = 15;   colors[4].blue = 15;
        colors[ 5].red = 184;  colors[ 5].green = 20;   colors[5].blue = 20;
        colors[ 6].red = 235;  colors[ 6].green = 71;   colors[6].blue = 71;
        colors[ 7].red = 240;  colors[ 7].green = 117;  colors[7].blue = 117;
        colors[ 8].red = 15;   colors[ 8].green = 138;  colors[8].blue = 15;
        colors[ 9].red = 20;   colors[ 9].green = 184;  colors[9].blue = 20;
        colors[10].red = 71;   colors[10].green = 235;  colors[10].blue = 71;
        colors[11].red = 117;  colors[11].green = 240;  colors[11].blue = 117;
        colors[12].red = 15;   colors[12].green = 15;   colors[12].blue = 138;
        colors[13].red = 20;   colors[13].green = 20;   colors[13].blue = 184;
        colors[14].red = 71;   colors[14].green = 71;   colors[14].blue = 235;
        colors[15].red = 117;  colors[15].green = 117;  colors[15].blue = 240;
        colors[16].red = 138;  colors[16].green = 138;  colors[16].blue = 15;
        colors[17].red = 184;  colors[17].green = 184;  colors[17].blue = 20;
        colors[18].red = 235;  colors[18].green = 235;  colors[18].blue = 71;
        colors[19].red = 240;  colors[19].green = 240;  colors[19].blue = 117;
        colors[20].red = 138;  colors[20].green = 15;   colors[20].blue = 138;
        colors[21].red = 184;  colors[21].green = 20;   colors[21].blue = 184;
        colors[22].red = 235;  colors[22].green = 71;   colors[22].blue = 235;
        colors[23].red = 240;  colors[23].green = 117;  colors[23].blue = 240;
        colors[24].red = 15;   colors[24].green = 138;  colors[24].blue = 138;
        colors[25].red = 20;   colors[25].green = 184;  colors[25].blue = 184;
        colors[26].red = 71;   colors[26].green = 235;  colors[26].blue = 235;
        colors[27].red = 117;  colors[27].green = 240;  colors[27].blue = 240;
    }


    /*
     *  Allocate memory for segments
     */

    if (gSize3D.numSegs) {
        segs = (segment *) calloc (gSize3D.numSegs, sizeof (segment));
        if (!segs) {
            puts ("Unable to allocate memory for segments !");
            ret = 0;
        }
    }


    /*
     * Allocate memory for polygons
     */

    if (gSize3D.numPolys) {
        polys = (polygon *) calloc(gSize3D.numPolys, sizeof (polygon));
        if (!polys) {
            puts ("Unable to allocate memory for polygons !");
            ret = 0;
        }
    }

    /*
     * In case of error, free allocated memory
     */


    if (!ret) {
        if (points) free (points);
        if (colors) free (colors);
        if (segs)   free (segs);
        if (polys)  free (polys);

        points = NULL;
        colors = NULL;
        segs   = NULL;
        polys  = NULL;
    }

    return (ret);
}

void FillX3DBuffer (X3DBuffer *buff)
{
/******************************************************************************
   Read points, Read segments & Read polygons
******************************************************************************/


    int n, i, j, p, q, c;
    int oldNumOfPoints, oldNumOfSegments;

    if (buff) {

        oldNumOfPoints   = currPoint;
        oldNumOfSegments = currSeg;

        /*
         * Read points
         */

        for (i = 0; i < buff->numPoints; i++, currPoint++) {
            points[currPoint].x = buff->points[3*i  ];
            points[currPoint].y = buff->points[3*i+1];
            points[currPoint].z = buff->points[3*i+2];
        }


        /*
         * Read segments
         */

        for (i = 0; i < buff->numSegs; i++, currSeg++) {
            c = buff->segs[3*i];
            p = oldNumOfPoints + buff->segs[3*i+1];
            q = oldNumOfPoints + buff->segs[3*i+2];

            segs[currSeg].color = &(colors[c]);
            segs[currSeg].P     = &(points[p]);
            segs[currSeg].Q     = &(points[q]);

            /*
             * Update points' segment lists
             */

            if(points[p].numSegs == 0){
                if((points[p].segs = (segment **)calloc(1, sizeof(segment *))) == NULL){
                    puts("Unable to allocate memory for point segments !");
                    return;
                }
            }else{
                if((points[p].segs = (segment **)realloc(points[p].segs,
                    (points[p].numSegs + 1) * sizeof(segment *))) == NULL){
                    puts("Unable to allocate memory for point segments !");
                    return;
                }
            }

            if(points[q].numSegs == 0){
                if((points[q].segs = (segment **)calloc(1, sizeof(segment *))) == NULL){
                    puts("Unable to allocate memory for point segments !");
                    return;
                }
            }else{
                if((points[q].segs = (segment **)realloc(points[q].segs,
                    (points[q].numSegs + 1) * sizeof(segment *))) == NULL){
                    puts("Unable to allocate memory for point segments !");
                    return;
                }
            }
            points[p].segs[points[p].numSegs] = &(segs[currSeg]);
            points[q].segs[points[q].numSegs] = &(segs[currSeg]);
            points[p].numSegs++;
            points[q].numSegs++;

        }

        /*
         * Read polygons
         */

        n = 0;

        for (i = 0; i < buff->numPolys; i++, currPoly++) {
            c = buff->polys[n++];
            polys[currPoly].color   = &(colors)[c];
            polys[currPoly].numSegs = buff->polys[n++];

            polys[currPoly].segs    = (segment **) calloc(polys[currPoly].numSegs, sizeof(segment *));
            if (!polys[currPoly].segs) {
                puts("Unable to allocate memory for polygon segments !");
                return;
            }
            for (j = 0; j < polys[currPoly].numSegs; j++) {
                int seg = oldNumOfSegments + buff->polys[n++];
                polys[currPoly].segs[j] = &(segs[seg]);

                /*
                 * Update segments' polygon lists
                 */

                if(segs[seg].numPolys == 0) {
                    if((segs[seg].polys = (polygon **) calloc(1, sizeof(polygon *)))== NULL){
                        puts("Unable to allocate memory for segment polygons !");
                        return;
                    }
                }
                else{
                    if((segs[seg].polys = (polygon **) realloc(segs[seg].polys,
                        (segs[seg].numPolys + 1) * sizeof(polygon *))) == NULL){
                        puts("Unable to allocate memory for segment polygons !");
                        return;
                    }
                }
                segs[seg].polys[segs[seg].numPolys] = &(polys[currPoly]);
                segs[seg].numPolys++;
            }
        }
    }
}

#endif

