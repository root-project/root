import { REVISION } from '../three.mjs';

import { settings } from '../core.mjs';

import { createLineSegments, PointsCreator, create3DLineMaterial } from '../base/base3d.mjs';

import { drawDummy3DGeom } from '../geom/TGeoPainter.mjs';

function before3DDraw(painter, obj) {
   let fp = painter.getFramePainter();

   if (!fp || !fp.mode3d || !obj)
      return null;

   if (fp.toplevel)
      return fp;

   let geop = painter.getMainPainter();
   if(!geop)
      return drawDummy3DGeom(painter);
   else if (typeof geop.drawExtras == 'function')
      return geop.drawExtras(obj);

   return null;
}


/** @summary direct draw function for TPolyMarker3D object
  * @private */
function drawPolyMarker3D() {

   let poly = this.getObject(),
       fp = before3DDraw(this, poly);

   if (!fp || (typeof fp !== 'object') || !fp.grx || !fp.gry || !fp.grz)
      return fp;

   let step = 1, sizelimit = 50000, numselect = 0;

   for (let i = 0; i < poly.fP.length; i += 3) {
      if ((poly.fP[i] < fp.scale_xmin) || (poly.fP[i] > fp.scale_xmax) ||
          (poly.fP[i+1] < fp.scale_ymin) || (poly.fP[i+1] > fp.scale_ymax) ||
          (poly.fP[i+2] < fp.scale_zmin) || (poly.fP[i+2] > fp.scale_zmax)) continue;
      ++numselect;
   }

   if ((settings.OptimizeDraw > 0) && (numselect > sizelimit)) {
      step = Math.floor(numselect/sizelimit);
      if (step <= 2) step = 2;
   }

   let size = Math.floor(numselect/step),
       pnts = new PointsCreator(size, fp.webgl, fp.size_x3d/100),
       index = new Int32Array(size),
       select = 0, icnt = 0;

   for (let i = 0; i < poly.fP.length; i += 3) {

      if ((poly.fP[i] < fp.scale_xmin) || (poly.fP[i] > fp.scale_xmax) ||
          (poly.fP[i+1] < fp.scale_ymin) || (poly.fP[i+1] > fp.scale_ymax) ||
          (poly.fP[i+2] < fp.scale_zmin) || (poly.fP[i+2] > fp.scale_zmax)) continue;

      if (step > 1) {
         select = (select+1) % step;
         if (select!==0) continue;
      }

      index[icnt++] = i;

      pnts.addPoint(fp.grx(poly.fP[i]), fp.gry(poly.fP[i+1]), fp.grz(poly.fP[i+2]));
   }

   return pnts.createPoints({ color: this.getColor(poly.fMarkerColor), style: poly.fMarkerStyle }).then(mesh => {

      mesh.tip_color = (poly.fMarkerColor === 3) ? 0xFF0000 : 0x00FF00;
      mesh.tip_name = poly.fName || "Poly3D";
      mesh.poly = poly;
      mesh.painter = fp;
      mesh.scale0 = 0.7*pnts.scale;
      mesh.index = index;

      fp.toplevel.add(mesh);

      mesh.tooltip = function(intersect) {
         if (!Number.isInteger(intersect.index)) {
            console.error(`intersect.index not provided, three.js version ${REVISION}`);
            return null;
         }
         let indx = Math.floor(intersect.index / this.nvertex);
         if ((indx<0) || (indx >= this.index.length)) return null;

         indx = this.index[indx];

         let p = this.painter,
             grx = p.grx(this.poly.fP[indx]),
             gry = p.gry(this.poly.fP[indx+1]),
             grz = p.grz(this.poly.fP[indx+2]);

         return  {
            x1: grx - this.scale0,
            x2: grx + this.scale0,
            y1: gry - this.scale0,
            y2: gry + this.scale0,
            z1: grz - this.scale0,
            z2: grz + this.scale0,
            color: this.tip_color,
            lines: [ this.tip_name,
                     "pnt: " + indx/3,
                     "x: " + p.axisAsText("x", this.poly.fP[indx]),
                     "y: " + p.axisAsText("y", this.poly.fP[indx+1]),
                     "z: " + p.axisAsText("z", this.poly.fP[indx+2])
                   ]
         };
      };

      fp.render3D(100); // set timeout to be able draw other points

      return this;
   });
}

/** @summary Direct draw function for TPolyLine3D object
  * @desc Takes into account dashed properties
  * @private */
function drawPolyLine3D() {
   let line = this.getObject(),
       fp = before3DDraw(this, line);

   if (!fp || (typeof fp !== 'object') || !fp.grx || !fp.gry || !fp.grz)
      return fp;

   let limit = 3*line.fN, p = line.fP, pnts = [];

   for (let n = 3; n < limit; n += 3)
      pnts.push(fp.grx(p[n-3]), fp.gry(p[n-2]), fp.grz(p[n-1]),
                fp.grx(p[n]), fp.gry(p[n+1]), fp.grz(p[n+2]));

   let lines = createLineSegments(pnts, create3DLineMaterial(this, line));

   fp.toplevel.add(lines);

   return true;
}

export { drawPolyMarker3D, drawPolyLine3D };

