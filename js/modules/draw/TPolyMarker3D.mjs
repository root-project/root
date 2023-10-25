import { settings, isObject } from '../core.mjs';
import { PointsCreator } from '../base/base3d.mjs';

/** @summary direct draw function for TPolyMarker3D object
  * @private */
async function drawPolyMarker3D() {
   const fp = this.$fp || this.getFramePainter();

   delete this.$fp;

   if (!isObject(fp) || !fp.grx || !fp.gry || !fp.grz)
      return this;

   const poly = this.getObject(), sizelimit = 50000, fP = poly.fP;
   let step = 1, numselect = 0;

   for (let i = 0; i < fP.length; i += 3) {
      if ((fP[i] < fp.scale_xmin) || (fP[i] > fp.scale_xmax) ||
          (fP[i+1] < fp.scale_ymin) || (fP[i+1] > fp.scale_ymax) ||
          (fP[i+2] < fp.scale_zmin) || (fP[i+2] > fp.scale_zmax)) continue;
      ++numselect;
   }

   if ((settings.OptimizeDraw > 0) && (numselect > sizelimit)) {
      step = Math.floor(numselect/sizelimit);
      if (step <= 2) step = 2;
   }

   const size = Math.floor(numselect/step),
         pnts = new PointsCreator(size, fp.webgl, fp.size_x3d/100),
         index = new Int32Array(size);
   let select = 0, icnt = 0;

   for (let i = 0; i < fP.length; i += 3) {
      if ((fP[i] < fp.scale_xmin) || (fP[i] > fp.scale_xmax) ||
          (fP[i+1] < fp.scale_ymin) || (fP[i+1] > fp.scale_ymax) ||
          (fP[i+2] < fp.scale_zmin) || (fP[i+2] > fp.scale_zmax)) continue;

      if (step > 1) {
         select = (select+1) % step;
         if (select !== 0) continue;
      }

      index[icnt++] = i;

      pnts.addPoint(fp.grx(fP[i]), fp.gry(fP[i+1]), fp.grz(fP[i+2]));
   }

   return pnts.createPoints({ color: this.getColor(poly.fMarkerColor), style: poly.fMarkerStyle }).then(mesh => {
      mesh.tip_color = (poly.fMarkerColor === 3) ? 0xFF0000 : 0x00FF00;
      mesh.tip_name = poly.fName || 'Poly3D';
      mesh.poly = poly;
      mesh.fp = fp;
      mesh.scale0 = 0.7*pnts.scale;
      mesh.index = index;

      fp.add3DMesh(mesh, this, true);

      mesh.tooltip = function(intersect) {
         let indx = Math.floor(intersect.index / this.nvertex);
         if ((indx < 0) || (indx >= this.index.length)) return null;

         indx = this.index[indx];

         const fp = this.fp,
               grx = fp.grx(this.poly.fP[indx]),
               gry = fp.gry(this.poly.fP[indx+1]),
               grz = fp.grz(this.poly.fP[indx+2]);

         return {
            x1: grx - this.scale0,
            x2: grx + this.scale0,
            y1: gry - this.scale0,
            y2: gry + this.scale0,
            z1: grz - this.scale0,
            z2: grz + this.scale0,
            color: this.tip_color,
            lines: [this.tip_name,
                     'pnt: ' + indx/3,
                     'x: ' + fp.axisAsText('x', this.poly.fP[indx]),
                     'y: ' + fp.axisAsText('y', this.poly.fP[indx+1]),
                     'z: ' + fp.axisAsText('z', this.poly.fP[indx+2])
                   ]
         };
      };

      fp.render3D(100); // set timeout to be able draw other points

      return this;
   });
}

export { drawPolyMarker3D };
