import { isObject, isFunc } from '../core.mjs';
import { createLineSegments, create3DLineMaterial } from '../base/base3d.mjs';
import { drawDummy3DGeom } from '../geom/TGeoPainter.mjs';
import { drawPolyMarker3D as drawPolyMarker3Dplain } from './TPolyMarker3D.mjs';


/** @summary Prepare frame painter for 3D drawing
  * @private */
function before3DDraw(painter) {
   const fp = painter.getFramePainter();

   if (!fp?.mode3d || !painter.getObject())
      return null;

   if (fp?.toplevel)
      return fp;

   const main = painter.getMainPainter();

   if (main && !isFunc(main.drawExtras))
      return null;

   const pr = main ? Promise.resolve(main) : drawDummy3DGeom(painter);

   return pr.then(geop => {
      const pp = painter.getPadPainter();
      if (pp) pp._disable_dragging = true;

      if (geop._dummy && isFunc(painter.get3DBox))
         geop.extendCustomBoundingBox(painter.get3DBox());
      return geop.drawExtras(painter.getObject(), '', true, true);
   });
}

/** @summary Function to extract 3DBox for poly marker and line
  * @private */
function get3DBox() {
   const obj = this.getObject();
   if (!obj?.fP.length)
      return null;
   const box = { min: { x: 0, y: 0, z: 0 }, max: { x: 0, y: 0, z: 0 } };

   for (let k = 0; k < obj.fP.length; k += 3) {
      const x = obj.fP[k],
            y = obj.fP[k + 1],
            z = obj.fP[k + 2];
      if (k === 0) {
         box.min.x = box.max.x = x;
         box.min.y = box.max.y = y;
         box.min.z = box.max.z = z;
      } else {
         box.min.x = Math.min(x, box.min.x);
         box.max.x = Math.max(x, box.max.x);
         box.min.y = Math.min(y, box.min.y);
         box.max.y = Math.max(y, box.max.y);
         box.min.z = Math.min(z, box.min.z);
         box.max.z = Math.max(z, box.max.z);
      }
   }

   return box;
}


/** @summary direct draw function for TPolyMarker3D object (optionally with geo painter)
  * @private */
async function drawPolyMarker3D() {
   this.get3DBox = get3DBox;

   const fp = before3DDraw(this);

   if (!isObject(fp) || !fp.grx || !fp.gry || !fp.grz)
      return fp;

   this.$fp = fp;

   return drawPolyMarker3Dplain.bind(this)();
}

/** @summary Direct draw function for TPolyLine3D object
  * @desc Takes into account dashed properties
  * @private */
async function drawPolyLine3D() {
   this.get3DBox = get3DBox;

   const line = this.getObject(),
         fp = before3DDraw(this);

   if (!isObject(fp) || !fp.grx || !fp.gry || !fp.grz)
      return fp;

   const limit = 3*line.fN, p = line.fP, pnts = [];

   for (let n = 3; n < limit; n += 3) {
      pnts.push(fp.grx(p[n-3]), fp.gry(p[n-2]), fp.grz(p[n-1]),
                fp.grx(p[n]), fp.gry(p[n+1]), fp.grz(p[n+2]));
   }

   const lines = createLineSegments(pnts, create3DLineMaterial(this, line));

   fp.add3DMesh(lines, this, true);

   fp.render3D(100);

   return true;
}

export { drawPolyMarker3D, drawPolyLine3D };
