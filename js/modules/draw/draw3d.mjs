import { isObject, isFunc } from '../core.mjs';
import { createLineSegments, create3DLineMaterial } from '../base/base3d.mjs';
import { drawDummy3DGeom } from '../geom/TGeoPainter.mjs';
import { drawPolyMarker3D as drawPolyMarker3Dplain } from './TPolyMarker3D.mjs';


/** @summary Prepare frame painter for 3D drawing
  * @private */
function before3DDraw(painter, obj) {
   let fp = painter.getFramePainter();

   if (!fp?.mode3d || !obj)
      return null;

   if (fp?.toplevel)
      return fp;

   let geop = painter.getMainPainter();
   if(!geop)
      return drawDummy3DGeom(painter);
   if (isFunc(geop.drawExtras))
      return geop.drawExtras(obj);

   return null;
}


/** @summary direct draw function for TPolyMarker3D object (optionally with geo painter)
  * @private */
async function drawPolyMarker3D() {

   let poly = this.getObject(),
       fp = before3DDraw(this, poly);

   if (!isObject(fp) || !fp.grx || !fp.gry || !fp.grz)
      return fp;

   this.$fp = fp;

   return drawPolyMarker3Dplain.bind(this)();
}

/** @summary Direct draw function for TPolyLine3D object
  * @desc Takes into account dashed properties
  * @private */
async function drawPolyLine3D() {
   let line = this.getObject(),
       fp = before3DDraw(this, line);

   if (!isObject(fp) || !fp.grx || !fp.gry || !fp.grz)
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

