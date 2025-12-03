import { isFunc } from '../core.mjs';
import { makeTranslate } from '../base/BasePainter.mjs';
import { TTextPainter } from './TTextPainter.mjs';
import { build3dlatex } from '../base/latex3d.mjs';
import { ensureTCanvas } from '../gpad/TCanvasPainter.mjs';

function getRotation(camera, mesh) {
   const dx = camera.position.x - mesh.position.x,
         dy = camera.position.y - mesh.position.y;
   return Math.atan2(dy, dx) + Math.PI / 2;
}

class TAnnotation3DPainter extends TTextPainter {

   /** @summary Redraw annotation
    * @desc handle 3d and 2d mode */

   async redraw() {
      const fp = this.getFramePainter(),
            text = this.getObject();

      if (fp?.mode3d && !this.use_2d) {
         const mesh = build3dlatex(text, '', this, fp);
         mesh.traverse(o => o.geometry?.rotateX(Math.PI / 2));
         mesh.position.set(fp.grx(text.fX), fp.gry(text.fY), fp.grz(text.fZ));
         mesh.rotation.set(0, 0, getRotation(fp.camera, mesh));
         fp.processRender3D = true;
         fp.add3DMesh(mesh, this, true);
         fp.render3D(100);
         return this;
      }

      const mode = fp?.mode3d && isFunc(fp?.convert3DtoPadNDC) ? '3d' : '2d';
      let x = text.fX, y = text.fY;

      if (mode === '3d') {
         const pos = fp.convert3DtoPadNDC(text.fX, text.fY, text.fZ);
         x = pos.x;
         y = pos.y;
      }

      return this._redrawText(x, y, mode).then(() => {
         fp.processRender3D = mode === '3d';
         return this;
      });
   }

   /** @summary Extra handling during 3d rendering
     * @desc Allows to reposition annotation when rotate/zoom drawing */
   handleRender3D() {
      const text = this.getObject(),
            fp = this.getFramePainter();
      if (this.use_2d) {
         const pos = fp.convert3DtoPadNDC(text.fX, text.fY, text.fZ),
               new_x = this.axisToSvg('x', pos.x, true),
               new_y = this.axisToSvg('y', pos.y, true);
         makeTranslate(this.getG(), new_x - this.pos_x, new_y - this.pos_y);
      } else
         fp.get3DMeshes(this).forEach(mesh => mesh.rotation.set(0, 0, getRotation(fp.camera, mesh)));
   }

   /** @summary draw TAnnotation3D object */
   static async draw(dom, obj, opt) {
      const painter = new TAnnotation3DPainter(dom, obj, opt);
      painter.use_2d = (opt === '2d') || (opt === '2D');
      return ensureTCanvas(painter, painter.use_2d ? true : '3d').then(() => painter.redraw());
   }

} // class TAnnotation3DPainter


export { TAnnotation3DPainter };
