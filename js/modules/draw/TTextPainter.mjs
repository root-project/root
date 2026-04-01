import { clTLink, clTLatex, clTMathText, BIT } from '../core.mjs';
import { ObjectPainter } from '../base/ObjectPainter.mjs';
import { makeTranslate, DrawOptions } from '../base/BasePainter.mjs';
import { addMoveHandler } from '../gui/utils.mjs';
import { ensureTCanvas } from '../gpad/TCanvasPainter.mjs';
import { assignContextMenu } from '../gui/menu.mjs';


const kTextNDC = BIT(14);

class TTextPainter extends ObjectPainter {

   async _redrawText(x, y, annot) {
      const text = this.getObject(),
            pp = this.getPadPainter(),
            fp = this.getFramePainter(),
            is_url = text.fName.startsWith('http://') || text.fName.startsWith('https://');

      // special handling of dummy frame painter
      if (fp?.getDrawDom() === null)
         return this;

      let fact = 1, use_frame = false;

      this.createAttText({ attr: text });

      if ((annot === '3d') || text.TestBit(kTextNDC))
         this.isndc = true;
      else if (!annot && pp?.getRootPad(true)) {
         // force pad coordinates
         const d = new DrawOptions(this.getDrawOpt());
         use_frame = d.check('FRAME');
      } else if (!annot) {
         // place in the middle
         this.isndc = true;
         x = y = 0.5;
         text.fTextAlign = 22;
      }

      const g = this.createG(use_frame ? 'frame2d' : undefined, is_url);

      g.attr('transform', null); // remove transform from interactive changes

      x = this.axisToSvg('x', x, this.isndc);
      y = this.axisToSvg('y', y, this.isndc);
      this.swap_xy = use_frame && fp?.swap_xy();

      if (this.swap_xy)
         [x, y] = [y, x];

      const arg = this.textatt.createArg({ x, y, text: text.fTitle, latex: 0 });

      if (this.matchObjectType(clTLatex) || annot)
         arg.latex = 1;
      else if (this.matchObjectType(clTMathText)) {
         arg.latex = 2;
         fact = 0.8;
      }

      if (is_url) {
         g.attr('href', text.fName);
         if (!this.isBatchMode())
            g.append('svg:title').text(`link on ${text.fName}`);
      }

      return this.startTextDrawingAsync(this.textatt.font, this.textatt.getSize(pp, fact))
               .then(() => this.drawText(arg))
               .then(() => this.finishTextDrawing())
               .then(() => {
                  if (this.isBatchMode())
                     return this;

                  if (pp.isButton() && !pp.isEditable()) {
                     g.on('click', () => this.getCanvPainter().selectActivePad(pp));
                     return this;
                  }

                  Object.assign(this, { pos_x: x, pos_y: y, pos_dx: 0, pos_dy: 0 });

                  if (annot !== '3d')
                     addMoveHandler(this, true, is_url);

                  assignContextMenu(this);

                  if (this.matchObjectType(clTLink))
                     g.style('cursor', 'pointer').on('click', () => this.submitCanvExec('ExecuteEvent(kButton1Up, 0, 0);;'));

                  return this;
               });
   }

   async redraw() {
      return this._redrawText(this.getObject().fX, this.getObject().fY);
   }

   moveDrag(dx, dy) {
      this.pos_dx += dx;
      this.pos_dy += dy;
      makeTranslate(this.getG(), this.pos_dx, this.pos_dy);
   }

   moveEnd(not_changed) {
      if (not_changed)
         return;
      const txt = this.getObject();
      let fx = this.svgToAxis('x', this.pos_x + this.pos_dx, this.isndc),
          fy = this.svgToAxis('y', this.pos_y + this.pos_dy, this.isndc);
      if (this.swap_xy)
         [fx, fy] = [fy, fx];

      txt.fX = fx;
      txt.fY = fy;
      this.submitCanvExec(`SetX(${fx});;SetY(${fy});;`);
   }

   fillContextMenuItems(menu) {
      const text = this.getObject();
      menu.add('Change text', () => menu.input('Enter new text', text.fTitle).then(t => {
         text.fTitle = t;
         this.interactiveRedraw('pad', `exec:SetTitle("${t}")`);
      }));
   }

   /** @summary draw TText-derived object */
   static async draw(dom, obj, opt) {
      const painter = new TTextPainter(dom, obj, opt);
      return ensureTCanvas(painter, false).then(() => painter.redraw());
   }

} // class TTextPainter


export { TTextPainter };
