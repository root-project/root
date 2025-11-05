import { makeTranslate, DrawOptions, floatToString } from '../base/BasePainter.mjs';
import { ObjectPainter } from '../base/ObjectPainter.mjs';
import { drawObjectTitle } from '../hist/TPavePainter.mjs';
import { ensureTCanvas } from '../gpad/TCanvasPainter.mjs';
import { addMoveHandler } from '../gui/utils.mjs';
import { assignContextMenu } from '../gui/menu.mjs';


/**
 * @summary Painter for TBox class
 * @private
 */

class TPiePainter extends ObjectPainter {

   #rx; // recent rx
   #ry; // recent ry
   #slices; // recent slices
   #movex; // moving X coordinate
   #movey; // moving Y coordinate
   #angle0; // initial angle
   #offset0; // initial offset
   #slice; // moving slice
   #mode; // moving mode
   #padh; // pad height

   /** @summary Decode options */
   decodeOptions(opt) {
      const d = new DrawOptions(opt),
            o = this.getOptions();
      o.is3d = d.check('3D');
      o.lblor = 0;
      o.sort = 0;
      o.samecolor = false;
      o.same = d.check('SAME');
      if (d.check('SC'))
         o.samecolor = true; // around
      if (d.check('T'))
         o.lblor = 2; // around
      if (d.check('R'))
         o.lblor = 1; // along the radius
      if (d.check('>'))
         o.sort = 1;
      if (d.check('<'))
         o.sort = -1;
   }

   /** @summary start of drag handler
     * @private */
   moveStart(x, y) {
      if ((!x && !y) || !this.#slices || !this.#rx || !this.#ry)
         return;
      let angle = Math.atan2(y / this.#ry, x / this.#rx);

      while (angle < 0.5 * Math.PI)
         angle += 2 * Math.PI;

      const pie = this.getObject(),
            len = Math.sqrt((x / this.#rx) ** 2 + (y / this.#ry) ** 2),
            slice = this.#slices.find(elem => {
               return ((elem.a1 < angle) && (angle < elem.a2)) ||
                      ((elem.a1 < angle + 2 * Math.PI) && (angle + 2 * Math.PI < elem.a2));
            });

      // kind of cursor shown
      this.#mode = ((len > 0.95) && (x > this.#rx * 0.95) && this.options.is3d) ? 'n-resize' : ((slice && len < 0.7) ? 'grab' : 'w-resize');

      this.#movex = x;
      this.#movey = y;

      this.getG().style('cursor', this.#mode);

      if (this.#mode === 'grab') {
         this.#slice = slice.n;
         this.#angle0 = len;
         this.#offset0 = pie.fPieSlices[this.#slice].fRadiusOffset;
      } else if (this.#mode === 'n-resize') {
         this.#padh = this.getPadPainter().getPadHeight();
         this.#angle0 = pie.fAngle3D;
         this.#offset0 = y;
      } else {
         this.#angle0 = angle;
         this.#offset0 = pie.fAngularOffset;
      }
   }

   /** @summary drag handler
     * @private */
   moveDrag(dx, dy) {
      this.#movex += dx;
      this.#movey += dy;

      const pie = this.getObject();

      if (this.#mode === 'grab') {
         const len = Math.sqrt((this.#movex / this.#rx) ** 2 + (this.#movey / this.#ry) ** 2);
         pie.fPieSlices[this.#slice].fRadiusOffset = Math.max(0, this.#offset0 + 0.25 * (len - this.#angle0));
      } else if (this.#mode === 'n-resize')
         pie.fAngle3D = Math.max(5, Math.min(85, this.#angle0 + (this.#movey - this.#offset0) / this.#padh * 180));
      else {
         const angle = Math.atan2(this.#movey / this.#ry, this.#movex / this.#rx);
         pie.fAngularOffset = this.#offset0 - (angle - this.#angle0) / Math.PI * 180;
      }

      this.drawPie();
   }

   /** @summary end of drag handler
     * @private */
   moveEnd(not_changed) {
      if (not_changed)
         return;

      const pie = this.getObject();

      let exec;

      if (this.#mode === 'grab')
         exec = `SetEntryRadiusOffset(${this.#slice},${pie.fPieSlices[this.#slice].fRadiusOffset})`;
      else if (this.#mode === 'n-resize')
         exec = `SetAngle3D(${pie.fAngle3D})`;
      else
         exec = `SetAngularOffset(${pie.fAngularOffset})`;

      if (exec)
         this.submitCanvExec(exec + ';;Notify();;');

      this.#mode = null;

      this.getG().style('cursor', null);
   }

   /** @summary Update TPie object */
   updateObject(obj, opt) {
      if (!this.matchObjectType(obj))
         return false;

      this.decodeOptions(opt);

      Object.assign(this.getObject(), obj);

      return true;
   }

   /** @summary Redraw pie */
   async drawPie() {
      const maing = this.createG(),
            pie = this.getObject(),
            o = this.getOptions(),
            xc = this.axisToSvg('x', pie.fX),
            yc = this.axisToSvg('y', pie.fY),
            pp = this.getPadPainter(),
            radX = pie.fRadius;

      let radY = radX, pixelHeight = 1;

      if (o.is3d) {
         radY *= Math.sin(pie.fAngle3D / 180 * Math.PI);
         pixelHeight = this.axisToSvg('y', pie.fY - pie.fHeight) - yc;
      }

      maing.style('cursor', this.#mode || null);

      this.createAttText({ attr: pie });

      const rx = this.axisToSvg('x', pie.fX + radX) - xc,
            ry = this.axisToSvg('y', pie.fY - radY) - yc,
            dist_to_15pi = a => {
               while (a < 0.5 * Math.PI)
                  a += 2 * Math.PI;
               while (a >= 2.5 * Math.PI)
                  a -= 2 * Math.PI;
               return Math.abs(a - 1.5 * Math.PI);
            };

      makeTranslate(maing, xc, yc);

      const arr = [], promises = [];
      let total = 0, af = -pie.fAngularOffset / 180 * Math.PI;
      while (af < 2.5 * Math.PI)
         af += 2 * Math.PI;
      for (let n = 0; n < pie.fPieSlices.length; n++) {
         const value = pie.fPieSlices[n].fValue;
         total += value;
         arr.push({ n, value });
      }
      // sort in increase/decrease order
      if (o.sort !== 0)
         arr.sort((v1, v2) => { return o.sort * (v1.value - v2.value); });

      // now assign angles for each slice
      for (let n = 0; n < arr.length; n++) {
         const entry = arr[n];
         entry.a2 = af;
         af -= entry.value / total * 2 * Math.PI;
         entry.a1 = af;
         entry.a = dist_to_15pi((entry.a1 + entry.a2) / 2);
      }

      // sort for visualization in increasing order from Pi/2 angle
      arr.sort((v1, v2) => { return v1.a - v2.a; });

      for (let indx = 0; indx < arr.length; indx++) {
         const entry = arr[indx],
               slice = pie.fPieSlices[entry.n],
               g = maing.append('svg:g'),
               mid_angle = (entry.a1 + entry.a2) / 2;
         if (slice.fRadiusOffset) {
            const coef = radX > 0 ? slice.fRadiusOffset / radX : 0.1,
                  dx = Math.round(rx * coef * Math.cos(mid_angle)),
                  dy = Math.round(ry * coef * Math.sin(mid_angle));
            makeTranslate(g, dx, dy);
         }

         // Draw the slices
         const a1 = entry.a1, a2 = entry.a2,
               x1 = Math.round(rx * Math.cos(a1)),
               y1 = Math.round(ry * Math.sin(a1)),
               x2 = Math.round(rx * Math.cos(a2)),
               y2 = Math.round(ry * Math.sin(a2)),
               attline = this.createAttLine({ attr: slice, std: false }),
               attfill = this.createAttFill({ attr: slice, std: false });

         // paint pseudo-3d object
         if (o.is3d) {
            const add_curved_side = (aa1, aa2) => {
               if (dist_to_15pi((aa1 + aa2) / 2) < 0.5 * Math.PI)
                  return;
               const xx1 = Math.round(rx * Math.cos(aa1)),
                     yy1 = Math.round(ry * Math.sin(aa1)),
                     xx2 = Math.round(rx * Math.cos(aa2)),
                     yy2 = Math.round(ry * Math.sin(aa2));
               g.append('svg:path')
                .attr('d', `M${xx1},${yy1}a${rx},${ry},0,0,1,${xx2 - xx1},${yy2 - yy1}v${pixelHeight}a${rx},${ry},0,0,0,${xx1 - xx2},${yy1 - yy2}z`)
                .call(attline.func)
                .call(attfill.func);
            }, add_planar_side = (x, y) => {
               g.append('svg:path')
                .attr('d', `M0,0v${pixelHeight}l${x},${y}v${-pixelHeight}z`)
                .call(attline.func)
                .call(attfill.func);
            }, build_pie = func => {
               // use same segments for side and top/bottom curves
               let a = a1, border = 0;
               while (border <= a1)
                  border += Math.PI;
               while (a < a2) {
                  if (border >= a2) {
                     func(a, a2);
                     a = a2;
                  } else {
                     func(a, border);
                     a = border;
                     border += Math.PI;
                  }
               }
            };

            let pie_path = '';
            build_pie((aa1, aa2) => {
               const xx1 = Math.round(rx * Math.cos(aa1)),
                     yy1 = Math.round(ry * Math.sin(aa1)),
                     xx2 = Math.round(rx * Math.cos(aa2)),
                     yy2 = Math.round(ry * Math.sin(aa2));
               pie_path += `a${rx},${ry},0,0,1,${xx2 - xx1},${yy2 - yy1}`;
            });

            // bottom
            g.append('svg:path')
             .attr('d', `M0,${pixelHeight}l${x1},${y1}${pie_path}z`)
             .call(attline.func)
             .call(attfill.func);


            // planar
            if (dist_to_15pi(a1) > dist_to_15pi(a2)) {
               add_planar_side(x2, y2);
               add_planar_side(x1, y1);
            } else {
               add_planar_side(x1, y1);
               add_planar_side(x2, y2);
            }

            // curved
            build_pie(add_curved_side);

            // upper
            g.append('svg:path')
             .attr('d', `M0,0l${x1},${y1}${pie_path}z`)
             .call(attline.func)
             .call(attfill.func);
         } else {
            g.append('svg:path')
             .attr('d', `M0,0l${x1},${y1}a${rx},${ry},0,0,1,${x2 - x1},${y2 - y1}z`)
             .call(attline.func)
             .call(attfill.func);
         }

         const frac = total ? slice.fValue / total : 0;
         let tmptxt = pie.fLabelFormat;
         tmptxt = tmptxt.replaceAll('%txt', slice.fTitle);
         tmptxt = tmptxt.replaceAll('%val', floatToString(slice.fValue, pie.fValueFormat));
         tmptxt = tmptxt.replaceAll('%frac', floatToString(frac, pie.fFractionFormat));
         tmptxt = tmptxt.replaceAll('%perc', floatToString(frac * 100, pie.fPercentFormat) + '%');

         const arg = {
            draw_g: g,
            x: rx * (1 + pie.fLabelsOffset) * Math.cos(mid_angle),
            y: ry * (1 + pie.fLabelsOffset) * Math.sin(mid_angle),
            latex: 1,
            align: 22,
            text: tmptxt
         };

         if (o.samecolor)
            arg.color = this.getColor(slice.fFillColor);

         if (o.lblor === 1) {
            // radial positioning of the labels
            arg.rotate = Math.atan2(arg.y, arg.x) / Math.PI * 180;
            if (arg.x > 0)
               arg.align = 12;
            else {
               arg.align = 32;
               arg.rotate += 180;
            }
         } else if (o.lblor === 2) {
            // in the slice
            arg.rotate = Math.atan2(y2 - y1, x2 - x1) / Math.PI * 180;
            if ((arg.rotate > 90) || (arg.rotate < -90)) {
               arg.rotate += 180;
               arg.align = 21;
            } else
               arg.align = 23;
         } else if ((arg.x >= 0) && (arg.y >= 0)) {
            arg.align = 13;
            if (o.is3d)
               arg.y += pixelHeight;
         } else if ((arg.x > 0) && (arg.y < 0))
            arg.align = 11;
         else if ((arg.x < 0) && (arg.y >= 0)) {
            arg.align = 33;
            if (o.is3d)
               arg.y += pixelHeight;
         } else if ((arg.x < 0) && (arg.y < 0))
            arg.align = 31;

         const pr = this.startTextDrawingAsync(this.textatt.font, this.textatt.getSize(pp), g)
                        .then(() => this.drawText(arg)).then(() => this.finishTextDrawing(g));

         promises.push(pr);
      }

      return Promise.all(promises).then(() => {
         this.#rx = rx;
         this.#ry = ry;
         this.#slices = arr;
         return this;
      });
   }

   /** @summary Draw TPie title */
   async drawTitle(first_time) {
      return drawObjectTitle(this, first_time, !this.options.same, true);
   }

   /** @summary Redraw TPie object */
   async redraw() {
      return this.drawPie().then(() => this.drawTitle()).then(() => {
         assignContextMenu(this);
         addMoveHandler(this);
         return this;
      });
   }

   /** @summary Fill specific items */
   fillContextMenuItems(menu) {
      const pie = this.getObject();
      menu.add('Change title', () => menu.input('Enter new title', pie.fTitle).then(t => {
         pie.fTitle = t;
         this.interactiveRedraw('pad', `exec:SetTitle("${t}")`);
      }));
   }


   /** @summary Draw TPie object */
   static async draw(dom, obj, opt) {
      const painter = new TPiePainter(dom, obj, opt);
      painter.decodeOptions(opt);
      return ensureTCanvas(painter, false)
         .then(() => painter.drawPie())
         .then(() => painter.drawTitle(true))
         .then(() => {
            assignContextMenu(painter);
            addMoveHandler(painter);
            return painter;
         });
   }

} // class TPiePainter


export { TPiePainter };
