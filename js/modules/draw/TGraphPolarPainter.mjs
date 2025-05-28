import { settings, gStyle, create, BIT, clTPaveText, kTitle } from '../core.mjs';
import { scaleLinear, pointer as d3_pointer } from '../d3.mjs';
import { DrawOptions, buildSvgCurve, makeTranslate } from '../base/BasePainter.mjs';
import { ObjectPainter, getElementMainPainter } from '../base/ObjectPainter.mjs';
import { TPavePainter, kPosTitle } from '../hist/TPavePainter.mjs';
import { ensureTCanvas } from '../gpad/TCanvasPainter.mjs';
import { TooltipHandler } from '../gpad/TFramePainter.mjs';
import { assignContextMenu, kNoReorder } from '../gui/menu.mjs';

const kNoTitle = BIT(17);

/**
 * @summary Painter for TGraphPolargram objects.
 *
 * @private */

class TGraphPolargramPainter extends TooltipHandler {

   /** @summary Create painter
     * @param {object|string} dom - DOM element for drawing or element id
     * @param {object} polargram - object to draw */
   constructor(dom, polargram, opt) {
      super(dom, polargram, opt);
      this.$polargram = true; // indicate that this is polargram
      this.zoom_rmin = this.zoom_rmax = 0;
      this.t0 = 0;
      this.mult = 1;
      this.decodeOptions(opt);
      this.setTooltipEnabled(true);
   }

   /** @summary Returns true if fixed coordinates are configured */
   isNormalAngles() {
      const polar = this.getObject();
      return polar?.fRadian || polar?.fGrad || polar?.fDegree;
   }

   /** @summary Decode draw options */
   decodeOptions(opt) {
      const d = new DrawOptions(opt);

      this.setOptions({
         rdot: d.check('RDOT'),
         rangle: d.check('RANGLE', true) ? d.partAsInt() : 0,
         NoLabels: d.check('N'),
         OrthoLabels: d.check('O')
      });

      this.storeDrawOpt(opt);
   }

   /** @summary Set angles range displayed by the polargram */
   setAnglesRange(tmin, tmax, set_obj) {
      if (tmin >= tmax)
         tmax = tmin + 1;
      if (set_obj) {
         const polar = this.getObject();
         polar.fRwtmin = tmin;
         polar.fRwtmax = tmax;
      }
      this.t0 = tmin;
      this.mult = 2*Math.PI/(tmax - tmin);
   }

   /** @summary Translate coordinates */
   translate(input_angle, radius, keep_float) {
      // recalculate angle
      const angle = (input_angle - this.t0) * this.mult;
      let rx = this.r(radius),
          ry = rx/this.szx*this.szy,
          grx = rx * Math.cos(-angle),
          gry = ry * Math.sin(-angle);

      if (!keep_float) {
         grx = Math.round(grx);
         gry = Math.round(gry);
         rx = Math.round(rx);
         ry = Math.round(ry);
      }
      return { grx, gry, rx, ry };
   }

   /** @summary format label for radius ticks */
   format(radius) {
      if (radius === Math.round(radius)) return radius.toString();
      if (this.ndig > 10) return radius.toExponential(4);
      return radius.toFixed((this.ndig > 0) ? this.ndig : 0);
   }

   /** @summary Convert axis values to text */
   axisAsText(axis, value) {
      if (axis === 'r') {
         if (value === Math.round(value))
            return value.toString();
         if (this.ndig > 10)
            return value.toExponential(4);
         return value.toFixed(this.ndig+2);
      }

      value *= 180/Math.PI;
      return (value === Math.round(value)) ? value.toString() : value.toFixed(1);
   }

   /** @summary Returns coordinate of frame - without using frame itself */
   getFrameRect() {
      const pp = this.getPadPainter(),
            pad = pp.getRootPad(true),
            w = pp.getPadWidth(),
            h = pp.getPadHeight(),
            rect = {};

      if (pad) {
         rect.szx = Math.round(Math.max(0.1, 0.5 - Math.max(pad.fLeftMargin, pad.fRightMargin))*w);
         rect.szy = Math.round(Math.max(0.1, 0.5 - Math.max(pad.fBottomMargin, pad.fTopMargin))*h);
      } else {
         rect.szx = Math.round(0.5*w);
         rect.szy = Math.round(0.5*h);
      }

      rect.width = 2 * rect.szx;
      rect.height = 2 * rect.szy;
      rect.x = Math.round(w / 2 - rect.szx);
      rect.y = Math.round(h / 2 - rect.szy);

      rect.hint_delta_x = rect.szx;
      rect.hint_delta_y = rect.szy;

      rect.transform = makeTranslate(rect.x, rect.y) || '';

      return rect;
   }

   /** @summary Process mouse event */
   mouseEvent(kind, evnt) {
      let pnt = null;
      if (kind !== 'leave') {
         const pos = d3_pointer(evnt, this.getG()?.node());
         pnt = { x: pos[0], y: pos[1], touch: false };
      }
      this.processFrameTooltipEvent(pnt);
   }

   /** @summary Process mouse wheel event */
   mouseWheel(evnt) {
      evnt.stopPropagation();
      evnt.preventDefault();

      this.processFrameTooltipEvent(null); // remove all tooltips

      const polar = this.getObject();
      if (!polar) return;

      let delta = evnt.wheelDelta ? -evnt.wheelDelta : (evnt.deltaY || evnt.detail);
      if (!delta) return;

      delta = (delta < 0) ? -0.2 : 0.2;

      let rmin = this.scale_rmin, rmax = this.scale_rmax;
      const range = rmax - rmin;

      // rmin -= delta*range;
      rmax += delta*range;

      if ((rmin < polar.fRwrmin) || (rmax > polar.fRwrmax))
         rmin = rmax = 0;

      if ((this.zoom_rmin !== rmin) || (this.zoom_rmax !== rmax)) {
         this.zoom_rmin = rmin;
         this.zoom_rmax = rmax;
         this.redrawPad();
      }
   }

   /** @summary Process mouse double click event */
   mouseDoubleClick() {
      if (this.zoom_rmin || this.zoom_rmax) {
         this.zoom_rmin = this.zoom_rmax = 0;
         this.redrawPad();
      }
   }

   /** @summary Draw polargram polar labels */
   async drawPolarLabels(polar, nmajor) {
      const fontsize = Math.round(polar.fPolarTextSize * this.szy * 2),
            o = this.getOptions();

      return this.startTextDrawingAsync(polar.fPolarLabelFont, fontsize)
                 .then(() => {
         const lbls = (nmajor === 8) ? ['0', '#frac{#pi}{4}', '#frac{#pi}{2}', '#frac{3#pi}{4}', '#pi', '#frac{5#pi}{4}', '#frac{3#pi}{2}', '#frac{7#pi}{4}'] : ['0', '#frac{2#pi}{3}', '#frac{4#pi}{3}'],
               aligns = [12, 11, 21, 31, 32, 33, 23, 13];

         for (let n = 0; n < nmajor; ++n) {
            const angle = -n*2*Math.PI/nmajor;
            this.getG().append('svg:path')
               .attr('d', `M0,0L${Math.round(this.szx*Math.cos(angle))},${Math.round(this.szy*Math.sin(angle))}`)
               .call(this.lineatt.func);

            let align = 12, rotate = 0;

            if (o.OrthoLabels) {
               rotate = -n/nmajor*360;
               if ((rotate > -271) && (rotate < -91)) {
                  align = 32; rotate += 180;
               }
            } else {
               const aindx = Math.round(16 - angle/Math.PI*4) % 8; // index in align table, here absolute angle is important
               align = aligns[aindx];
            }

            this.drawText({ align, rotate,
                           x: Math.round((this.szx + fontsize)*Math.cos(angle)),
                           y: Math.round((this.szy + fontsize/this.szx*this.szy)*(Math.sin(angle))),
                           text: lbls[n],
                           color: this.getColor(polar.fPolarLabelColor), latex: 1 });
         }

         return this.finishTextDrawing();
      });
   }

   /** @summary Redraw polargram */
   async redraw() {
      if (!this.isMainPainter())
         return;

      const polar = this.getObject(),
            o = this.getOptions(),
            rect = this.getPadPainter().getFrameRect(),
            g = this.createG();

      makeTranslate(g, Math.round(rect.x + rect.width/2), Math.round(rect.y + rect.height/2));
      this.szx = rect.szx;
      this.szy = rect.szy;

      this.scale_rmin = polar.fRwrmin;
      this.scale_rmax = polar.fRwrmax;
      if (this.zoom_rmin !== this.zoom_rmax) {
         this.scale_rmin = this.zoom_rmin;
         this.scale_rmax = this.zoom_rmax;
      }

      this.r = scaleLinear().domain([this.scale_rmin, this.scale_rmax]).range([0, this.szx]);

      if (polar.fRadian) {
         polar.fRwtmin = 0; polar.fRwtmax = 2*Math.PI;
      } else if (polar.fDegree) {
         polar.fRwtmin = 0; polar.fRwtmax = 360;
      } else if (polar.fGrad) {
         polar.fRwtmin = 0; polar.fRwtmax = 200;
      }

      this.setAnglesRange(polar.fRwtmin, polar.fRwtmax);

      const ticks = this.r.ticks(5);
      let nminor = Math.floor((polar.fNdivRad % 10000) / 100),
          nmajor = polar.fNdivPol % 100;
      if (nmajor !== 3)
         nmajor = 8;

      this.createAttLine({ attr: polar });
      if (!this.gridatt)
         this.gridatt = this.createAttLine({ color: polar.fLineColor, style: 2, width: 1, std: false });

      const range = Math.abs(polar.fRwrmax - polar.fRwrmin);
      this.ndig = (range <= 0) ? -3 : Math.round(Math.log10(ticks.length / range));

      // verify that all radius labels are unique
      let lbls = [], indx = 0;
      while (indx<ticks.length) {
         const lbl = this.format(ticks[indx]);
         if (lbls.indexOf(lbl) >= 0) {
            if (++this.ndig>10) break;
            lbls = []; indx = 0; continue;
          }
         lbls.push(lbl);
         indx++;
      }

      let exclude_last = false;
      const pointer_events = this.isBatchMode() ? null : 'visibleFill';

      if ((ticks.at(-1) < polar.fRwrmax) && (this.zoom_rmin === this.zoom_rmax)) {
         ticks.push(polar.fRwrmax);
         exclude_last = true;
      }

      return this.startTextDrawingAsync(polar.fRadialLabelFont, Math.round(polar.fRadialTextSize * this.szy * 2)).then(() => {
         const axis_angle = - (o.rangle || polar.fAxisAngle) / 180 * Math.PI,
               ca = Math.cos(axis_angle),
               sa = Math.sin(axis_angle);
         for (let n = 0; n < ticks.length; ++n) {
            let rx = this.r(ticks[n]),
                ry = rx / this.szx * this.szy;
            g.append('ellipse')
             .attr('cx', 0)
             .attr('cy', 0)
             .attr('rx', Math.round(rx))
             .attr('ry', Math.round(ry))
             .style('fill', 'none')
             .style('pointer-events', pointer_events)
             .call(this.lineatt.func);

            if ((n < ticks.length - 1) || !exclude_last) {
               const halign = ca > 0.7 ? 1 : (ca > 0 ? 3 : (ca > -0.7 ? 1 : 3)),
                     valign = Math.abs(ca) < 0.7 ? 1 : 3;
               this.drawText({ align: 10 * halign + valign,
                               x: Math.round(rx*ca),
                               y: Math.round(ry*sa),
                               text: this.format(ticks[n]),
                               color: this.getColor(polar.fRadialLabelColor), latex: 0 });
               if (o.rdot) {
                  g.append('ellipse')
                   .attr('cx', Math.round(rx * ca))
                   .attr('cy', Math.round(ry * sa))
                   .attr('rx', 3)
                   .attr('ry', 3)
                   .style('fill', 'red');
               }
            }

            if ((nminor > 1) && ((n < ticks.length - 1) || !exclude_last)) {
               const dr = (ticks[1] - ticks[0]) / nminor;
               for (let nn = 1; nn < nminor; ++nn) {
                  const gridr = ticks[n] + dr*nn;
                  if (gridr > this.scale_rmax) break;
                  rx = this.r(gridr);
                  ry = rx / this.szx * this.szy;
                  g.append('ellipse')
                   .attr('cx', 0)
                   .attr('cy', 0)
                   .attr('rx', Math.round(rx))
                   .attr('ry', Math.round(ry))
                   .style('fill', 'none')
                   .style('pointer-events', pointer_events)
                   .call(this.gridatt.func);
               }
            }
         }

         if (ca < 0.999) {
            g.append('path')
             .attr('d', `M0,0L${Math.round(this.szx*ca)},${Math.round(this.szy*sa)}`)
             .style('pointer-events', pointer_events)
             .call(this.lineatt.func);
         }

         return this.finishTextDrawing();
      }).then(() => {
         return o.NoLabels ? true : this.drawPolarLabels(polar, nmajor);
      }).then(() => {
         nminor = Math.floor((polar.fNdivPol % 10000) / 100);

         if (nminor > 1) {
            for (let n = 0; n < nmajor * nminor; ++n) {
               if (n % nminor === 0) continue;
               const angle = -n*2*Math.PI/nmajor/nminor;
               g.append('svg:path')
                .attr('d', `M0,0L${Math.round(this.szx*Math.cos(angle))},${Math.round(this.szy*Math.sin(angle))}`)
                .call(this.gridatt.func);
            }
         }

         if (this.isBatchMode())
            return;

         assignContextMenu(this, kNoReorder);

         this.assignZoomHandler(g);
      });
   }

   /** @summary Fill TGraphPolargram context menu */
   fillContextMenuItems(menu) {
      const pp = this.getObject(), o = this.getOptions();
      menu.sub('Axis range');
      menu.addchk(pp.fRadian, 'Radian', flag => { pp.fRadian = flag; pp.fDegree = pp.fGrad = false; this.interactiveRedraw('pad', flag ? 'exec:SetToRadian()' : 'exec:SetTwoPi()'); }, 'Handle data angles as radian range 0..2*Pi');
      menu.addchk(pp.fDegree, 'Degree', flag => { pp.fDegree = flag; pp.fRadian = pp.fGrad = false; this.interactiveRedraw('pad', flag ? 'exec:SetToDegree()' : 'exec:SetTwoPi()'); }, 'Handle data angles as degree range 0..360');
      menu.addchk(pp.fGrad, 'Grad', flag => { pp.fGrad = flag; pp.fRadian = pp.fDegree = false; this.interactiveRedraw('pad', flag ? 'exec:SetToGrad()' : 'exec:SetTwoPi()'); }, 'Handle data angles as grad range 0..200');
      menu.endsub();
      menu.addSizeMenu('Axis angle', 0, 315, 45, o.rangle || pp.fAxisAngle, v => {
         o.rangle = pp.fAxisAngle = v;
         this.interactiveRedraw('pad', `exec:SetAxisAngle(${v})`);
      });
   }

   /** @summary Assign zoom handler to element
    * @private */
   assignZoomHandler(elem) {
      elem.on('mouseenter', evnt => this.mouseEvent('enter', evnt))
          .on('mousemove', evnt => this.mouseEvent('move', evnt))
          .on('mouseleave', evnt => this.mouseEvent('leave', evnt));

      if (settings.Zooming)
         elem.on('dblclick', evnt => this.mouseDoubleClick(evnt));

      if (settings.Zooming && settings.ZoomWheel)
         elem.on('wheel', evnt => this.mouseWheel(evnt));
   }

   /** @summary Draw TGraphPolargram */
   static async draw(dom, polargram, opt) {
      const main = getElementMainPainter(dom);
      if (main) {
         if (main.getObject() === polargram)
            return main;
         throw Error('Cannot superimpose TGraphPolargram with any other drawings');
      }

      const painter = new TGraphPolargramPainter(dom, polargram, opt);
      return ensureTCanvas(painter, false).then(() => {
         painter.setAsMainPainter();
         return painter.redraw();
      }).then(() => painter);
   }

} // class TGraphPolargramPainter


/**
 * @summary Painter for TGraphPolar objects.
 *
 * @private
 */

class TGraphPolarPainter extends ObjectPainter {

   /** @summary Decode options for drawing TGraphPolar */
   decodeOptions(opt) {
      const d = new DrawOptions(opt || 'L'),
            rdot = d.check('RDOT'),
            rangle = d.check('RANGLE', true) ? d.partAsInt() : 0,
            o = this.setOptions({
               mark: d.check('P'),
               err: d.check('E'),
               fill: d.check('F'),
               line: d.check('L'),
               curve: d.check('C'),
               radian: d.check('R'),
               degree: d.check('D'),
               grad: d.check('G'),
               Axis: d.check('N') ? 'N' : ''
            }, opt);

      if (d.check('O'))
         o.Axis += 'O';
      if (rdot)
         o.Axis += '_rdot';
      if (rangle)
         o.Axis += `_rangle${rangle}`;

      this.storeDrawOpt(opt);
   }

   /** @summary Update TGraphPolar with polargram */
   updateObject(obj, opt) {
      if (!this.matchObjectType(obj))
         return false;

      if (opt && (opt !== this.getOptions().original))
         this.decodeOptions(opt);

      if (this._draw_axis && obj.fPolargram)
         this.getMainPainter().updateObject(obj.fPolargram);

      delete obj.fPolargram;
      // copy all properties but not polargram
      Object.assign(this.getObject(), obj);
      return true;
   }

   /** @summary Redraw TGraphPolar */
   redraw() {
      return this.drawGraphPolar().then(() => this.updateTitle());
   }

   /** @summary Drawing TGraphPolar */
   async drawGraphPolar() {
      const graph = this.getObject(),
            o = this.getOptions(),
            main = this.getMainPainter();

      if (!graph || !main?.$polargram)
         return;

      if (o.mark)
         this.createAttMarker({ attr: graph });
      if (o.err || o.line || o.curve)
         this.createAttLine({ attr: graph });
      if (o.fill)
         this.createAttFill({ attr: graph });

      const g = this.createG();

      if (this._draw_axis && !main.isNormalAngles()) {
         const has_err = graph.fEX?.length;
         let rwtmin = graph.fX[0],
             rwtmax = graph.fX[0];
         for (let n = 0; n < graph.fNpoints; ++n) {
            rwtmin = Math.min(rwtmin, graph.fX[n] - (has_err ? graph.fEX[n] : 0));
            rwtmax = Math.max(rwtmax, graph.fX[n] + (has_err ? graph.fEX[n] : 0));
         }
         rwtmax += (rwtmax - rwtmin) / graph.fNpoints;
         main.setAnglesRange(rwtmin, rwtmax, true);
      }

      g.attr('transform', main.getG().attr('transform'));

      let mpath = '', epath = '';
      const bins = [], pointer_events = this.isBatchMode() ? null : 'visibleFill';

      for (let n = 0; n < graph.fNpoints; ++n) {
         if (graph.fY[n] > main.scale_rmax)
            continue;

         if (o.err) {
            const p1 = main.translate(graph.fX[n], graph.fY[n] - graph.fEY[n]),
                  p2 = main.translate(graph.fX[n], graph.fY[n] + graph.fEY[n]),
                  p3 = main.translate(graph.fX[n] + graph.fEX[n], graph.fY[n]),
                  p4 = main.translate(graph.fX[n] - graph.fEX[n], graph.fY[n]);

            epath += `M${p1.grx},${p1.gry}L${p2.grx},${p2.gry}` +
                     `M${p3.grx},${p3.gry}A${p4.rx},${p4.ry},0,0,1,${p4.grx},${p4.gry}`;
         }

         const pos = main.translate(graph.fX[n], graph.fY[n]);

         if (o.mark)
            mpath += this.markeratt.create(pos.grx, pos.gry);

         if (o.curve || o.line || o.fill)
            bins.push(pos);
      }

      if ((o.fill || o.line) && bins.length) {
         const lpath = buildSvgCurve(bins, { line: true });
         if (o.fill) {
            g.append('svg:path')
             .attr('d', lpath + 'Z')
             .style('pointer-events', pointer_events)
             .call(this.fillatt.func);
         }

         if (o.line) {
            g.append('svg:path')
             .attr('d', lpath)
             .style('fill', 'none')
             .style('pointer-events', pointer_events)
             .call(this.lineatt.func);
         }
      }

      if (o.curve && bins.length) {
         g.append('svg:path')
          .attr('d', buildSvgCurve(bins))
          .style('fill', 'none')
          .style('pointer-events', pointer_events)
          .call(this.lineatt.func);
      }

      if (epath) {
         g.append('svg:path')
          .attr('d', epath)
          .style('fill', 'none')
          .style('pointer-events', pointer_events)
          .call(this.lineatt.func);
      }

      if (mpath) {
         g.append('svg:path')
          .attr('d', mpath)
          .style('pointer-events', pointer_events)
          .call(this.markeratt.func);
      }

      if (!this.isBatchMode()) {
         assignContextMenu(this, kNoReorder);
         main.assignZoomHandler(g);
      }
   }

   /** @summary Create polargram object */
   createPolargram(gr) {
      const o = this.getOptions();
      if (!gr.fPolargram) {
         gr.fPolargram = create('TGraphPolargram');
         if (o.radian)
            gr.fPolargram.fRadian = true;
         else if (o.degree)
            gr.fPolargram.fDegree = true;
         else if (o.grad)
            gr.fPolargram.fGrad = true;
      }

      let rmin = gr.fY[0] || 0, rmax = rmin;
      const has_err = gr.fEY?.length;
      for (let n = 0; n < gr.fNpoints; ++n) {
         rmin = Math.min(rmin, gr.fY[n] - (has_err ? gr.fEY[n] : 0));
         rmax = Math.max(rmax, gr.fY[n] + (has_err ? gr.fEY[n] : 0));
      }

      gr.fPolargram.fRwrmin = rmin - (rmax-rmin)*0.1;
      gr.fPolargram.fRwrmax = rmax + (rmax-rmin)*0.1;

      return gr.fPolargram;
   }

   /** @summary Provide tooltip at specified point */
   extractTooltip(pnt) {
      if (!pnt) return null;

      const graph = this.getObject(),
            main = this.getMainPainter();
      let best_dist2 = 1e10, bestindx = -1, bestpos = null;

      for (let n = 0; n < graph.fNpoints; ++n) {
         const pos = main.translate(graph.fX[n], graph.fY[n]),
               dist2 = (pos.grx - pnt.x)**2 + (pos.gry - pnt.y)**2;
         if (dist2 < best_dist2) {
            best_dist2 = dist2;
            bestindx = n;
            bestpos = pos;
         }
      }

      let match_distance = 5;
      if (this.markeratt?.used)
         match_distance = this.markeratt.getFullSize();

      if (Math.sqrt(best_dist2) > match_distance)
         return null;

      const res = {
         name: this.getObject().fName, title: this.getObject().fTitle,
         x: bestpos.grx, y: bestpos.gry,
         color1: (this.markeratt?.used ? this.markeratt.color : undefined) ?? (this.fillatt?.used ? this.fillatt.color : undefined) ?? this.lineatt?.color,
         exact: Math.sqrt(best_dist2) < 4,
         lines: [this.getObjectHint()],
         binindx: bestindx,
         menu_dist: match_distance,
         radius: match_distance
      };

      res.lines.push(`r = ${main.axisAsText('r', graph.fY[bestindx])}`,
                     `phi = ${main.axisAsText('phi', graph.fX[bestindx])}`);

      if (graph.fEY && graph.fEY[bestindx])
         res.lines.push(`error r = ${main.axisAsText('r', graph.fEY[bestindx])}`);

      if (graph.fEX && graph.fEX[bestindx])
         res.lines.push(`error phi = ${main.axisAsText('phi', graph.fEX[bestindx])}`);

      return res;
   }

   /** @summary Only redraw histogram title
     * @return {Promise} with painter */
   async updateTitle() {
      // case when histogram drawn over other histogram (same option)
      if (!this._draw_axis)
         return this;

      const tpainter = this.getPadPainter()?.findPainterFor(null, kTitle, clTPaveText),
            pt = tpainter?.getObject();

      if (!tpainter || !pt)
         return this;

      const gr = this.getObject(),
            draw_title = !gr.TestBit(kNoTitle) && (gStyle.fOptTitle > 0);

      pt.Clear();
      if (draw_title) pt.AddText(gr.fTitle);
      return tpainter.redraw().then(() => this);
   }


   /** @summary Draw histogram title
     * @return {Promise} with painter */
   async drawTitle() {
      // case when histogram drawn over other histogram (same option)
      if (!this._draw_axis)
         return this;

      const gr = this.getObject(),
            st = gStyle,
            draw_title = !gr.TestBit(kNoTitle) && (st.fOptTitle > 0),
            pp = this.getPadPainter();

      let pt = pp.findInPrimitives(kTitle, clTPaveText);

      if (pt) {
         pt.Clear();
         if (draw_title)
            pt.AddText(gr.fTitle);
         return this;
      }

      pt = create(clTPaveText);
      Object.assign(pt, { fName: kTitle, fFillColor: st.fTitleColor, fFillStyle: st.fTitleStyle, fBorderSize: st.fTitleBorderSize,
                           fTextFont: st.fTitleFont, fTextSize: st.fTitleFontSize, fTextColor: st.fTitleTextColor, fTextAlign: 22 });

      if (draw_title)
         pt.AddText(gr.fTitle);
      return TPavePainter.draw(pp, pt, kPosTitle)
                         .then(p => { p?.setSecondaryId(this, kTitle); return this; });
   }

   /** @summary Show tooltip */
   showTooltip(hint) {
      let ttcircle = this.getG()?.selectChild('.tooltip_bin');

      if (!hint || !this.getG()) {
         ttcircle?.remove();
         return;
      }

      if (ttcircle.empty()) {
         ttcircle = this.getG().append('svg:ellipse')
                             .attr('class', 'tooltip_bin')
                             .style('pointer-events', 'none');
      }

      hint.changed = ttcircle.property('current_bin') !== hint.binindx;

      if (hint.changed) {
         ttcircle.attr('cx', hint.x)
               .attr('cy', hint.y)
               .attr('rx', Math.round(hint.radius))
               .attr('ry', Math.round(hint.radius))
               .style('fill', 'none')
               .style('stroke', hint.color1)
               .property('current_bin', hint.binindx);
      }
   }

   /** @summary Process tooltip event */
   processTooltipEvent(pnt) {
      const hint = this.extractTooltip(pnt);
      if (!pnt || !pnt.disabled)
         this.showTooltip(hint);
      return hint;
   }

   /** @summary Draw TGraphPolar */
   static async draw(dom, graph, opt) {
      const painter = new TGraphPolarPainter(dom, graph, opt);
      painter.decodeOptions(opt);

      const main = painter.getMainPainter();
      if (main && !main.$polargram) {
         console.error('Cannot superimpose TGraphPolar with plain histograms');
         return null;
      }

      let pr = Promise.resolve(null);
      if (!main) {
         // indicate that axis defined by this graph
         painter._draw_axis = true;
         pr = TGraphPolargramPainter.draw(dom, painter.createPolargram(graph), painter.options.Axis);
      }

      return pr.then(gram_painter => {
         gram_painter?.setSecondaryId(painter, 'polargram');
         painter.addToPadPrimitives();
         return painter.drawGraphPolar();
      }).then(() => painter.drawTitle());
   }

} // class TGraphPolarPainter

export { TGraphPolargramPainter, TGraphPolarPainter };
