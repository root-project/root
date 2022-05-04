import { gStyle, browser, settings, clone, create, isBatchMode } from '../core.mjs';

import { select as d3_select, rgb as d3_rgb, pointer as d3_pointer } from '../d3.mjs';

import { Prob } from '../base/math.mjs';

import { floatToString } from '../base/BasePainter.mjs';

import { getElementMainPainter, ObjectPainter } from '../base/ObjectPainter.mjs';

import { TAttLineHandler } from '../base/TAttLineHandler.mjs';

import { TAttMarkerHandler } from '../base/TAttMarkerHandler.mjs';

import { createMenu } from '../gui/menu.mjs';

import { TAxisPainter } from '../gpad/TAxisPainter.mjs';

import { addDragHandler } from '../gpad/TFramePainter.mjs';

import { ensureTCanvas } from '../gpad/TCanvasPainter.mjs';

/**
 * @summary painter for TPave-derived classes
 *
 * @private
 */

class TPavePainter extends ObjectPainter {

   /** @summary constructor
     * @param {object|string} dom - DOM element for drawing or element id
     * @param {object} pave - TPave-based object */
   constructor(dom, pave) {
      super(dom, pave);
      this.Enabled = true;
      this.UseContextMenu = true;
      this.UseTextColor = false; // indicates if text color used, enabled menu entry
   }

   /** @summary Draw pave and content */
   drawPave(arg) {

      this.UseTextColor = false;

      if (!this.Enabled) {
         this.removeG();
         return Promise.resolve(this);
      }

      let pt = this.getObject(), opt = pt.fOption.toUpperCase(), fp = this.getFramePainter();

      if (pt.fInit === 0) {
         this.stored = Object.assign({}, pt); // store coordinates to use them when updating
         pt.fInit = 1;
         let pad = this.getPadPainter().getRootPad(true);

         if ((pt._typename == "TPaletteAxis") && !pt.fX1 && !pt.fX2 && !pt.fY1 && !pt.fY2) {
            if (fp) {
               pt.fX1NDC = fp.fX2NDC + 0.01;
               pt.fX2NDC = Math.min(0.96, fp.fX2NDC + 0.06);
               pt.fY1NDC = fp.fY1NDC;
               pt.fY2NDC = fp.fY2NDC;
            } else {
               pt.fX2NDC = 0.8;
               pt.fX1NDC = 0.9;
               pt.fY1NDC = 0.1;
               pt.fY2NDC = 0.9;
            }
         } else if (opt.indexOf("NDC") >= 0) {
            pt.fX1NDC = pt.fX1; pt.fX2NDC = pt.fX2;
            pt.fY1NDC = pt.fY1; pt.fY2NDC = pt.fY2;
         } else if (pad && (pad.fX1 == 0) && (pad.fX2 == 1) && (pad.fY1 == 0) && (pad.fY2 == 1) && (typeof arg == "string") && (arg.indexOf('postpone') >= 0)) {
            // special case when pad not yet initialized
            pt.fInit = 0; // do not init until axes drawn
            pt.fX1NDC = pt.fY1NDC = 0.99;
            pt.fX2NDC = pt.fY2NDC = 1;
         } else if (pad) {
            if (pad.fLogx) {
               if (pt.fX1 > 0) pt.fX1 = Math.log10(pt.fX1);
               if (pt.fX2 > 0) pt.fX2 = Math.log10(pt.fX2);
            }
            if (pad.fLogy) {
               if (pt.fY1 > 0) pt.fY1 = Math.log10(pt.fY1);
               if (pt.fY2 > 0) pt.fY2 = Math.log10(pt.fY2);
            }
            pt.fX1NDC = (pt.fX1 - pad.fX1) / (pad.fX2 - pad.fX1);
            pt.fY1NDC = (pt.fY1 - pad.fY1) / (pad.fY2 - pad.fY1);
            pt.fX2NDC = (pt.fX2 - pad.fX1) / (pad.fX2 - pad.fX1);
            pt.fY2NDC = (pt.fY2 - pad.fY1) / (pad.fY2 - pad.fY1);

         } else {
            pt.fX1NDC = pt.fY1NDC = 0.1;
            pt.fX2NDC = pt.fY2NDC = 0.9;
         }

         if ((pt.fX1NDC == pt.fX2NDC) && (pt.fY1NDC == pt.fY2NDC) && (pt._typename == "TLegend")) {
            pt.fX1NDC = Math.max(pad ? pad.fLeftMargin : 0, pt.fX2NDC - 0.3);
            pt.fX2NDC = Math.min(pt.fX1NDC + 0.3, pad ? 1-pad.fRightMargin : 1);
            let h0 = Math.max(pt.fPrimitives ? pt.fPrimitives.arr.length*0.05 : 0, 0.2);
            pt.fY2NDC = Math.min(pad ? 1-pad.fTopMargin : 1, pt.fY1NDC + h0);
            pt.fY1NDC = Math.max(pt.fY2NDC - h0, pad ? pad.fBottomMargin : 0);
         }
      }

      let pad_rect = this.getPadPainter().getPadRect(),
          brd = pt.fBorderSize,
          dx = (opt.indexOf("L")>=0) ? -1 : ((opt.indexOf("R")>=0) ? 1 : 0),
          dy = (opt.indexOf("T")>=0) ? -1 : ((opt.indexOf("B")>=0) ? 1 : 0);

      // container used to recalculate coordinates
      this.createG();

      this._pave_x = Math.round(pt.fX1NDC * pad_rect.width);
      this._pave_y = Math.round((1.0 - pt.fY2NDC) * pad_rect.height);
      let width = Math.round((pt.fX2NDC - pt.fX1NDC) * pad_rect.width),
          height = Math.round((pt.fY2NDC - pt.fY1NDC) * pad_rect.height);

      this.draw_g.attr("transform", `translate(${this._pave_x},${this._pave_y})`);

      //if (!this.lineatt)
      //   this.lineatt = new TAttLineHandler(pt, brd>0 ? 1 : 0);

      this.createAttLine({ attr: pt, width: (brd > 0) ? pt.fLineWidth : 0 });

      this.createAttFill({ attr: pt });

      if (pt._typename == "TDiamond") {
         let h2 = Math.round(height/2), w2 = Math.round(width/2),
             dpath = `l${w2},${-h2}l${w2},${h2}l${-w2},${h2}z`;

         if ((brd > 1) && (pt.fShadowColor > 0) && (dx || dy) && !this.fillatt.empty())
            this.draw_g.append("svg:path")
                 .attr("d","M0,"+(h2+brd) + dpath)
                 .style("fill", this.getColor(pt.fShadowColor))
                 .style("stroke", this.getColor(pt.fShadowColor))
                 .style("stroke-width", "1px");

         this.draw_g.append("svg:path")
             .attr("d", "M0,"+h2 +dpath)
             .call(this.fillatt.func)
             .call(this.lineatt.func);

         let text_g = this.draw_g.append("svg:g")
                                 .attr("transform", `translate(${Math.round(width/4)},${Math.round(height/4)})`);

         return this.drawPaveText(w2, h2, arg, text_g);
      }

      // add shadow decoration before main rect
      if ((brd > 1) && (pt.fShadowColor > 0) && !pt.fNpaves && (dx || dy)) {
         let spath = "", scol = this.getColor(pt.fShadowColor);
         if (this.fillatt.empty()) {
            if ((dx < 0) && (dy < 0))
               spath = `M0,0v${height-brd}h${-brd}v${-height}h${width}v${brd}`;
            else // ((dx<0) && (dy>0))
               spath = `M0,${height}v${brd-height}h${-brd}v${height}h${width}v${-brd}`;
         } else {
            // when main is filled, one also can use fill for shadow to avoid complexity
            spath = `M${dx*brd},${dy*brd}v${height}h${width}v${-height}`;
         }
         this.draw_g.append("svg:path")
                    .attr("d", spath + "z")
                    .style("fill", scol)
                    .style("stroke", scol)
                    .style("stroke-width", "1px");
      }

      if (pt.fNpaves)
         for (let n = pt.fNpaves-1; n>0; --n)
            this.draw_g.append("svg:path")
               .attr("d", `M${dx*4*n},${dy*4*n}h${width}v${height}h${-width}z`)
               .call(this.fillatt.func)
               .call(this.lineatt.func);

      let rect;
      if (!isBatchMode() || !this.fillatt.empty() || !this.lineatt.empty())
           rect = this.draw_g.append("svg:path")
                      .attr("d", `M0,0H${width}V${height}H0Z`)
                      .call(this.fillatt.func)
                      .call(this.lineatt.func);

      let promise = this.paveDrawFunc ? this.paveDrawFunc(width, height, arg) : Promise.resolve(true);

      return promise.then(() => {

         if (isBatchMode() || (pt._typename === "TPave")) return this;

         // here all kind of interactive settings
         rect.style("pointer-events", "visibleFill")
             .on("mouseenter", () => this.showObjectStatus());

         addDragHandler(this, { obj: pt, x: this._pave_x, y: this._pave_y, width: width, height: height,
                                      minwidth: 10, minheight: 20, canselect: true,
                        redraw: () => { this.interactiveRedraw(false, "pave_moved"); this.drawPave(); },
                        ctxmenu: browser.touches && settings.ContextMenu && this.UseContextMenu });

         if (this.UseContextMenu && settings.ContextMenu)
             this.draw_g.on("contextmenu", evnt => this.paveContextMenu(evnt));

         if (pt._typename == "TPaletteAxis")
            this.interactivePaletteAxis(width, height);

         return this;
      });
   }

   /** @summary Fill option object used in TWebCanvas */
   fillWebObjectOptions(res) {
      if (!res) {
         if (!this.snapid) return null;
         res = { _typename: "TWebObjectOptions", snapid: this.snapid.toString(), opt: this.getDrawOpt(), fcust: "", fopt: [] };
      }

      let pave = this.getObject();

      if (pave && pave.fInit) {
         res.fcust = "pave";
         res.fopt = [pave.fX1NDC,pave.fY1NDC,pave.fX2NDC,pave.fY2NDC];
      }

      return res;
   }

   /** @summary draw TPaveLabel object */
   drawPaveLabel(width, height) {
      this.UseTextColor = true;

      let pave = this.getObject();

      this.startTextDrawing(pave.fTextFont, height/1.2);

      this.drawText({ align: pave.fTextAlign, width, height, text: pave.fLabel, color: this.getColor(pave.fTextColor) });

      return this.finishTextDrawing();
   }

   /** @summary draw TPaveStats object */
   drawPaveStats(width, height) {

      if (this.isStats()) this.fillStatistic();

      let pt = this.getObject(), lines = [],
          tcolor = this.getColor(pt.fTextColor),
          first_stat = 0, num_cols = 0, maxlen = 0;

      // now draw TLine and TBox objects
      for (let j = 0; j < pt.fLines.arr.length; ++j) {
         let entry = pt.fLines.arr[j];
         if ((entry._typename=="TText") || (entry._typename=="TLatex"))
            lines.push(entry.fTitle);
      }

      let nlines = lines.length;

      // adjust font size
      for (let j = 0; j < nlines; ++j) {
         let line = lines[j];
         if (j > 0) maxlen = Math.max(maxlen, line.length);
         if ((j == 0) || (line.indexOf('|') < 0)) continue;
         if (first_stat === 0) first_stat = j;
         let parts = line.split("|");
         if (parts.length > num_cols)
            num_cols = parts.length;
      }

      // for characters like 'p' or 'y' several more pixels required to stay in the box when drawn in last line
      let stepy = height / nlines, has_head = false, margin_x = pt.fMargin * width;

      this.startTextDrawing(pt.fTextFont, height/(nlines * 1.2));

      this.UseTextColor = true;

      if (nlines == 1) {
         this.drawText({ align: pt.fTextAlign, width, height, text: lines[0], color: tcolor, latex: 1 });
      } else
      for (let j = 0; j < nlines; ++j) {
         let posy = j*stepy;
         this.UseTextColor = true;

         if (first_stat && (j >= first_stat)) {
            let parts = lines[j].split("|");
            for (let n = 0; n < parts.length; ++n)
               this.drawText({ align: "middle", x: width * n / num_cols, y: posy, latex: 0,
                               width: width/num_cols, height: stepy, text: parts[n], color: tcolor });
         } else if (lines[j].indexOf('=') < 0) {
            if (j == 0) {
               has_head = true;
               let max_hlen = Math.max(maxlen, Math.round((width-2*margin_x)/stepy/0.65));
               if (lines[j].length > max_hlen + 5)
                  lines[j] = lines[j].slice(0,max_hlen+2) + "...";
            }
            this.drawText({ align: (j == 0) ? "middle" : "start", x: margin_x, y: posy,
                            width: width-2*margin_x, height: stepy, text: lines[j], color: tcolor });
         } else {
            let parts = lines[j].split("="), args = [];

            for (let n = 0; n < 2; ++n) {
               let arg = {
                  align: (n == 0) ? "start" : "end", x: margin_x, y: posy,
                  width: width-2*margin_x, height: stepy, text: parts[n], color: tcolor,
                  _expected_width: width-2*margin_x, _args: args,
                  post_process: function(painter) {
                    if (this._args[0].ready && this._args[1].ready)
                       painter.scaleTextDrawing(1.05*(this._args[0].result_width+this._args[1].result_width)/this._expected_width, painter.draw_g);
                  }
               };
               args.push(arg);
            }

            for (let n = 0; n < 2; ++n)
               this.drawText(args[n]);
         }
      }

      let lpath = "";

      if ((pt.fBorderSize > 0) && has_head)
         lpath += `M0,${Math.round(stepy)}h${width}`;

      if ((first_stat > 0) && (num_cols > 1)) {
         for (let nrow = first_stat; nrow < nlines; ++nrow)
            lpath += `M0,${Math.round(nrow * stepy)}h${width}`;
         for (let ncol = 0; ncol < num_cols - 1; ++ncol)
            lpath += `M${Math.round(width / num_cols * (ncol + 1))},${Math.round(first_stat * stepy)}V${height}`;
      }

      if (lpath) this.draw_g.append("svg:path").attr("d",lpath).call(this.lineatt.func);

      // this.draw_g.classed("most_upper_primitives", true); // this primitive will remain on top of list

      return this.finishTextDrawing(undefined, (nlines > 1));
   }

   /** @summary draw TPaveText object */
   drawPaveText(width, height, dummy_arg, text_g) {

      let pt = this.getObject(),
          tcolor = this.getColor(pt.fTextColor),
          nlines = 0, lines = [],
          pp = this.getPadPainter(),
          pad_height = pp.getPadHeight(),
          individual_positioning = false,
          draw_header = (pt.fLabel.length > 0),
          promises = [];

      if (!text_g) text_g = this.draw_g;

      // first check how many text lines in the list
      pt.fLines.arr.forEach(entry => {
         if ((entry._typename == "TText") || (entry._typename == "TLatex")) {
            nlines++; // count lines
            if ((entry.fX > 0) || (entry.fY > 0)) individual_positioning = true;
         }
      });

      let fast_draw = (nlines==1) && pp && pp._fast_drawing, nline = 0;

      // now draw TLine and TBox objects
      pt.fLines.arr.forEach(entry => {
         let ytext = (nlines > 0) ? Math.round((1-(nline-0.5)/nlines)*height) : 0;
         switch (entry._typename) {
            case "TText":
            case "TLatex":
               nline++; // just count line number
               if (individual_positioning) {
                  // each line should be drawn and scaled separately

                  let lx = entry.fX, ly = entry.fY;

                  lx = ((lx > 0) && (lx < 1)) ? Math.round(lx*width) : pt.fMargin * width;
                  ly = ((ly > 0) && (ly < 1)) ? Math.round((1-ly)*height) : ytext;

                  let jcolor = entry.fTextColor ? this.getColor(entry.fTextColor) : "";
                  if (!jcolor) {
                     jcolor = tcolor;
                     this.UseTextColor = true;
                  }

                  let sub_g = text_g.append("svg:g");

                  this.startTextDrawing(pt.fTextFont, (entry.fTextSize || pt.fTextSize) * pad_height, sub_g);

                  this.drawText({ align: entry.fTextAlign || pt.fTextAlign, x: lx, y: ly, text: entry.fTitle, color: jcolor,
                                  latex: (entry._typename == "TText") ? 0 : 1,  draw_g: sub_g, fast: fast_draw });

                  promises.push(this.finishTextDrawing(sub_g));
               } else {
                  lines.push(entry); // make as before
               }
               break;
            case "TLine":
            case "TBox":
               let lx1 = entry.fX1, lx2 = entry.fX2,
                   ly1 = entry.fY1, ly2 = entry.fY2;
               if (lx1!==0) lx1 = Math.round(lx1*width);
               lx2 = lx2 ? Math.round(lx2*width) : width;
               ly1 = ly1 ? Math.round((1-ly1)*height) : ytext;
               ly2 = ly2 ? Math.round((1-ly2)*height) : ytext;

               if (entry._typename == "TLine") {
                  let lineatt = new TAttLineHandler(entry);
                  text_g.append("svg:path")
                        .attr("d", `M${lx1},${ly1}L${lx2},${ly2}`)
                        .call(lineatt.func);
               } else {
                  let fillatt = this.createAttFill(entry);

                  text_g.append("svg:path")
                      .attr("d", `M${lx1},${ly1}H${lx2}V${ly2}H${lx1}Z`)
                      .call(fillatt.func);
               }
               break;
         }
      });

      if (!individual_positioning) {
         // for characters like 'p' or 'y' several more pixels required to stay in the box when drawn in last line
         let stepy = height / nlines, margin_x = pt.fMargin * width, max_font_size = 0;

         // for single line (typically title) limit font size
         if ((nlines == 1) && (pt.fTextSize > 0)) {
            max_font_size = Math.round(pt.fTextSize * pad_height);
            if (max_font_size < 3) max_font_size = 3;
         }

         this.startTextDrawing(pt.fTextFont, height/(nlines * 1.2), text_g, max_font_size);

         for (let j = 0; j < nlines; ++j) {
            let arg = null, lj = lines[j];

            if (nlines == 1) {
               arg = { x:0, y:0, width: width, height: height };
            } else {
               arg = { x: margin_x, y: j*stepy, width: width-2*margin_x, height: stepy };
               if (lj.fTextColor) arg.color = this.getColor(lj.fTextColor);
               if (lj.fTextSize) arg.font_size = Math.round(lj.fTextSize * pad_height);
            }

            arg.align = pt.fTextAlign;
            arg.draw_g = text_g;
            arg.latex = (lj._typename == "TText" ? 0 : 1);
            arg.text = lj.fTitle;
            arg.fast = fast_draw;
            if (!arg.color) { this.UseTextColor = true; arg.color = tcolor; }
            this.drawText(arg);
         }
         promises.push(this.finishTextDrawing(text_g, nlines > 1));
      }

      if (draw_header) {
         let x = Math.round(width*0.25),
             y = Math.round(-height*0.02),
             w = Math.round(width*0.5),
             h = Math.round(height*0.04),
             lbl_g = text_g.append("svg:g");

         lbl_g.append("svg:path")
               .attr("d", `M${x},${y}h${w}v${h}h${-w}z`)
               .call(this.fillatt.func)
               .call(this.lineatt.func);

         this.startTextDrawing(pt.fTextFont, h/1.5, lbl_g);

         this.drawText({ align: 22, x, y, width: w, height: h, text: pt.fLabel, color: tcolor, draw_g: lbl_g });

         promises.push(this.finishTextDrawing(lbl_g));

         this.UseTextColor = true;
      }

      return Promise.all(promises).then(() => { return this; });
   }

   /** @summary Method used to convert value to string according specified format
     * @desc format can be like 5.4g or 4.2e or 6.4f or "stat" or "fit" or "entries" */
   format(value, fmt) {
      if (!fmt) fmt = "stat";

      let pave = this.getObject();

      switch(fmt) {
         case "stat" : fmt = pave.fStatFormat || gStyle.fStatFormat; break;
         case "fit": fmt = pave.fFitFormat || gStyle.fFitFormat; break;
         case "entries": if ((Math.abs(value) < 1e9) && (Math.round(value) == value)) return value.toFixed(0); fmt = "14.7g"; break;
         case "last": fmt = this.lastformat; break;
      }

      let res = floatToString(value, fmt || "6.4g", true);

      this.lastformat = res[1];

      return res[0];
   }

   /** @summary Draw TLegend object */
   drawLegend(w, h) {

      let legend = this.getObject(),
          nlines = legend.fPrimitives.arr.length,
          ncols = legend.fNColumns,
          nrows = nlines;

      if (ncols < 2) {
         ncols = 1;
      } else {
         while ((nrows-1)*ncols >= nlines) nrows--;
      }

      const isEmpty = entry => !entry.fObject && !entry.fOption && (!entry.fLabel || (entry.fLabel == " "));

      if (ncols == 1)
         for (let ii = 0; ii < nlines; ++ii)
            if (isEmpty(legend.fPrimitives.arr[ii])) nrows--;

      if (nrows < 1) nrows = 1;

      let tcolor = this.getColor(legend.fTextColor),
          column_width = Math.round(w/ncols),
          padding_x = Math.round(0.03*w/ncols),
          padding_y = Math.round(0.03*h),
          step_y = (h - 2*padding_y)/nrows,
          font_size = 0.9*step_y,
          max_font_size = 0, // not limited in the beggining
          pp = this.getPadPainter(),
          ph = pp.getPadHeight(),
          any_opt = false, i = -1;

      if (legend.fTextSize && (ph*legend.fTextSize > 2) && (ph*legend.fTextSize < font_size))
         font_size = max_font_size = Math.round(ph*legend.fTextSize);

      this.startTextDrawing(legend.fTextFont, font_size, this.draw_g, max_font_size);

      for (let ii = 0; ii < nlines; ++ii) {
         let leg = legend.fPrimitives.arr[ii];

         if (isEmpty(leg)) continue; // let discard empty entry

         if (ncols==1) ++i; else i = ii;

         let lopt = leg.fOption.toLowerCase(),
             icol = i % ncols, irow = (i - icol) / ncols,
             x0 = icol * column_width,
             tpos_x = x0 + Math.round(legend.fMargin*column_width),
             mid_x = Math.round((x0 + tpos_x)/2),
             pos_y = Math.round(irow*step_y + padding_y), // top corner
             mid_y = Math.round((irow+0.5)*step_y + padding_y), // center line
             o_fill = leg, o_marker = leg, o_line = leg,
             mo = leg.fObject,
             painter = null, isany = false;

         const draw_fill = lopt.indexOf('f') != -1,
               draw_line = lopt.indexOf('l') != -1,
               draw_error = lopt.indexOf('e') != -1,
               draw_marker = lopt.indexOf('p') != -1;

         if ((mo !== null) && (typeof mo == 'object')) {
            if ('fLineColor' in mo) o_line = mo;
            if ('fFillColor' in mo) o_fill = mo;
            if ('fMarkerColor' in mo) o_marker = mo;

            painter = pp.findPainterFor(mo);
         }

         // Draw fill pattern (in a box)
         if (draw_fill) {
            let lineatt, fillatt = (painter && painter.fillatt) ? painter.fillatt : this.createAttFill(o_fill);
            if ((lopt.indexOf('l') < 0 && lopt.indexOf('e') < 0) && (lopt.indexOf('p') < 0)) {
               lineatt = (painter && painter.lineatt) ? painter.lineatt : new TAttLineHandler(o_line);
               if (lineatt.empty()) lineatt = null;
            }

            if (!fillatt.empty() || lineatt) {
                isany = true;
               // box total height is yspace*0.7
               // define x,y as the center of the symbol for this entry
               let rect = this.draw_g.append("svg:path")
                              .attr("d", `M${x0 + padding_x},${Math.round(pos_y+step_y*0.1)}v${Math.round(step_y*0.8)}h${tpos_x-2*padding_x-x0}v${-Math.round(step_y*0.8)}z`)
                              .call(fillatt.func);
                if (lineatt)
                   rect.call(lineatt.func);
            }
         }

         // Draw line and error (when specified)
         if (draw_line || draw_error) {
            let lineatt = (painter && painter.lineatt) ? painter.lineatt : new TAttLineHandler(o_line);
            if (!lineatt.empty()) {
               isany = true;
               this.draw_g.append("svg:path")
                  .attr("d", `M${x0 + padding_x},${mid_y}H${tpos_x - padding_x}`)
                  .call(lineatt.func);
               if (draw_error)
                  this.draw_g.append("svg:path")
                      .attr("d", `M${mid_x},${Math.round(pos_y+step_y*0.1)}V${Math.round(pos_y+step_y*0.9)}`)
                      .call(lineatt.func);
            }
         }

         // Draw Polymarker
         if (draw_marker) {
            let marker = (painter && painter.markeratt) ? painter.markeratt : new TAttMarkerHandler(o_marker);
            if (!marker.empty()) {
               isany = true;
               this.draw_g
                   .append("svg:path")
                   .attr("d", marker.create((x0 + tpos_x)/2, mid_y))
                   .call(marker.func);
            }
         }

         // special case - nothing draw, try to show rect with line attributes
         if (!isany && painter && painter.lineatt && !painter.lineatt.empty())
            this.draw_g.append("svg:path")
                       .attr("d", `M${x0 + padding_x},${Math.round(pos_y+step_y*0.1)}v${Math.round(step_y*0.8)}h${tpos_x-2*padding_x-x0}v${-Math.round(step_y*0.8)}z`)
                       .style("fill", "none")
                       .call(painter.lineatt.func);

         let pos_x = tpos_x;
         if (lopt.length > 0)
            any_opt = true;
         else if (!any_opt)
            pos_x = x0 + padding_x;

         if (leg.fLabel)
            this.drawText({ align: legend.fTextAlign, x: pos_x, y: pos_y, width: x0+column_width-pos_x-padding_x, height: step_y, text: leg.fLabel, color: tcolor });
      }

      // rescale after all entries are shown
      return this.finishTextDrawing();
   }

   /** @summary draw color palette with axis */
   drawPaletteAxis(s_width, s_height, arg) {

      let palette = this.getObject(),
          axis = palette.fAxis,
          can_move = (typeof arg == "string") && (arg.indexOf('can_move') >= 0),
          postpone_draw = (typeof arg == "string") && (arg.indexOf('postpone') >= 0),
          cjust = (typeof arg == "string") && (arg.indexOf('cjust') >= 0),
          width = this.getPadPainter().getPadWidth(),
          height = this.getPadPainter().getPadHeight(),
          pad = this.getPadPainter().getRootPad(true),
          main = palette.$main_painter || this.getMainPainter(),
          framep = this.getFramePainter(),
          zmin = 0, zmax = 100, gzmin, gzmax,
          contour = main.fContour,
          levels = contour ? contour.getLevels() : null,
          draw_palette = main.fPalette, axis_transform = "";

      this._palette_vertical = (palette.fX2NDC - palette.fX1NDC) < (palette.fY2NDC - palette.fY1NDC);

      axis.fTickSize = 0.6 * s_width / width; // adjust axis ticks size

      if (contour && framep) {
         if ((framep.zmin !== undefined) && (framep.zmax !== undefined) && (framep.zmin !== framep.zmax)) {
            gzmin = framep.zmin;
            gzmax = framep.zmax;
            zmin = framep.zoom_zmin;
            zmax = framep.zoom_zmax;
            if (zmin === zmax) { zmin = gzmin; zmax = gzmax; }
         } else {
            zmin = levels[0];
            zmax = levels[levels.length-1];
         }
         //zmin = Math.min(levels[0], framep.zmin);
         //zmax = Math.max(levels[levels.length-1], framep.zmax);
      } else if ((main.gmaxbin!==undefined) && (main.gminbin!==undefined)) {
         // this is case of TH2 (needs only for size adjustment)
         zmin = main.gminbin; zmax = main.gmaxbin;
      } else if ((main.hmin!==undefined) && (main.hmax!==undefined)) {
         // this is case of TH1
         zmin = main.hmin; zmax = main.hmax;
      }

      this.draw_g.selectAll("rect").style("fill", 'white');

      if ((gzmin === undefined) || (gzmax === undefined) || (gzmin == gzmax)) {
         gzmin = zmin; gzmax = zmax;
      }

      if (this._palette_vertical) {
         this._swap_side = palette.fX2NDC < 0.5;
         this.z_handle.configureAxis("zaxis", gzmin, gzmax, zmin, zmax, true, [0, s_height], { log: pad ? pad.fLogz : 0, fixed_ticks: cjust ? levels : null, max_tick_size: Math.round(s_width*0.7), swap_side: this._swap_side });
         axis_transform = this._swap_side ? "" : `translate(${s_width})`;
      } else {
         this._swap_side = palette.fY1NDC > 0.5;
         this.z_handle.configureAxis("zaxis", gzmin, gzmax, zmin, zmax, false, [0, s_width], { log: pad ? pad.fLogz : 0, fixed_ticks: cjust ? levels : null, max_tick_size: Math.round(s_height*0.7), swap_side: this._swap_side });
         axis_transform = this._swap_side ? "" : `translate(0,${s_height})`;
      }

      if (!contour || !draw_palette || postpone_draw)
         // we need such rect to correctly calculate size
         this.draw_g.append("svg:path")
                    .attr("d", `M0,0H${s_width}V${s_height}H0Z`)
                    .style("fill", 'white');
      else
         for (let i = 0; i < levels.length-1; ++i) {
            let z0 = Math.round(this.z_handle.gr(levels[i])),
                z1 = Math.round(this.z_handle.gr(levels[i+1])),
                lvl = (levels[i]+levels[i+1])/2, d;

            if (this._palette_vertical) {
               if ((z1 >= s_height) || (z0 < 0)) continue;
               z0 += 1; // ensure correct gap filling between colors

               if (z0 > s_height) {
                  z0 = s_height;
                  lvl = levels[i]*0.001+levels[i+1]*0.999;
               } else if (z1 < 0) {
                  z1 = 0;
                  lvl = levels[i]*0.999+levels[i+1]*0.001;
               }
               d = `M0,${z1}H${s_width}V${z0}H0Z`;
            } else {
               if ((z0 >= s_width) || (z1 < 0)) continue;
               z1 += 1; // ensure correct gap filling between colors

               if (z1 > s_width) {
                  z1 = s_width;
                  lvl = levels[i]*0.999+levels[i+1]*0.001;
               } else if (z0 < 0) {
                  z0 = 0;
                  lvl = levels[i]*0.001+levels[i+1]*0.999;
               }
               d = `M${z0},0V${s_height}H${z1}V0Z`;
            }

            let col = contour.getPaletteColor(draw_palette, lvl);
            if (!col) continue;

            let r = this.draw_g.append("svg:path")
                       .attr("d", d)
                       .style("fill", col)
                       .property("fill0", col)
                       .property("fill1", d3_rgb(col).darker(0.5).formatHex());

            if (this.isTooltipAllowed())
               r.on('mouseover', function() {
                  d3_select(this).transition().duration(100).style("fill", d3_select(this).property('fill1'));
               }).on('mouseout', function() {
                  d3_select(this).transition().duration(100).style("fill", d3_select(this).property('fill0'));
               }).append("svg:title").text(levels[i].toFixed(2) + " - " + levels[i+1].toFixed(2));

            if (settings.Zooming)
               r.on("dblclick", () => this.getFramePainter().unzoom("z"));
         }

      return this.z_handle.drawAxis(this.draw_g, s_width, s_height, axis_transform).then(() => {

         if (can_move && ('getBoundingClientRect' in this.draw_g.node())) {
            let rect = this.draw_g.node().getBoundingClientRect();

            if (this._palette_vertical) {
               let shift = (this._pave_x + parseInt(rect.width)) - Math.round(0.995*width) + 3;

               if (shift > 0) {
                  this._pave_x -= shift;
                  this.draw_g.attr("transform", `translate(${this._pave_x},${this._pave_y})`);
                  palette.fX1NDC -= shift/width;
                  palette.fX2NDC -= shift/width;
               }
            } else {
               let shift = Math.round((1.05 - gStyle.fTitleY)*height) - rect.y;
               if (shift > 0) {
                  this._pave_y += shift;
                  this.draw_g.attr("transform", `translate(${this._pave_x},${this._pave_y})`);
                  palette.fY1NDC -= shift/height;
                  palette.fY2NDC -= shift/height;
               }
            }
         }

         return this;
      });
   }

   /** @summary Add interactive methods for palette drawing */
   interactivePaletteAxis(s_width, s_height) {
      let doing_zoom = false, sel1 = 0, sel2 = 0, zoom_rect = null;

      const moveRectSel = evnt => {

         if (!doing_zoom) return;
         evnt.preventDefault();

         let m = d3_pointer(evnt, this.draw_g.node());
         if (this._palette_vertical) {
            sel2 = Math.min(Math.max(m[1], 0), s_height);
            zoom_rect.attr("y", Math.min(sel1, sel2))
                     .attr("height", Math.abs(sel2-sel1));
         } else {
            sel2 = Math.min(Math.max(m[0], 0), s_width);
            zoom_rect.attr("x", Math.min(sel1, sel2))
                     .attr("width", Math.abs(sel2-sel1));
         }
      }, endRectSel = evnt => {
         if (!doing_zoom) return;

         evnt.preventDefault();
         d3_select(window).on("mousemove.colzoomRect", null)
                          .on("mouseup.colzoomRect", null);
         zoom_rect.remove();
         zoom_rect = null;
         doing_zoom = false;

         let z = this.z_handle.gr, z1 = z.invert(sel1), z2 = z.invert(sel2);

         this.getFramePainter().zoom("z", Math.min(z1, z2), Math.max(z1, z2));
      }, startRectSel = evnt => {
         // ignore when touch selection is activated
         if (doing_zoom) return;
         doing_zoom = true;

         evnt.preventDefault();
         evnt.stopPropagation();

         let origin = d3_pointer(evnt, this.draw_g.node());

         zoom_rect = this.draw_g.append("svg:rect").attr("class", "zoom").attr("id", "colzoomRect");

         if (this._palette_vertical) {
            sel1 = sel2 = origin[1];
            zoom_rect.attr("x", "0")
                     .attr("width", s_width)
                     .attr("y", sel1)
                     .attr("height", 1);
         } else {
            sel1 = sel2 = origin[0];
            zoom_rect.attr("x", sel1)
                     .attr("width", 1)
                     .attr("y", 0)
                     .attr("height", s_height);
         }

         d3_select(window).on("mousemove.colzoomRect", moveRectSel)
                          .on("mouseup.colzoomRect", endRectSel, true);
      };

      if (settings.Zooming)
         this.draw_g.selectAll(".axis_zoom")
                    .on("mousedown", startRectSel)
                    .on("dblclick", () => this.getFramePainter().unzoom("z"));

      if (settings.ZoomWheel)
            this.draw_g.on("wheel", evnt => {
               let pos = d3_pointer(evnt, this.draw_g.node()),
                   coord = this._palette_vertical ? (1 - pos[1] / s_height) : pos[0] / s_width;

               let item = this.z_handle.analyzeWheelEvent(evnt, coord);
               if (item && item.changed)
                  this.getFramePainter().zoom("z", item.min, item.max);
            });
   }

   /** @summary Fill context menu for the TPave object */
   fillContextMenu(menu) {
      let pave = this.getObject();

      menu.add("header: " + pave._typename + "::" + pave.fName);
      if (this.isStats()) {
         menu.add("Default position", function() {
            pave.fX2NDC = gStyle.fStatX;
            pave.fX1NDC = pave.fX2NDC - gStyle.fStatW;
            pave.fY2NDC = gStyle.fStatY;
            pave.fY1NDC = pave.fY2NDC - gStyle.fStatH;
            pave.fInit = 1;
            this.interactiveRedraw(true, "pave_moved")
         });

         menu.add("SetStatFormat", () => {
            menu.input("Enter StatFormat", pave.fStatFormat).then(fmt => {
               if (!fmt) return;
               pave.fStatFormat = fmt;
               this.interactiveRedraw(true, `exec:SetStatFormat("${fmt}")`);
            });
         });
         menu.add("SetFitFormat", () => {
            menu.input("Enter FitFormat", pave.fFitFormat).then(fmt => {
               if (!fmt) return;
               pave.fFitFormat = fmt;
               this.interactiveRedraw(true, `exec:SetFitFormat("${fmt}")`);
            });
         });
         menu.add("separator");
         menu.add("sub:SetOptStat", () => {
            menu.input("Enter OptStat", pave.fOptStat, "int").then(fmt => {
               pave.fOptStat = fmt;
               this.interactiveRedraw(true, `exec:SetOptStat(${fmt}`);
            });
         });
         function AddStatOpt(pos, name) {
            let opt = (pos<10) ? pave.fOptStat : pave.fOptFit;
            opt = parseInt(parseInt(opt) / parseInt(Math.pow(10,pos % 10))) % 10;
            menu.addchk(opt, name, opt * 100 + pos, function(arg) {
               let newopt = (arg % 100 < 10) ? pave.fOptStat : pave.fOptFit;
               let oldopt = parseInt(arg / 100);
               newopt -= (oldopt>0 ? oldopt : -1) * parseInt(Math.pow(10, arg % 10));
               if (arg % 100 < 10) {
                  pave.fOptStat = newopt;
                  this.interactiveRedraw(true, `exec:SetOptStat(${newopt})`);
               } else {
                  pave.fOptFit = newopt;
                  this.interactiveRedraw(true, `exec:SetOptFit(${newopt})`);
               }
            });
         }

         AddStatOpt(0, "Histogram name");
         AddStatOpt(1, "Entries");
         AddStatOpt(2, "Mean");
         AddStatOpt(3, "Std Dev");
         AddStatOpt(4, "Underflow");
         AddStatOpt(5, "Overflow");
         AddStatOpt(6, "Integral");
         AddStatOpt(7, "Skewness");
         AddStatOpt(8, "Kurtosis");
         menu.add("endsub:");

         menu.add("sub:SetOptFit", () => {
            menu.input("Enter OptStat", pave.fOptFit, "int").then(fmt => {
               pave.fOptFit = fmt;
               this.interactiveRedraw(true, `exec:SetOptFit(${fmt})`);
            });
         });
         AddStatOpt(10, "Fit parameters");
         AddStatOpt(11, "Par errors");
         AddStatOpt(12, "Chi square / NDF");
         AddStatOpt(13, "Probability");
         menu.add("endsub:");

         menu.add("separator");
      } else if (pave.fName === "title")
         menu.add("Default position", function() {
            pave.fX1NDC = 0.28;
            pave.fY1NDC = 0.94;
            pave.fX2NDC = 0.72;
            pave.fY2NDC = 0.99;
            pave.fInit = 1;
            this.interactiveRedraw(true, "pave_moved");
         });

      if (this.UseTextColor)
         menu.addTextAttributesMenu(this);

      menu.addAttributesMenu(this);

      if ((menu.size() > 0) && this.showInspector('check'))
         menu.add('Inspect', this.showInspector);

      return menu.size() > 0;
   }

   /** @summary Show pave context menu */
   paveContextMenu(evnt) {
      if (this.z_handle) {
         let fp = this.getFramePainter();
         if (fp && fp.showContextMenu)
             fp.showContextMenu("z", evnt);
         return;
      }

      evnt.stopPropagation(); // disable main context menu
      evnt.preventDefault();  // disable browser context menu

      createMenu(evnt, this).then(menu => {
         this.fillContextMenu(menu);
         return this.fillObjectExecMenu(menu, "title");
       }).then(menu => menu.show());
   }

   /** @summary Returns true when stat box is drawn */
   isStats() {
      return this.matchObjectType('TPaveStats');
   }

   /** @summary Clear text in the pave */
   clearPave() {
      this.getObject().Clear();
   }

   /** @summary Add text to pave */
   addText(txt) {
      this.getObject().AddText(txt);
   }

   /** @summary Fill function parameters */
   fillFunctionStat(f1, dofit) {
      if (!dofit || !f1) return false;

      let print_fval    = dofit % 10,
          print_ferrors = Math.floor(dofit/10) % 10,
          print_fchi2   = Math.floor(dofit/100) % 10,
          print_fprob   = Math.floor(dofit/1000) % 10;

      if (print_fchi2 > 0)
         this.addText("#chi^2 / ndf = " + this.format(f1.fChisquare,"fit") + " / " + f1.fNDF);
      if (print_fprob > 0)
         this.addText("Prob = "  + this.format(Prob(f1.fChisquare, f1.fNDF)));
      if (print_fval > 0)
         for(let n = 0; n < f1.GetNumPars(); ++n) {
            let parname = f1.GetParName(n), parvalue = f1.GetParValue(n), parerr = f1.GetParError(n);

            parvalue = (parvalue===undefined) ? "<not avail>" : this.format(Number(parvalue),"fit");
            if (parerr !== undefined) {
               parerr = this.format(parerr,"last");
               if ((Number(parerr)===0) && (f1.GetParError(n) != 0))
                  parerr = this.format(f1.GetParError(n),"4.2g");
            }

            if ((print_ferrors > 0) && parerr)
               this.addText(parname + " = " + parvalue + " #pm " + parerr);
            else
               this.addText(parname + " = " + parvalue);
         }

      return true;
   }

   /** @summary Fill statistic */
   fillStatistic() {

      let pp = this.getPadPainter();
      if (pp && pp._fast_drawing) return false;

      let pave = this.getObject(),
          main = pave.$main_painter || this.getMainPainter();

      if (pave.fName !== "stats") return false;
      if (!main || (typeof main.fillStatistic !== 'function')) return false;

      let dostat = parseInt(pave.fOptStat), dofit = parseInt(pave.fOptFit);
      if (!Number.isInteger(dostat)) dostat = gStyle.fOptStat;
      if (!Number.isInteger(dofit)) dofit = gStyle.fOptFit;

      // we take statistic from main painter
      if (!main.fillStatistic(this, dostat, dofit)) return false;

      // adjust the size of the stats box with the number of lines
      let nlines = pave.fLines.arr.length,
          stath = nlines * gStyle.StatFontSize;
      if ((stath <= 0) || (gStyle.StatFont % 10 === 3)) {
         stath = 0.25 * nlines * gStyle.StatH;
         pave.fY1NDC = pave.fY2NDC - stath;
      }

      return true;
   }

   /** @summary Is dummy pos of the pave painter */
   isDummyPos(p) {
      if (!p) return true;

      return !p.fInit && !p.fX1 && !p.fX2 && !p.fY1 && !p.fY2 && !p.fX1NDC && !p.fX2NDC && !p.fY1NDC && !p.fY2NDC;
   }

   /** @summary Update TPave object  */
   updateObject(obj) {
      if (!this.matchObjectType(obj)) return false;

      let pave = this.getObject();

      if (!pave.modified_NDC && !this.isDummyPos(obj)) {
         // if position was not modified interactively, update from source object

         if (this.stored && !obj.fInit && (this.stored.fX1 == obj.fX1)
             && (this.stored.fX2 == obj.fX2) && (this.stored.fY1 == obj.fY1) && (this.stored.fY2 == obj.fY2)) {
            // case when source object not initialized and original coordinates are not changed
            // take over only modified NDC coordinate, used in tutorials/graphics/canvas.C
            if (this.stored.fX1NDC != obj.fX1NDC) pave.fX1NDC = obj.fX1NDC;
            if (this.stored.fX2NDC != obj.fX2NDC) pave.fX2NDC = obj.fX2NDC;
            if (this.stored.fY1NDC != obj.fY1NDC) pave.fY1NDC = obj.fY1NDC;
            if (this.stored.fY2NDC != obj.fY2NDC) pave.fY2NDC = obj.fY2NDC;
         } else {
            pave.fInit = obj.fInit;
            pave.fX1 = obj.fX1; pave.fX2 = obj.fX2;
            pave.fY1 = obj.fY1; pave.fY2 = obj.fY2;
            pave.fX1NDC = obj.fX1NDC; pave.fX2NDC = obj.fX2NDC;
            pave.fY1NDC = obj.fY1NDC; pave.fY2NDC = obj.fY2NDC;
         }

         this.stored = Object.assign({}, obj); // store latest coordinates
      }

      pave.fOption = obj.fOption;
      pave.fBorderSize = obj.fBorderSize;

      switch (obj._typename) {
         case 'TPaveText':
            pave.fLines = clone(obj.fLines);
            return true;
         case 'TPavesText':
            pave.fLines = clone(obj.fLines);
            pave.fNpaves = obj.fNpaves;
            return true;
         case 'TPaveLabel':
            pave.fLabel = obj.fLabel;
            return true;
         case 'TPaveStats':
            pave.fOptStat = obj.fOptStat;
            pave.fOptFit = obj.fOptFit;
            return true;
         case 'TLegend':
            let oldprim = pave.fPrimitives;
            pave.fPrimitives = obj.fPrimitives;
            pave.fNColumns = obj.fNColumns;
            if (oldprim && oldprim.arr && pave.fPrimitives && pave.fPrimitives.arr && (oldprim.arr.length == pave.fPrimitives.arr.length)) {
               // try to sync object reference, new object does not displayed automatically
               // in ideal case one should use snapids in the entries
               for (let k = 0; k < oldprim.arr.length; ++k) {
                  let oldobj = oldprim.arr[k].fObject, newobj = pave.fPrimitives.arr[k].fObject;

                  if (oldobj && newobj && oldobj._typename == newobj._typename && oldobj.fName == newobj.fName)
                     pave.fPrimitives.arr[k].fObject = oldobj;
               }
            }
            return true;
         case 'TPaletteAxis':
            pave.fBorderSize = 1;
            pave.fShadowColor = 0;
            return true;
      }

      return false;
   }

   /** @summary redraw pave object */
   redraw() {
      return this.drawPave();
   }

   /** @summary cleanup pave painter */
   cleanup() {
      if (this.z_handle) {
         this.z_handle.cleanup();
         delete this.z_handle;
      }

      super.cleanup();
   }

   /** @summary Returns true if object is supported */
   static canDraw(obj) {
      let typ = obj?._typename;
      return typ == "TPave" || typ == "TPaveLabel" || typ == "TPaveStats" || typ == "TPaveText"
             || typ == "TPavesText" || typ == "TDiamond" || typ == "TLegend" || typ == "TPaletteAxis";
   }

   /** @summary Draw TPave */
   static draw(dom, pave, opt) {
      let painter = new TPavePainter(dom, pave);

      return ensureTCanvas(painter, false).then(() => {

         if ((pave.fName === "title") && (pave._typename === "TPaveText")) {
            let tpainter = painter.getPadPainter().findPainterFor(null, "title");
            if (tpainter && (tpainter !== painter)) {
               tpainter.removeFromPadPrimitives();
               tpainter.cleanup();
            } else if ((opt == "postitle") || painter.isDummyPos(pave)) {
               let st = gStyle, fp = painter.getFramePainter();
               if (st && fp) {
                  let midx = st.fTitleX, y2 = st.fTitleY, w = st.fTitleW, h = st.fTitleH;
                  if (!h) h = (y2-fp.fY2NDC)*0.7;
                  if (!w) w = fp.fX2NDC - fp.fX1NDC;
                  if (!Number.isFinite(h) || (h <= 0)) h = 0.06;
                  if (!Number.isFinite(w) || (w <= 0)) w = 0.44;
                  pave.fX1NDC = midx - w/2;
                  pave.fY1NDC = y2 - h;
                  pave.fX2NDC = midx + w/2;
                  pave.fY2NDC = y2;
                  pave.fInit = 1;
               }
            }
         } else if (pave._typename === "TPaletteAxis") {
            pave.fBorderSize = 1;
            pave.fShadowColor = 0;

            // check some default values of TGaxis object, otherwise axis will not be drawn
            if (pave.fAxis) {
               if (!pave.fAxis.fChopt) pave.fAxis.fChopt = "+";
               if (!pave.fAxis.fNdiv) pave.fAxis.fNdiv = 12;
               if (!pave.fAxis.fLabelOffset) pave.fAxis.fLabelOffset = 0.005;
            }

            painter.z_handle = new TAxisPainter(dom, pave.fAxis, true);
            painter.z_handle.setPadName(painter.getPadName());

            painter.UseContextMenu = true;
         }

         switch (pave._typename) {
            case "TPaveLabel":
               painter.paveDrawFunc = painter.drawPaveLabel;
               break;
            case "TPaveStats":
               painter.paveDrawFunc = painter.drawPaveStats;
               painter.$secondary = true; // indicates that painter created from others
               break;
            case "TPaveText":
            case "TPavesText":
            case "TDiamond":
               painter.paveDrawFunc = painter.drawPaveText;
               break;
            case "TLegend":
               painter.paveDrawFunc = painter.drawLegend;
               break;
            case "TPaletteAxis":
               painter.paveDrawFunc = painter.drawPaletteAxis;
               break;
         }

         return painter.drawPave(opt);
      });
   }

} // TPavePainter

/** @summary Produce and draw TLegend object for the specified dom
  * @desc Should be called when all other objects are painted
  * Invoked when item "$legend" specified in url string
  * @returns {Object} Promise with TLegend painter
  * @private */
function produceLegend(dom, opt) {
   let main_painter = getElementMainPainter(dom),
       pp = main_painter ? main_painter.getPadPainter() : null,
       pad = pp ? pp.getRootPad(true) : null;
   if (!pad) return Promise.resolve(null);

   let leg = create("TLegend");

   for (let k = 0; k < pp.painters.length; ++k) {
      let painter = pp.painters[k],
          obj = painter.getObject();

      if (!obj) continue;

      let entry = create("TLegendEntry");
      entry.fObject = obj;
      entry.fLabel = (opt == "all") ? obj.fName : painter.getItemName();
      entry.fOption = "";
      if (!entry.fLabel) continue;

      if (painter.lineatt && painter.lineatt.used) entry.fOption+="l";
      if (painter.fillatt && painter.fillatt.used) entry.fOption+="f";
      if (painter.markeratt && painter.markeratt.used) entry.fOption+="m";
      if (!entry.fOption) entry.fOption = "l";

      leg.fPrimitives.Add(entry);
   }

   // no entries - no need to draw legend
   let szx = 0.4, szy = leg.fPrimitives.arr.length;
   if (!szy) return;
   if (szy > 8) szy = 8;
   szy *= 0.1;

   leg.fX1NDC = szx*pad.fLeftMargin + (1-szx)*(1-pad.fRightMargin);
   leg.fY1NDC = (1-szy)*(1-pad.fTopMargin) + szy*pad.fBottomMargin;
   leg.fX2NDC = 0.99-pad.fRightMargin;
   leg.fY2NDC = 0.99-pad.fTopMargin;
   leg.fFillStyle = 1001;

   return TPavePainter.draw(dom, leg);
}

export { TPavePainter, produceLegend };
