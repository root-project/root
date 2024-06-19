import { gStyle, browser, settings, clone, isObject, isFunc, isStr, BIT,
         clTPave, clTPaveText, clTPavesText, clTPaveStats, clTPaveLabel, clTPaveClass, clTDiamond, clTLegend, clTPaletteAxis,
         clTText, clTLatex, clTLine, clTBox, kTitle } from '../core.mjs';
import { select as d3_select, rgb as d3_rgb, pointer as d3_pointer } from '../d3.mjs';
import { Prob } from '../base/math.mjs';
import { floatToString, makeTranslate, compressSVG, svgToImage, addHighlightStyle } from '../base/BasePainter.mjs';
import { ObjectPainter, EAxisBits } from '../base/ObjectPainter.mjs';
import { showPainterMenu } from '../gui/menu.mjs';
import { TAxisPainter } from '../gpad/TAxisPainter.mjs';
import { addDragHandler } from '../gpad/TFramePainter.mjs';
import { ensureTCanvas } from '../gpad/TCanvasPainter.mjs';

const kTakeStyle = BIT(17);

/** @summary Returns true if stat box on default place and can be adjusted
  * @private */
function isDefaultStatPosition(pt) {
   const test = (v1, v2) => (Math.abs(v1-v2) < 1e-3);
   return test(pt.fX1NDC, gStyle.fStatX - gStyle.fStatW) &&
          test(pt.fY1NDC, gStyle.fStatY - gStyle.fStatH) &&
          test(pt.fX2NDC, gStyle.fStatX) &&
          test(pt.fY2NDC, gStyle.fStatY);
}

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
   }

   /** @summary Autoplace legend on the frame
     * @return {Promise} with boolean flag if position was changed  */
   async autoPlaceLegend(pt, pad, keep_origin) {
      const main_svg = this.getFrameSvg().selectChild('.main_layer');

      let svg_code = main_svg.node().outerHTML;

      svg_code = compressSVG(svg_code);

      svg_code = '<svg xmlns="http://www.w3.org/2000/svg"' + svg_code.slice(4);

      const lm = pad?.fLeftMargin ?? gStyle.fPadLeftMargin,
            rm = pad?.fRightMargin ?? gStyle.fPadRightMargin,
            tm = pad?.fTopMargin ?? gStyle.fPadTopMargin,
            bm = pad?.fBottomMargin ?? gStyle.fPadBottomMargin;

      return svgToImage(svg_code).then(canvas => {
         if (!canvas) return false;

         let nX = 100, nY = 100;
         const context = canvas.getContext('2d'),
               arr = context.getImageData(0, 0, canvas.width, canvas.height).data,
               boxW = Math.floor(canvas.width / nX), boxH = Math.floor(canvas.height / nY),
               raster = new Array(nX*nY);

         if (arr.length !== canvas.width * canvas.height * 4) {
            console.log(`Image size missmatch in TLegend autoplace ${arr.length} expected ${canvas.width*canvas.height * 4}`);
            nX = nY = 0;
         }

         for (let ix = 0; ix < nX; ++ix) {
            const px1 = ix * boxW, px2 = px1 + boxW;
            for (let iy = 0; iy < nY; ++iy) {
               const py1 = iy * boxH, py2 = py1 + boxH;
               let filled = 0;

               for (let x = px1; (x < px2) && !filled; ++x) {
                  for (let y = py1; y < py2; ++y) {
                     const indx = (y * canvas.width + x) * 4;
                     if (arr[indx] || arr[indx+1] || arr[indx+2] || arr[indx+3]) {
                        filled = 1;
                        break;
                     }
                  }
                }
                raster[iy * nX + ix] = filled;
            }
         }

         const legWidth = 0.3 / Math.max(0.2, (1 - lm - rm)),
               legHeight = Math.min(0.5, Math.max(0.1, pt.fPrimitives.arr.length*0.05)) / Math.max(0.2, (1 - tm - bm)),
               needW = Math.round(legWidth * nX), needH = Math.round(legHeight * nY),

          test = (x, y) => {
            for (let ix = x; ix < x + needW; ++ix) {
               for (let iy = y; iy < y + needH; ++iy)
                  if (raster[iy * nX + ix]) return false;
            }
            return true;
         };

         for (let ix = 0; ix < (nX - needW); ++ix) {
            for (let iy = nY-needH - 1; iy >= 0; --iy) {
               if (test(ix, iy)) {
                  pt.fX1NDC = lm + ix / nX * (1 - lm - rm);
                  pt.fX2NDC = pt.fX1NDC + legWidth * (1 - lm - rm);
                  pt.fY2NDC = 1 - tm - iy/nY * (1 - bm - tm);
                  pt.fY1NDC = pt.fY2NDC - legHeight * (1 - bm - tm);
                  return true;
               }
            }
         }
      }).then(res => {
         if (res || keep_origin)
            return res;

         pt.fX1NDC = Math.max(lm ?? 0, pt.fX2NDC - 0.3);
         pt.fX2NDC = Math.min(pt.fX1NDC + 0.3, 1 - rm);
         const h0 = Math.max(pt.fPrimitives ? pt.fPrimitives.arr.length*0.05 : 0, 0.2);
         pt.fY2NDC = Math.min(1 - tm, pt.fY1NDC + h0);
         pt.fY1NDC = Math.max(pt.fY2NDC - h0, bm);
         return true;
      });
   }

   /** @summary Draw pave and content
     * @return {Promise} */
   async drawPave(arg) {
      if (!this.Enabled) {
         this.removeG();
         return this;
      }

      const pt = this.getObject(), opt = pt.fOption.toUpperCase(),
            fp = this.getFramePainter(), pp = this.getPadPainter(),
            pad = pp.getRootPad(true);
      let interactive_element, width, height;

      if (pt.fInit === 0) {
         this.stored = Object.assign({}, pt); // store coordinates to use them when updating
         pt.fInit = 1;

         if ((pt._typename === clTPaletteAxis) && !pt.fX1 && !pt.fX2 && !pt.fY1 && !pt.fY2) {
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
         } else if (opt.indexOf('NDC') >= 0) {
            pt.fX1NDC = pt.fX1; pt.fX2NDC = pt.fX2;
            pt.fY1NDC = pt.fY1; pt.fY2NDC = pt.fY2;
         } else if (pad && (pad.fX1 === 0) && (pad.fX2 === 1) && (pad.fY1 === 0) && (pad.fY2 === 1) && isStr(arg) && (arg.indexOf('postpone') >= 0)) {
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
      }

      let promise = Promise.resolve(true);

      if ((pt._typename === clTLegend) && (this.AutoPlace || ((pt.fX1NDC === pt.fX2NDC) && (pt.fY1NDC === pt.fY2NDC)))) {
         promise = this.autoPlaceLegend(pt, pad).then(res => {
            delete this.AutoPlace;
            if (!res) {
               pt.fX1NDC = fp.fX2NDC - 0.2; pt.fX2NDC = fp.fX2NDC;
               pt.fY1NDC = fp.fY2NDC - 0.1; pt.fY2NDC = fp.fY2NDC;
            }
            return res;
         });
      }

      return promise.then(() => {
         // fill stats before drawing to have coordinates early
         if (this.isStats() && !this.NoFillStats && !pp._fast_drawing) {
            const main = pt.$main_painter || this.getMainPainter();

            if (isFunc(main?.fillStatistic)) {
               let dostat = parseInt(pt.fOptStat), dofit = parseInt(pt.fOptFit);
               if (!Number.isInteger(dostat) || pt.TestBit(kTakeStyle)) dostat = gStyle.fOptStat;
               if (!Number.isInteger(dofit)|| pt.TestBit(kTakeStyle)) dofit = gStyle.fOptFit;

               // we take statistic from main painter
               if (main.fillStatistic(this, dostat, dofit)) {
                  // adjust the size of the stats box with the number of lines
                  let nlines = pt.fLines?.arr.length || 0;
                  if ((nlines > 0) && !this.moved_interactive && isDefaultStatPosition(pt)) {
                     // in ROOT TH2 and TH3 always add full statsh for fit parameters
                     const extrah = this._has_fit && (this._fit_dim > 1) ? gStyle.fStatH : 0;
                     // but fit parameters not used in full size calculations
                     if (extrah) nlines -= this._fit_cnt;
                     let stath = gStyle.fStatH, statw = gStyle.fStatW;
                     if (this._has_fit)
                        statw = 1.8 * gStyle.fStatW;
                     if ((gStyle.fStatFontSize <= 0) || (gStyle.fStatFont % 10 === 3))
                        stath = nlines * 0.25 * gStyle.fStatH;
                     else if (gStyle.fStatFontSize < 1)
                        stath = nlines * gStyle.fStatFontSize;
                     pt.fX1NDC = Math.max(0.02, pt.fX2NDC - statw);
                     pt.fY1NDC = Math.max(0.02, pt.fY2NDC - stath - extrah);
                  }
               }
            }
         }

         const pad_rect = pp.getPadRect(),
               brd = pt.fBorderSize,
               noborder = opt.indexOf('NB') >= 0,
               dx = (opt.indexOf('L') >= 0) ? -1 : ((opt.indexOf('R') >= 0) ? 1 : 0),
               dy = (opt.indexOf('T') >= 0) ? -1 : ((opt.indexOf('B') >= 0) ? 1 : 0);

         // container used to recalculate coordinates
         this.createG();

         this._pave_x = Math.round(pt.fX1NDC * pad_rect.width);
         this._pave_y = Math.round((1.0 - pt.fY2NDC) * pad_rect.height);
         width = Math.round((pt.fX2NDC - pt.fX1NDC) * pad_rect.width);
         height = Math.round((pt.fY2NDC - pt.fY1NDC) * pad_rect.height);

         makeTranslate(this.draw_g, this._pave_x, this._pave_y);

         this.createAttLine({ attr: pt, width: (brd > 0) ? pt.fLineWidth : 0 });

         this.createAttFill({ attr: pt });

         if (pt._typename === clTDiamond) {
            const h2 = Math.round(height/2), w2 = Math.round(width/2),
                  dpath = `l${w2},${-h2}l${w2},${h2}l${-w2},${h2}z`;

            if ((brd > 1) && (pt.fShadowColor > 0) && (dx || dy) && !this.fillatt.empty() && !noborder) {
                this.draw_g.append('svg:path')
                    .attr('d', 'M0,'+(h2+brd) + dpath)
                    .style('fill', this.getColor(pt.fShadowColor))
                    .style('stroke', this.getColor(pt.fShadowColor))
                    .style('stroke-width', '1px');
            }

            interactive_element = this.draw_g.append('svg:path')
                                      .attr('d', 'M0,'+h2 +dpath)
                                      .call(this.fillatt.func)
                                      .call(this.lineatt.func);

            const text_g = this.draw_g.append('svg:g');
            makeTranslate(text_g, Math.round(width/4), Math.round(height/4));

            return this.drawPaveText(w2, h2, arg, text_g);
         } else {
            // add shadow decoration before main rect
            if ((brd > 1) && (pt.fShadowColor > 0) && !pt.fNpaves && (dx || dy) && !noborder) {
               const scol = this.getColor(pt.fShadowColor);
               let spath = '';

               if ((dx < 0) && (dy < 0))
                  spath = `M0,0v${height-brd}h${-brd}v${-height}h${width}v${brd}z`;
               else if ((dx < 0) && (dy > 0))
                  spath = `M0,${height}v${brd-height}h${-brd}v${height}h${width}v${-brd}z`;
               else if ((dx > 0) && (dy < 0))
                  spath = `M${brd},0v${-brd}h${width}v${height}h${-brd}v${brd-height}z`;
               else
                  spath = `M${width},${brd}h${brd}v${height}h${-width}v${-brd}h${width-brd}z`;

               this.draw_g.append('svg:path')
                          .attr('d', spath)
                          .style('fill', scol)
                          .style('stroke', scol)
                          .style('stroke-width', '1px');
            }

            if (pt.fNpaves) {
               for (let n = pt.fNpaves-1; n > 0; --n) {
                  this.draw_g.append('svg:path')
                      .attr('d', `M${dx*4*n},${dy*4*n}h${width}v${height}h${-width}z`)
                      .call(this.fillatt.func)
                      .call(this.lineatt.func);
               }
            }

            if (!this.isBatchMode() || !this.fillatt.empty() || (!this.lineatt.empty() && !noborder)) {
               interactive_element = this.draw_g.append('svg:path')
                                                .attr('d', `M0,0H${width}V${height}H0Z`)
                                                .call(this.fillatt.func);
               if (!noborder)
                  interactive_element.call(this.lineatt.func);
            }

            return isFunc(this.paveDrawFunc) ? this.paveDrawFunc(width, height, arg) : true;
         }
      }).then(() => {
         if (this.isBatchMode() || (pt._typename === clTPave))
            return this;

         // here all kind of interactive settings
         if (interactive_element) {
            interactive_element.style('pointer-events', 'visibleFill')
                               .on('mouseenter', () => this.showObjectStatus());
         }

         addDragHandler(this, { obj: pt, x: this._pave_x, y: this._pave_y, width, height,
                                minwidth: 10, minheight: 20, canselect: true,
                        redraw: () => { this.moved_interactive = true; this.interactiveRedraw(false, 'pave_moved'); this.drawPave(); },
                        ctxmenu: browser.touches && settings.ContextMenu && this.UseContextMenu });

         if (this.UseContextMenu && settings.ContextMenu)
             this.draw_g.on('contextmenu', evnt => this.paveContextMenu(evnt));

         if (pt._typename === clTPaletteAxis)
            this.interactivePaletteAxis(width, height);

         return this;
      });
   }

   /** @summary Fill option object used in TWebCanvas */
   fillWebObjectOptions(res) {
      const pave = this.getObject();

      if (pave?.fInit) {
         res.fcust = 'pave';
         res.fopt = [pave.fX1NDC, pave.fY1NDC, pave.fX2NDC, pave.fY2NDC];

         if ((pave.fName === 'stats') && this.isStats()) {
             pave.fLines.arr.forEach(entry => {
                if ((entry._typename === clTText) || (entry._typename === clTLatex))
                   res.fcust += `;;${entry.fTitle}`;
             });
         }
      }

      return res;
   }

   /** @summary draw TPaveLabel object */
   async drawPaveLabel(width, height) {
      const pave = this.getObject();
      if (!pave.fLabel || !pave.fLabel.trim())
         return this;

      this.createAttText({ attr: pave, can_rotate: false });

      this.startTextDrawing(this.textatt.font, height/1.2);

      this.drawText(this.textatt.createArg({ width, height, text: pave.fLabel, norotate: true }));

      return this.finishTextDrawing();
   }

   /** @summary draw TPaveStats object */
   drawPaveStats(width, height) {
      const pt = this.getObject(), lines = [], colors = [];
      let first_stat = 0, num_cols = 0, maxlen = 0;

      // extract only text
      for (let j = 0; j < pt.fLines.arr.length; ++j) {
         const entry = pt.fLines.arr[j];
         if ((entry._typename === clTText) || (entry._typename === clTLatex)) {
            lines.push(entry.fTitle);
            colors.push(entry.fTextColor);
          }
      }

      const nlines = lines.length;

      // adjust font size
      for (let j = 0; j < nlines; ++j) {
         const line = lines[j];
         if (j > 0) maxlen = Math.max(maxlen, line.length);
         if ((j === 0) || (line.indexOf('|') < 0)) continue;
         if (first_stat === 0) first_stat = j;
         const parts = line.split('|');
         if (parts.length > num_cols)
            num_cols = parts.length;
      }

      // for characters like 'p' or 'y' several more pixels required to stay in the box when drawn in last line
      const stepy = height / nlines, margin_x = pt.fMargin * width;
      let has_head = false;

      this.createAttText({ attr: pt, can_rotate: false });

      this.startTextDrawing(this.textatt.font, height/(nlines * 1.2));

      if (nlines === 1)
         this.drawText(this.textatt.createArg({ width, height, text: lines[0], latex: 1, norotate: true }));
       else {
          for (let j = 0; j < nlines; ++j) {
            const y = j*stepy,
                  color = (colors[j] > 1) ? this.getColor(colors[j]) : this.textatt.color;

            if (first_stat && (j >= first_stat)) {
               const parts = lines[j].split('|');
               for (let n = 0; n < parts.length; ++n) {
                  this.drawText({ align: 'middle', x: width * n / num_cols, y, latex: 0,
                                  width: width/num_cols, height: stepy, text: parts[n], color });
               }
            } else if (lines[j].indexOf('=') < 0) {
               if (j === 0) {
                  has_head = true;
                  const max_hlen = Math.max(maxlen, Math.round((width-2*margin_x)/stepy/0.65));
                  if (lines[j].length > max_hlen + 5)
                     lines[j] = lines[j].slice(0, max_hlen+2) + '...';
               }
               this.drawText({ align: (j === 0) ? 'middle' : 'start', x: margin_x, y,
                               width: width-2*margin_x, height: stepy, text: lines[j], color });
            } else {
               const parts = lines[j].split('='), args = [];

               for (let n = 0; n < 2; ++n) {
                  const arg = {
                     align: (n === 0) ? 'start' : 'end', x: margin_x, y,
                     width: width - 2*margin_x, height: stepy, text: parts[n], color,
                     _expected_width: width-2*margin_x, _args: args,
                     post_process(painter) {
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
      }

      let lpath = '';

      if ((pt.fBorderSize > 0) && has_head)
         lpath += `M0,${Math.round(stepy)}h${width}`;

      if ((first_stat > 0) && (num_cols > 1)) {
         for (let nrow = first_stat; nrow < nlines; ++nrow)
            lpath += `M0,${Math.round(nrow * stepy)}h${width}`;
         for (let ncol = 0; ncol < num_cols - 1; ++ncol)
            lpath += `M${Math.round(width / num_cols * (ncol + 1))},${Math.round(first_stat * stepy)}V${height}`;
      }

      if (lpath) this.draw_g.append('svg:path').attr('d', lpath).call(this.lineatt.func);

      // this.draw_g.classed('most_upper_primitives', true); // this primitive will remain on top of list

      return this.finishTextDrawing(undefined, (nlines > 1));
   }

   /** @summary draw TPaveText object */
   drawPaveText(width, height, _dummy_arg, text_g) {
      const pt = this.getObject(),
            arr = pt.fLines?.arr || [],
            nlines = arr.length,
            pp = this.getPadPainter(),
            pad_height = pp.getPadHeight(),
            draw_header = (pt.fLabel.length > 0),
            promises = [],
            margin_x = pt.fMargin * width,
            stepy = height / (nlines || 1);
      let max_font_size = 0;

      this.createAttText({ attr: pt, can_rotate: false });

      // for single line (typically title) limit font size
      if ((nlines === 1) && (this.textatt.size > 0))
         max_font_size = Math.max(3, this.textatt.getSize(pad_height));

      if (!text_g) text_g = this.draw_g;

      const fast = (nlines === 1) && pp._fast_drawing;
      let num_default = 0, is_any_text = false;

      for (let nline = 0; nline < nlines; ++nline) {
         const entry = arr[nline], texty = nline*stepy;

         switch (entry._typename) {
            case clTText:
            case clTLatex: {
               if (!entry.fTitle || !entry.fTitle.trim()) continue;

               let color = entry.fTextColor ? this.getColor(entry.fTextColor) : '';
               if (!color) color = this.textatt.color;
               is_any_text = true;
               if (entry.fX || entry.fY || entry.fTextSize) {
                  // individual positioning
                  const align = entry.fTextAlign || this.textatt.align,
                        halign = Math.floor(align/10),
                        valign = align % 10,
                        x = entry.fX ? entry.fX*width : (halign === 1 ? margin_x : (halign === 2 ? width / 2 : width - margin_x)),
                        y = entry.fY ? (1 - entry.fY)*height : (texty + (valign === 2 ? stepy / 2 : (valign === 3 ? stepy : 0))),
                        sub_g = text_g.append('svg:g');

                  this.startTextDrawing(this.textatt.font, this.textatt.getAltSize(entry.fTextSize, pad_height), sub_g);

                  this.drawText({ align, x, y, text: entry.fTitle, color,
                                  latex: (entry._typename === clTText) ? 0 : 1, draw_g: sub_g, fast });

                  promises.push(this.finishTextDrawing(sub_g));
               } else {
                  // default position
                  if (num_default++ === 0)
                     this.startTextDrawing(this.textatt.font, 0.85*height/nlines, text_g, max_font_size);

                  this.drawText({ x: margin_x, y: texty, width: width - 2*margin_x, height: stepy,
                                  align: entry.fTextAlign || this.textatt.align,
                                  draw_g: text_g, latex: (entry._typename === clTText) ? 0 : 1,
                                  text: entry.fTitle, color, fast });
               }
               break;
            }

            case clTLine: {
               const lx1 = entry.fX1 ? Math.round(entry.fX1*width) : 0,
                     lx2 = entry.fX2 ? Math.round(entry.fX2*width) : width,
                     ly1 = entry.fY1 ? Math.round((1 - entry.fY1)*height) : Math.round(texty + stepy*0.5),
                     ly2 = entry.fY2 ? Math.round((1 - entry.fY2)*height) : Math.round(texty + stepy*0.5),
                     lineatt = this.createAttLine(entry);
               text_g.append('svg:path')
                     .attr('d', `M${lx1},${ly1}L${lx2},${ly2}`)
                     .call(lineatt.func);
               break;
            }
            case clTBox: {
               const bx1 = entry.fX1 ? Math.round(entry.fX1*width) : 0,
                     bx2 = entry.fX2 ? Math.round(entry.fX2*width) : width,
                     by1 = entry.fY1 ? Math.round((1 - entry.fY1)*height) : Math.round(texty),
                     by2 = entry.fY2 ? Math.round((1 - entry.fY2)*height) : Math.round(texty + stepy),
                     fillatt = this.createAttFill(entry);
               text_g.append('svg:path')
                     .attr('d', `M${bx1},${by1}H${bx2}V${by2}H${bx1}Z`)
                     .call(fillatt.func);
               break;
            }
         }
      }

      if (num_default > 0)
         promises.push(this.finishTextDrawing(text_g, num_default > 1));

      if (this.isTitle())
         this.draw_g.style('display', !is_any_text ? 'none' : null);

      if (draw_header) {
         const x = Math.round(width*0.25),
             y = Math.round(-height*0.02),
             w = Math.round(width*0.5),
             h = Math.round(height*0.04),
             lbl_g = text_g.append('svg:g');

         lbl_g.append('svg:path')
               .attr('d', `M${x},${y}h${w}v${h}h${-w}z`)
               .call(this.fillatt.func)
               .call(this.lineatt.func);

         this.startTextDrawing(this.textatt.font, h/1.5, lbl_g);

         this.drawText({ align: 22, x, y, width: w, height: h, text: pt.fLabel, color: this.textatt.color, draw_g: lbl_g });

         promises.push(this.finishTextDrawing(lbl_g));
      }

      return Promise.all(promises).then(() => this);
   }

   /** @summary Method used to convert value to string according specified format
     * @desc format can be like 5.4g or 4.2e or 6.4f or 'stat' or 'fit' or 'entries' */
   format(value, fmt) {
      if (!fmt) fmt = 'stat';

      const pave = this.getObject();

      switch (fmt) {
         case 'stat' : fmt = pave.fStatFormat || gStyle.fStatFormat; break;
         case 'fit': fmt = pave.fFitFormat || gStyle.fFitFormat; break;
         case 'entries':
            if ((Math.abs(value) < 1e9) && (Math.round(value) === value))
               return value.toFixed(0);
            fmt = '14.7g';
            break;
         case 'last': fmt = this.lastformat; break;
      }

      const res = floatToString(value, fmt || '6.4g', true);

      this.lastformat = res[1];

      return res[0];
   }

   /** @summary Draw TLegend object */
   drawLegend(w, h) {
      const legend = this.getObject(),
            nlines = legend.fPrimitives.arr.length;
      let ncols = legend.fNColumns,
          nrows = nlines,
          any_text = false,
          custom_textg = false; // each text entry has own attributes

      if (ncols < 2)
         ncols = 1;
      else
         while ((nrows-1)*ncols >= nlines) nrows--;

      const isEmpty = entry => !entry.fObject && !entry.fOption && (!entry.fLabel || (entry.fLabel === ' '));

      for (let ii = 0; ii < nlines; ++ii) {
         const entry = legend.fPrimitives.arr[ii];
         if (isEmpty(entry)) {
            if (ncols === 1)
               nrows--;
         } else if (entry.fLabel) {
            any_text = true;
            if ((entry.fTextFont && (entry.fTextFont !== legend.fTextFont)) ||
                (entry.fTextSize && (entry.fTextSize !== legend.fTextSize)))
                   custom_textg = true;
         }
      }

      if (nrows < 1) nrows = 1;

      // calculate positions of columns by weight - means more letters, more weight
      const column_pos = new Array(ncols + 1).fill(0);
      if (ncols > 1) {
         const column_weight = new Array(ncols).fill(1);

         for (let ii = 0; ii < nlines; ++ii) {
            const entry = legend.fPrimitives.arr[ii];
            if (isEmpty(entry)) continue; // let discard empty entry
            const icol = ii % ncols;
            column_weight[icol] = Math.max(column_weight[icol], entry.fLabel.length);
         }

         let sum_weight = 0;
         for (let icol = 0; icol < ncols; ++icol)
            sum_weight += column_weight[icol];
         for (let icol = 0; icol < ncols-1; ++icol)
            column_pos[icol+1] = column_pos[icol] + legend.fMargin*w/ncols + column_weight[icol] * (1-legend.fMargin) * w / sum_weight;
      }
      column_pos[ncols] = w;

      const padding_x = Math.round(0.03*w/ncols),
            padding_y = Math.round(0.03*h),
            step_y = (h - 2*padding_y)/nrows,
            text_promises = [],
            pp = this.getPadPainter();
      let font_size = 0.9*step_y,
          max_font_size = 0, // not limited in the beggining
          any_opt = false;

      this.createAttText({ attr: legend, can_rotate: false });

      const tsz = this.textatt.getSize(pp.getPadHeight());
      if (tsz && (tsz < font_size))
         font_size = max_font_size = tsz;

      if (any_text && !custom_textg)
         this.startTextDrawing(this.textatt.font, font_size, this.draw_g, max_font_size);

      for (let ii = 0, i = -1; ii < nlines; ++ii) {
         const entry = legend.fPrimitives.arr[ii];
         if (isEmpty(entry)) continue; // let discard empty entry

         if (ncols === 1) ++i; else i = ii;

         const lopt = entry.fOption.toLowerCase(),
               icol = i % ncols, irow = (i - icol) / ncols,
               x0 = Math.round(column_pos[icol]),
               column_width = Math.round(column_pos[icol + 1] - column_pos[icol]),
               tpos_x = x0 + Math.round(legend.fMargin*w/ncols),
               mid_x = Math.round((x0 + tpos_x)/2),
               pos_y = Math.round(irow*step_y + padding_y), // top corner
               mid_y = Math.round((irow+0.5)*step_y + padding_y), // center line
               mo = entry.fObject,
               draw_fill = lopt.indexOf('f') !== -1,
               draw_line = lopt.indexOf('l') !== -1,
               draw_error = lopt.indexOf('e') !== -1,
               draw_marker = lopt.indexOf('p') !== -1;

         let o_fill = entry, o_marker = entry, o_line = entry,
             painter = null, isany = false;

         if (isObject(mo)) {
            if ('fLineColor' in mo) o_line = mo;
            if ('fFillColor' in mo) o_fill = mo;
            if ('fMarkerColor' in mo) o_marker = mo;
            painter = pp.findPainterFor(mo);
         }

         // Draw fill pattern (in a box)
         if (draw_fill) {
            const fillatt = painter?.fillatt?.used ? painter.fillatt : this.createAttFill(o_fill);
            let lineatt;
            if (!draw_line && !draw_error && !draw_marker) {
               lineatt = painter?.lineatt?.used ? painter.lineatt : this.createAttLine(o_line);
               if (lineatt.empty()) lineatt = null;
            }

            if (!fillatt.empty() || lineatt) {
               isany = true;
               // box total height is yspace*0.7
               // define x,y as the center of the symbol for this entry
               const rect = this.draw_g.append('svg:path')
                              .attr('d', `M${x0 + padding_x},${Math.round(pos_y+step_y*0.1)}v${Math.round(step_y*0.8)}h${tpos_x-2*padding_x-x0}v${-Math.round(step_y*0.8)}z`);
               if (!fillatt.empty())
                  rect.call(fillatt.func);
               else
                  rect.style('fill', 'none');
               if (lineatt)
                  rect.call(lineatt.func);
            }
         }

         // Draw line and/or error (when specified)
         if (draw_line || draw_error) {
            const lineatt = painter?.lineatt?.used ? painter.lineatt : this.createAttLine(o_line);
            if (!lineatt.empty()) {
               isany = true;
               if (draw_line) {
                  this.draw_g.append('svg:path')
                      .attr('d', `M${x0 + padding_x},${mid_y}H${tpos_x - padding_x}`)
                      .call(lineatt.func);
               }
               if (draw_error) {
                  let endcaps = 0, edx = step_y*0.05;
                  if (isFunc(painter?.getHisto) && painter.options?.ErrorKind === 1)
                     endcaps = 1; // draw bars for e1 option in histogram
                  else if (isFunc(painter?.getGraph) && mo?.fLineWidth !== undefined && mo?.fMarkerSize !== undefined) {
                     endcaps = painter.options?.Ends ?? 1; // deafult is 1
                     edx = mo.fLineWidth + gStyle.fEndErrorSize;
                     if (endcaps > 1) edx = Math.max(edx, mo.fMarkerSize*8*0.66);
                  }

                  const eoff = (endcaps === 3) ? 0.03 : 0,
                        ey1 = Math.round(pos_y+step_y*(0.1 + eoff)),
                        ey2 = Math.round(pos_y+step_y*(0.9 - eoff)),
                        edy = Math.round(edx * 0.66);
                  edx = Math.round(edx);
                  let path = `M${mid_x},${ey1}V${ey2}`;
                  switch (endcaps) {
                     case 1: path += `M${mid_x-edx},${ey1}h${2*edx}M${mid_x-edx},${ey2}h${2*edx}`; break; // bars
                     case 2: path += `M${mid_x-edx},${ey1+edy}v${-edy}h${2*edx}v${edy}M${mid_x-edx},${ey2-edy}v${edy}h${2*edx}v${-edy}`; break; // ]
                     case 3: path += `M${mid_x-edx},${ey1}h${2*edx}l${-edx},${-edy}zM${mid_x-edx},${ey2}h${2*edx}l${-edx},${edy}z`; break; // triangle
                     case 4: path += `M${mid_x-edx},${ey1+edy}l${edx},${-edy}l${edx},${edy}M${mid_x-edx},${ey2-edy}l${edx},${edy}l${edx},${-edy}`; break; // arrow
                  }
                  this.draw_g.append('svg:path')
                      .attr('d', path)
                      .call(lineatt.func)
                      .style('fill', endcaps > 1 ? 'none' : null);
               }
            }
         }

         // Draw Polymarker
         if (draw_marker) {
            const marker = painter?.markeratt?.used ? painter.markeratt : this.createAttMarker(o_marker);
            if (!marker.empty()) {
               isany = true;
               this.draw_g
                   .append('svg:path')
                   .attr('d', marker.create((x0 + tpos_x)/2, mid_y))
                   .call(marker.func);
            }
         }

         // special case - nothing draw, try to show rect with line attributes
         if (!isany && painter?.lineatt && !painter.lineatt.empty()) {
            this.draw_g.append('svg:path')
                       .attr('d', `M${x0 + padding_x},${Math.round(pos_y+step_y*0.1)}v${Math.round(step_y*0.8)}h${tpos_x-2*padding_x-x0}v${-Math.round(step_y*0.8)}z`)
                       .style('fill', 'none')
                       .call(painter.lineatt.func);
         }

         let pos_x = tpos_x;
         if (isStr(lopt) && (lopt.toLowerCase() !== 'h'))
            any_opt = true;
         else if (!any_opt)
            pos_x = x0 + padding_x;

         if (entry.fLabel) {
            let lbl_g = this.draw_g;
            const textatt = this.createAttText({ attr: entry, std: false, attr_alt: legend });
            if (custom_textg) {
               lbl_g = this.draw_g.append('svg:g');
               const entry_font_size = textatt.getSize(pp.getPadHeight());
               this.startTextDrawing(textatt.font, entry_font_size, lbl_g, max_font_size);
            }

            this.drawText({ draw_g: lbl_g, align: textatt.align, x: pos_x, y: pos_y,
                            scale: (custom_textg && !entry.fTextSize) || !legend.fTextSize,
                            width: x0+column_width-pos_x-padding_x, height: step_y,
                            text: entry.fLabel, color: textatt.color });

            if (custom_textg)
               text_promises.push(this.finishTextDrawing(lbl_g));
         }
      }

      if (any_text && !custom_textg)
         text_promises.push(this.finishTextDrawing());

      // rescale after all entries are shown
      return Promise.all(text_promises);
   }

   /** @summary draw color palette with axis */
   drawPaletteAxis(s_width, s_height, arg) {
      const palette = this.getObject(),
            axis = palette.fAxis,
            can_move = isStr(arg) && (arg.indexOf('can_move') >= 0),
            postpone_draw = isStr(arg) && (arg.indexOf('postpone') >= 0),
            cjust = isStr(arg) && (arg.indexOf('cjust') >= 0),
            pp = this.getPadPainter(),
            width = pp.getPadWidth(),
            height = pp.getPadHeight(),
            pad = pp.getRootPad(true),
            main = palette.$main_painter || this.getMainPainter(),
            framep = this.getFramePainter(),
            contour = main.fContour,
            levels = contour?.getLevels(),
            is_th3 = isFunc(main.getDimension) && (main.getDimension() === 3),
            log = pad?.fLogv ?? (is_th3 ? false : pad?.fLogz),
            draw_palette = main._color_palette,
            zaxis = main.getObject()?.fZaxis,
            sizek = pad?.fTickz ? 0.35 : 0.7;

      let zmin = 0, zmax = 100, gzmin, gzmax, axis_transform = '', axis_second = 0;

      this._palette_vertical = (palette.fX2NDC - palette.fX1NDC) < (palette.fY2NDC - palette.fY1NDC);

      axis.fTickSize = 0.6 * s_width / width; // adjust axis ticks size
      if ((typeof zaxis?.fLabelOffset !== 'undefined') && !is_th3) {
         axis.fBits = zaxis.fBits & ~EAxisBits.kTickMinus & ~EAxisBits.kTickPlus;
         axis.fTitle = zaxis.fTitle;
         axis.fTitleSize = zaxis.fTitleSize;
         axis.fTitleOffset = zaxis.fTitleOffset;
         axis.fTextColor = zaxis.fTitleColor;
         axis.fTextFont = zaxis.fTitleFont;
         axis.fLineColor = zaxis.fAxisColor;
         axis.fLabelSize = zaxis.fLabelSize;
         axis.fLabelColor = zaxis.fLabelColor;
         axis.fLabelFont = zaxis.fLabelFont;
         axis.fLabelOffset = zaxis.fLabelOffset;
         this.z_handle.setHistPainter(main, 'z');
         this.z_handle.source_axis = zaxis;
      }

      if (contour && framep && !is_th3) {
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
         // zmin = Math.min(levels[0], framep.zmin);
         // zmax = Math.max(levels[levels.length-1], framep.zmax);
      } else if ((main.gmaxbin !== undefined) && (main.gminbin !== undefined)) {
         // this is case of TH2 (needs only for size adjustment)
         zmin = main.gminbin; zmax = main.gmaxbin;
      } else if ((main.hmin !== undefined) && (main.hmax !== undefined)) {
         // this is case of TH1
         zmin = main.hmin; zmax = main.hmax;
      }

      this.draw_g.selectAll('rect').style('fill', 'white');

      if ((gzmin === undefined) || (gzmax === undefined) || (gzmin === gzmax)) {
         gzmin = zmin; gzmax = zmax;
      }

      if (this._palette_vertical) {
         this._swap_side = palette.fX2NDC < 0.5;
         this.z_handle.configureAxis('zaxis', gzmin, gzmax, zmin, zmax, true, [0, s_height], { log, fixed_ticks: cjust ? levels : null, maxTickSize: Math.round(s_width*sizek), swap_side: this._swap_side });
         axis_transform = this._swap_side ? null : `translate(${s_width})`;
         if (pad?.fTickz) axis_second = this._swap_side ? s_width : -s_width;
      } else {
         this._swap_side = palette.fY1NDC > 0.5;
         this.z_handle.configureAxis('zaxis', gzmin, gzmax, zmin, zmax, false, [0, s_width], { log, fixed_ticks: cjust ? levels : null, maxTickSize: Math.round(s_height*sizek), swap_side: this._swap_side });
         axis_transform = this._swap_side ? null : `translate(0,${s_height})`;
         if (pad?.fTickz) axis_second = this._swap_side ? s_height : -s_height;
      }

      if (!contour || !draw_palette || postpone_draw) {
         // we need such rect to correctly calculate size
         this.draw_g.append('svg:path')
                    .attr('d', `M0,0H${s_width}V${s_height}H0Z`)
                    .style('fill', 'white');
      } else {
         for (let i = 0; i < levels.length-1; ++i) {
            let z0 = Math.round(this.z_handle.gr(levels[i])),
                z1 = Math.round(this.z_handle.gr(levels[i+1])),
                lvl = (levels[i] + levels[i+1])*0.5, d;

            if (this._palette_vertical) {
               if ((z1 >= s_height) || (z0 < 0)) continue;
               z0 += 1; // ensure correct gap filling between colors

               if (z0 > s_height) {
                  z0 = s_height;
                  lvl = levels[i]*0.001 + levels[i+1]*0.999;
                  if (z1 < 0) z1 = 0;
               } else if (z1 < 0) {
                  z1 = 0;
                  lvl = levels[i]*0.999 + levels[i+1]*0.001;
               }
               d = `M0,${z1}H${s_width}V${z0}H0Z`;
            } else {
               if ((z0 >= s_width) || (z1 < 0)) continue;
               z1 += 1; // ensure correct gap filling between colors

               if (z1 > s_width) {
                  z1 = s_width;
                  lvl = levels[i]*0.999 + levels[i+1]*0.001;
                  if (z0 < 0) z0 = 0;
               } else if (z0 < 0) {
                  z0 = 0;
                  lvl = levels[i]*0.001 + levels[i+1]*0.999;
               }
               d = `M${z0},0V${s_height}H${z1}V0Z`;
            }

            const col = contour.getPaletteColor(draw_palette, lvl);
            if (!col) continue;

            const r = this.draw_g.append('svg:path')
                       .attr('d', d)
                       .style('fill', col)
                       .property('fill0', col)
                       .property('fill1', d3_rgb(col).darker(0.5).formatHex());

            if (this.isTooltipAllowed()) {
               r.on('mouseover', function() {
                  d3_select(this).transition().duration(100).style('fill', d3_select(this).property('fill1'));
               }).on('mouseout', function() {
                  d3_select(this).transition().duration(100).style('fill', d3_select(this).property('fill0'));
               }).append('svg:title').text(levels[i].toFixed(2) + ' - ' + levels[i+1].toFixed(2));
            }

            if (settings.Zooming)
               r.on('dblclick', () => this.getFramePainter().unzoom('z'));
         }
      }

      return this.z_handle.drawAxis(this.draw_g, s_width, s_height, axis_transform, axis_second).then(() => {
         if (can_move && ('getBoundingClientRect' in this.draw_g.node())) {
            const rect = this.draw_g.node().getBoundingClientRect();

            if (this._palette_vertical) {
               const shift = (this._pave_x + parseInt(rect.width)) - Math.round(0.995*width) + 3;

               if (shift > 0) {
                  this._pave_x -= shift;
                  makeTranslate(this.draw_g, this._pave_x, this._pave_y);
                  palette.fX1NDC -= shift/width;
                  palette.fX2NDC -= shift/width;
               }
            } else {
               const shift = Math.round((1.05 - gStyle.fTitleY)*height) - rect.y;
               if (shift > 0) {
                  this._pave_y += shift;
                  makeTranslate(this.draw_g, this._pave_x, this._pave_y);
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

         const m = d3_pointer(evnt, this.draw_g.node());
         if (this._palette_vertical) {
            sel2 = Math.min(Math.max(m[1], 0), s_height);
            zoom_rect.attr('y', Math.min(sel1, sel2))
                     .attr('height', Math.abs(sel2-sel1));
         } else {
            sel2 = Math.min(Math.max(m[0], 0), s_width);
            zoom_rect.attr('x', Math.min(sel1, sel2))
                     .attr('width', Math.abs(sel2-sel1));
         }
      }, endRectSel = evnt => {
         if (!doing_zoom) return;

         evnt.preventDefault();
         d3_select(window).on('mousemove.colzoomRect', null)
                          .on('mouseup.colzoomRect', null);
         zoom_rect.remove();
         zoom_rect = null;
         doing_zoom = false;

         const z1 = this.z_handle.revertPoint(sel1),
               z2 = this.z_handle.revertPoint(sel2);

         this.getFramePainter().zoom('z', Math.min(z1, z2), Math.max(z1, z2));
         this.getFramePainter().zoomChangedInteractive('z', true);
      }, startRectSel = evnt => {
         // ignore when touch selection is activated
         if (doing_zoom) return;
         doing_zoom = true;

         evnt.preventDefault();
         evnt.stopPropagation();

         const origin = d3_pointer(evnt, this.draw_g.node());

         zoom_rect = this.draw_g.append('svg:rect').attr('id', 'colzoomRect').call(addHighlightStyle, true);

         if (this._palette_vertical) {
            sel1 = sel2 = origin[1];
            zoom_rect.attr('x', '0')
                     .attr('width', s_width)
                     .attr('y', sel1)
                     .attr('height', 1);
         } else {
            sel1 = sel2 = origin[0];
            zoom_rect.attr('x', sel1)
                     .attr('width', 1)
                     .attr('y', 0)
                     .attr('height', s_height);
         }

         d3_select(window).on('mousemove.colzoomRect', moveRectSel)
                          .on('mouseup.colzoomRect', endRectSel, true);
      };

      if (settings.Zooming) {
         this.draw_g.selectAll('.axis_zoom')
                    .on('mousedown', startRectSel)
                    .on('dblclick', () => this.getFramePainter().unzoom('z'));
      }

      if (settings.ZoomWheel) {
         this.draw_g.on('wheel', evnt => {
            const pos = d3_pointer(evnt, this.draw_g.node()),
                  coord = this._palette_vertical ? (1 - pos[1] / s_height) : pos[0] / s_width,
                  item = this.z_handle.analyzeWheelEvent(evnt, coord);
            if (item?.changed) {
               this.getFramePainter().zoom('z', item.min, item.max);
               this.getFramePainter().zoomChangedInteractive('z', true);
            }
         });
       }
   }

   /** @summary Fill context menu items for the TPave object */
   fillContextMenuItems(menu) {
      const pave = this.getObject();

      if (this.isStats()) {
         menu.add('Default position', () => {
            pave.fX2NDC = gStyle.fStatX;
            pave.fX1NDC = pave.fX2NDC - gStyle.fStatW;
            pave.fY2NDC = gStyle.fStatY;
            pave.fY1NDC = pave.fY2NDC - gStyle.fStatH;
            pave.fInit = 1;
            this.interactiveRedraw(true, 'pave_moved');
         });

         menu.add('Save to gStyle', () => {
            gStyle.fStatX = pave.fX2NDC;
            gStyle.fStatW = pave.fX2NDC - pave.fX1NDC;
            gStyle.fStatY = pave.fY2NDC;
            gStyle.fStatH = pave.fY2NDC - pave.fY1NDC;
            this.fillatt?.saveToStyle('fStatColor', 'fStatStyle');
            gStyle.fStatTextColor = pave.fTextColor;
            gStyle.fStatFontSize = pave.fTextSize;
            gStyle.fStatFont = pave.fTextFont;
         }, 'Store stats position and graphical attributes to gStyle');

         menu.add('SetStatFormat', () => {
            menu.input('Enter StatFormat', pave.fStatFormat).then(fmt => {
               if (!fmt) return;
               pave.fStatFormat = fmt;
               this.interactiveRedraw(true, `exec:SetStatFormat("${fmt}")`);
            });
         });
         menu.add('SetFitFormat', () => {
            menu.input('Enter FitFormat', pave.fFitFormat).then(fmt => {
               if (!fmt) return;
               pave.fFitFormat = fmt;
               this.interactiveRedraw(true, `exec:SetFitFormat("${fmt}")`);
            });
         });
         menu.add('separator');
         menu.add('sub:SetOptStat', () => {
            menu.input('Enter OptStat', pave.fOptStat, 'int').then(fmt => {
               pave.fOptStat = fmt;
               this.interactiveRedraw(true, `exec:SetOptStat(${fmt})`);
            });
         });
         const addStatOpt = (pos, name) => {
            let opt = (pos < 10) ? pave.fOptStat : pave.fOptFit;
            opt = parseInt(parseInt(opt) / parseInt(Math.pow(10, pos % 10))) % 10;
            menu.addchk(opt, name, opt * 100 + pos, arg => {
               const oldopt = parseInt(arg / 100);
               let newopt = (arg % 100 < 10) ? pave.fOptStat : pave.fOptFit;
               newopt -= (oldopt > 0 ? oldopt : -1) * parseInt(Math.pow(10, arg % 10));
               if (arg % 100 < 10) {
                  pave.fOptStat = newopt;
                  this.interactiveRedraw(true, `exec:SetOptStat(${newopt})`);
               } else {
                  pave.fOptFit = newopt;
                  this.interactiveRedraw(true, `exec:SetOptFit(${newopt})`);
               }
            });
         };

         addStatOpt(0, 'Histogram name');
         addStatOpt(1, 'Entries');
         addStatOpt(2, 'Mean');
         addStatOpt(3, 'Std Dev');
         addStatOpt(4, 'Underflow');
         addStatOpt(5, 'Overflow');
         addStatOpt(6, 'Integral');
         addStatOpt(7, 'Skewness');
         addStatOpt(8, 'Kurtosis');
         menu.add('endsub:');

         menu.add('sub:SetOptFit', () => {
            menu.input('Enter OptStat', pave.fOptFit, 'int').then(fmt => {
               pave.fOptFit = fmt;
               this.interactiveRedraw(true, `exec:SetOptFit(${fmt})`);
            });
         });
         addStatOpt(10, 'Fit parameters');
         addStatOpt(11, 'Par errors');
         addStatOpt(12, 'Chi square / NDF');
         addStatOpt(13, 'Probability');
         menu.add('endsub:');

         menu.add('separator');
      } else if (pave._typename === clTLegend) {
         menu.add('Autoplace', () => {
            this.autoPlaceLegend(pave, this.getPadPainter()?.getRootPad(true), true).then(res => {
               if (res) this.interactiveRedraw(true, 'pave_moved');
            });
         });
      } else if (pave.fName === kTitle) {
         menu.add('Default position', () => {
            pave.fX1NDC = gStyle.fTitleW > 0 ? gStyle.fTitleX - gStyle.fTitleW/2 : gStyle.fPadLeftMargin;
            pave.fY1NDC = gStyle.fTitleY - Math.min(gStyle.fTitleFontSize*1.1, 0.06);
            pave.fX2NDC = gStyle.fTitleW > 0 ? gStyle.fTitleX + gStyle.fTitleW/2 : 1 - gStyle.fPadRightMargin;
            pave.fY2NDC = gStyle.fTitleY;
            pave.fInit = 1;
            this.interactiveRedraw(true, 'pave_moved');
         });

         menu.add('Save to gStyle', () => {
            gStyle.fTitleX = (pave.fX2NDC + pave.fX1NDC)/2;
            gStyle.fTitleY = pave.fY2NDC;
            this.fillatt?.saveToStyle('fTitleColor', 'fTitleStyle');
            gStyle.fTitleTextColor = pave.fTextColor;
            gStyle.fTitleFontSize = pave.fTextSize;
            gStyle.fTitleFont = pave.fTextFont;
         }, 'Store title position and graphical attributes to gStyle');
      }

      menu.add('Bring to front', () => this.bringToFront(!this.isStats() && !this.z_handle));
   }

   /** @summary Show pave context menu */
   paveContextMenu(evnt) {
      if (this.z_handle) {
         const fp = this.getFramePainter();
         if (isFunc(fp?.showContextMenu))
             fp.showContextMenu('pal', evnt);
      } else
         showPainterMenu(evnt, this, this.isTitle() ? kTitle : undefined);
   }

   /** @summary Returns true when stat box is drawn */
   isStats() {
      return this.matchObjectType(clTPaveStats);
   }

   /** @summary Returns true when title is drawn */
   isTitle() {
      return this.matchObjectType(clTPaveText) && (this.getObject()?.fName === kTitle);
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
   fillFunctionStat(f1, dofit, ndim = 1) {
      this._has_fit = false;

      if (!dofit || !f1) return false;

      this._has_fit = true;
      this._fit_dim = ndim;
      this._fit_cnt = 0;

      const print_fval = (ndim === 1) ? dofit % 10 : 1,
            print_ferrors = (ndim === 1) ? Math.floor(dofit/10) % 10 : 1,
            print_fchi2 = (ndim === 1) ? Math.floor(dofit/100) % 10 : 1,
            print_fprob = (ndim === 1) ? Math.floor(dofit/1000) % 10 : 0;

      if (print_fchi2) {
         this.addText('#chi^{2} / ndf = ' + this.format(f1.fChisquare, 'fit') + ' / ' + f1.fNDF);
         this._fit_cnt++;
      }
      if (print_fprob) {
         this.addText('Prob = ' + this.format(Prob(f1.fChisquare, f1.fNDF)));
         this._fit_cnt++;
      }
      if (print_fval) {
         for (let n = 0; n < f1.GetNumPars(); ++n) {
            const parname = f1.GetParName(n);
            let parvalue = f1.GetParValue(n), parerr = f1.GetParError(n);

            parvalue = (parvalue === undefined) ? '<not avail>' : this.format(Number(parvalue), 'fit');
            if (parerr !== undefined) {
               parerr = this.format(parerr, 'last');
               if ((Number(parerr) === 0) && (f1.GetParError(n) !== 0))
                  parerr = this.format(f1.GetParError(n), '4.2g');
            }

            if (print_ferrors && parerr)
               this.addText(`${parname} = ${parvalue} #pm ${parerr}`);
            else
               this.addText(`${parname} = ${parvalue}`);
            this._fit_cnt++;
         }
      }


      return true;
   }

   /** @summary Is dummy pos of the pave painter */
   isDummyPos(p) {
      if (!p) return true;

      return !p.fInit && !p.fX1 && !p.fX2 && !p.fY1 && !p.fY2 && !p.fX1NDC && !p.fX2NDC && !p.fY1NDC && !p.fY2NDC;
   }

   /** @summary Update TPave object  */
   updateObject(obj, opt) {
      if (!this.matchObjectType(obj)) return false;

      const pave = this.getObject();

      if (!pave.modified_NDC && !this.isDummyPos(obj)) {
         // if position was not modified interactively, update from source object

         if (this.stored && !obj.fInit && (this.stored.fX1 === obj.fX1) &&
             (this.stored.fX2 === obj.fX2) && (this.stored.fY1 === obj.fY1) && (this.stored.fY2 === obj.fY2)) {
            // case when source object not initialized and original coordinates are not changed
            // take over only modified NDC coordinate, used in tutorials/graphics/canvas.C
            if (this.stored.fX1NDC !== obj.fX1NDC) pave.fX1NDC = obj.fX1NDC;
            if (this.stored.fX2NDC !== obj.fX2NDC) pave.fX2NDC = obj.fX2NDC;
            if (this.stored.fY1NDC !== obj.fY1NDC) pave.fY1NDC = obj.fY1NDC;
            if (this.stored.fY2NDC !== obj.fY2NDC) pave.fY2NDC = obj.fY2NDC;
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
      if (pave.fTextColor !== undefined && obj.fTextColor !== undefined) {
         pave.fTextAngle = obj.fTextAngle;
         pave.fTextSize = obj.fTextSize;
         pave.fTextAlign = obj.fTextAlign;
         pave.fTextColor = obj.fTextColor;
         pave.fTextFont = obj.fTextFont;
      }

      switch (obj._typename) {
         case clTDiamond:
         case clTPaveText:
            pave.fLines = clone(obj.fLines);
            return true;
         case clTPavesText:
            pave.fLines = clone(obj.fLines);
            pave.fNpaves = obj.fNpaves;
            return true;
         case clTPaveLabel:
         case clTPaveClass:
            pave.fLabel = obj.fLabel;
            return true;
         case clTPaveStats:
            pave.fOptStat = obj.fOptStat;
            pave.fOptFit = obj.fOptFit;
            return true;
         case clTLegend: {
            const oldprim = pave.fPrimitives;
            pave.fPrimitives = obj.fPrimitives;
            pave.fNColumns = obj.fNColumns;
            this.AutoPlace = opt === 'autoplace';
            if (oldprim?.arr?.length && (oldprim?.arr?.length === pave.fPrimitives?.arr?.length)) {
               // try to sync object reference, new object does not displayed automatically
               // in ideal case one should use snapids in the entries
               for (let k = 0; k < oldprim.arr.length; ++k) {
                  const oldobj = oldprim.arr[k].fObject, newobj = pave.fPrimitives.arr[k].fObject;
                  if (oldobj && newobj && oldobj._typename === newobj._typename && oldobj.fName === newobj.fName)
                     pave.fPrimitives.arr[k].fObject = oldobj;
               }
            }
            return true;
         }
         case clTPaletteAxis:
            pave.fBorderSize = 1;
            pave.fShadowColor = 0;
            return true;
      }

      return false;
   }

   /** @summary redraw pave object */
   async redraw() {
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
      const typ = obj?._typename;
      return typ === clTPave || typ === clTPaveLabel || typ === clTPaveClass || typ === clTPaveStats || typ === clTPaveText ||
             typ === clTPavesText || typ === clTDiamond || typ === clTLegend || typ === clTPaletteAxis;
   }

   /** @summary Draw TPave */
   static async draw(dom, pave, opt) {
      const painter = new TPavePainter(dom, pave);

      return ensureTCanvas(painter, false).then(() => {
         if ((pave.fName === kTitle) && (pave._typename === clTPaveText)) {
            const tpainter = painter.getPadPainter().findPainterFor(null, kTitle, clTPaveText);
            if (tpainter && (tpainter !== painter)) {
               tpainter.removeFromPadPrimitives();
               tpainter.cleanup();
            } else if ((opt === 'postitle') || painter.isDummyPos(pave)) {
               const st = gStyle, fp = painter.getFramePainter();
               if (st && fp) {
                  const midx = st.fTitleX, y2 = st.fTitleY;
                  let w = st.fTitleW, h = st.fTitleH;

                  if (!h) h = (y2 - fp.fY2NDC) * 0.7;
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
         } else if (pave._typename === clTPaletteAxis) {
            pave.fBorderSize = 1;
            pave.fShadowColor = 0;

            // check some default values of TGaxis object, otherwise axis will not be drawn
            if (pave.fAxis) {
               if (!pave.fAxis.fChopt) pave.fAxis.fChopt = '+';
               if (!pave.fAxis.fNdiv) pave.fAxis.fNdiv = 12;
               if (!pave.fAxis.fLabelOffset) pave.fAxis.fLabelOffset = 0.005;
            }

            painter.z_handle = new TAxisPainter(dom, pave.fAxis, true);
            painter.z_handle.setPadName(painter.getPadName());

            painter.UseContextMenu = true;
         }

         painter.NoFillStats = (opt === 'nofillstats') || (pave.fName !== 'stats');

         switch (pave._typename) {
            case clTPaveLabel:
            case clTPaveClass:
               painter.paveDrawFunc = painter.drawPaveLabel;
               break;
            case clTPaveStats:
               painter.paveDrawFunc = painter.drawPaveStats;
               break;
            case clTPaveText:
            case clTPavesText:
            case clTDiamond:
               painter.paveDrawFunc = painter.drawPaveText;
               break;
            case clTLegend:
               painter.AutoPlace = (opt === 'autoplace');
               painter.paveDrawFunc = painter.drawLegend;
               break;
            case clTPaletteAxis:
               painter.paveDrawFunc = painter.drawPaletteAxis;
               break;
         }

         return painter.drawPave(opt);
      });
   }

} // class TPavePainter


export { TPavePainter };
