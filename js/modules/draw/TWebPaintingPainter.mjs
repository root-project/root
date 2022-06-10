import { getColor } from '../base/colors.mjs';
import { ObjectPainter } from '../base/ObjectPainter.mjs';


/** @summary Draw direct TVirtualX commands into SVG
  * @private */

class TWebPaintingPainter extends ObjectPainter {

   updateObject(obj) {
      if (!this.matchObjectType(obj)) return false;
      this.assignObject(obj);
      return true;
   }

   redraw() {

      const obj = this.getObject(), func = this.getAxisToSvgFunc();

      if (!obj || !obj.fOper || !func) return;

      let indx = 0, attr = {}, lastpath = null, lastkind = "none", d = "",
          oper, npoints, n, arr = obj.fOper.split(";");

      const check_attributes = kind => {
         if (kind == lastkind) return;

         if (lastpath) {
            lastpath.attr("d", d); // flush previous
            d = ""; lastpath = null; lastkind = "none";
         }

         if (!kind) return;

         lastkind = kind;
         lastpath = this.draw_g.append("svg:path");
         switch (kind) {
            case "f": lastpath.call(this.fillatt.func); break;
            case "l": lastpath.call(this.lineatt.func).style('fill', 'none'); break;
            case "m": lastpath.call(this.markeratt.func); break;
         }
      }, read_attr = (str, names) => {
         let lastp = 0, obj = { _typename: "any" };
         for (let k = 0; k < names.length; ++k) {
            let p = str.indexOf(":", lastp+1);
            obj[names[k]] = parseInt(str.slice(lastp+1, (p > lastp) ? p : undefined));
            lastp = p;
         }
         return obj;
      }, process = k => {
         while (++k < arr.length) {
            oper = arr[k][0];
            switch (oper) {
               case "z":
                  this.createAttLine({ attr: read_attr(arr[k], ["fLineColor", "fLineStyle", "fLineWidth"]), force: true });
                  check_attributes();
                  continue;
               case "y":
                  this.createAttFill({ attr: read_attr(arr[k], ["fFillColor", "fFillStyle"]), force: true });
                  check_attributes();
                  continue;
               case "x":
                  this.createAttMarker({ attr: read_attr(arr[k], ["fMarkerColor", "fMarkerStyle", "fMarkerSize"]), force: true });
                  check_attributes();
                  continue;
               case "o":
                  attr = read_attr(arr[k], ["fTextColor", "fTextFont", "fTextSize", "fTextAlign", "fTextAngle"]);
                  if (attr.fTextSize < 0) attr.fTextSize *= -0.001;
                  check_attributes();
                  continue;
               case "r":
               case "b": {

                  check_attributes((oper == "b") ? "f" : "l");

                  let x1 = func.x(obj.fBuf[indx++]),
                      y1 = func.y(obj.fBuf[indx++]),
                      x2 = func.x(obj.fBuf[indx++]),
                      y2 = func.y(obj.fBuf[indx++]);

                  d += `M${x1},${y1}h${x2-x1}v${y2-y1}h${x1-x2}z`;

                  continue;
               }
               case "l":
               case "f": {

                  check_attributes(oper);

                  npoints = parseInt(arr[k].slice(1));

                  for (n = 0; n < npoints; ++n)
                     d += ((n > 0) ? "L" : "M") +
                           func.x(obj.fBuf[indx++]) + "," + func.y(obj.fBuf[indx++]);

                  if (oper == "f") d+="Z";

                  continue;
               }

               case "m": {

                  check_attributes(oper);

                  npoints = parseInt(arr[k].slice(1));

                  this.markeratt.resetPos();
                  for (n = 0; n < npoints; ++n)
                     d += this.markeratt.create(func.x(obj.fBuf[indx++]), func.y(obj.fBuf[indx++]));

                  continue;
               }

               case "h":
               case "t": {
                  if (attr.fTextSize) {

                     check_attributes();

                     let height = (attr.fTextSize > 1) ? attr.fTextSize : this.getPadPainter().getPadHeight() * attr.fTextSize,
                         angle = attr.fTextAngle,
                         txt = arr[k].slice(1),
                         group = this.draw_g.append("svg:g");

                     if (angle >= 360) angle -= Math.floor(angle/360) * 360;

                     this.startTextDrawing(attr.fTextFont, height, group);

                     if (oper == "h") {
                        let res = "";
                        for (n = 0; n < txt.length; n += 2)
                           res += String.fromCharCode(parseInt(txt.slice(n,n+2), 16));
                        txt = res;
                     }

                     // todo - correct support of angle
                     this.drawText({ align: attr.fTextAlign,
                                     x: func.x(obj.fBuf[indx++]),
                                     y: func.y(obj.fBuf[indx++]),
                                     rotate: -angle,
                                     text: txt,
                                     color: getColor(attr.fTextColor),
                                     latex: 0, draw_g: group });

                     return this.finishTextDrawing(group).then(() => process(k));
                  }
                  continue;
               }

               default:
                  console.log('unsupported operation ' + oper);
            }
         };

         return Promise.resolve(true);
      }

      this.createG();

      return process(-1).then(() => { check_attributes(); return this; });
   }

   static draw(dom, obj) {
      let painter = new TWebPaintingPainter(dom, obj);
      painter.addToPadPrimitives();
      return painter.redraw();
   }

} // class TWebPaintingPainter

export { TWebPaintingPainter };
