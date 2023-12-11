import { isStr, isFunc, nsREX } from '../core.mjs';
import { FontHandler } from './FontHandler.mjs';
import { ObjectPainter } from './ObjectPainter.mjs';


const kNormal = 1, /* kLessTraffic = 2, */ kOffline = 3;

class RObjectPainter extends ObjectPainter {

   constructor(dom, obj, opt, csstype) {
      super(dom, obj, opt);
      this.csstype = csstype;
   }

   /** @summary Evaluate v7 attributes using fAttr storage and configured RStyle */
   v7EvalAttr(name, dflt) {
      const obj = this.getObject();
      if (!obj) return dflt;
      if (this.cssprefix) name = this.cssprefix + name;

      const type_check = res => {
         if (dflt === undefined) return res;
         const typ1 = typeof dflt, typ2 = typeof res;
         if (typ1 === typ2) return res;
         if (typ1 === 'boolean') {
            if (typ2 === 'string') return (res !== '') && (res !== '0') && (res !== 'no') && (res !== 'off');
            return !!res;
         }
         if ((typ1 === 'number') && (typ2 === 'string'))
            return parseFloat(res);
         return res;
      };

      if (obj.fAttr?.m) {
         const value = obj.fAttr.m[name];
         if (value) return type_check(value.v); // found value direct in attributes
      }

      if (this.rstyle?.fBlocks) {
         const blks = this.rstyle.fBlocks;
         for (let k = 0; k < blks.length; ++k) {
            const block = blks[k],
                  match = (this.csstype && (block.selector === this.csstype)) ||
                          (obj.fId && (block.selector === ('#' + obj.fId))) ||
                          (obj.fCssClass && (block.selector === ('.' + obj.fCssClass)));

            if (match && block.map?.m) {
               const value = block.map.m[name.toLowerCase()];
               if (value) return type_check(value.v);
            }
         }
      }

      return dflt;
   }

   /** @summary Set v7 attributes value */
   v7SetAttr(name, value) {
      const obj = this.getObject();
      if (this.cssprefix) name = this.cssprefix + name;

      if (obj?.fAttr?.m)
         obj.fAttr.m[name] = { v: value };
   }

   /** @summary Decode pad length from string, return pixel value */
   v7EvalLength(name, sizepx, dflt) {
      if (sizepx <= 0) sizepx = 1;

      const value = this.v7EvalAttr(name);

      if (value === undefined)
         return Math.round(dflt*sizepx);

      if (typeof value === 'number')
         return Math.round(value*sizepx);

      if (value === null)
         return 0;

      let norm = 0, px = 0, val = value, operand = 0, pos = 0;

      while (val) {
         // skip empty spaces
         while ((pos < val.length) && ((val[pos] === ' ') || (val[pos] === '\t')))
            ++pos;

         if (pos >= val.length)
            break;

         if ((val[pos] === '-') || (val[pos] === '+')) {
            if (operand) {
               console.log('Fail to parse RPadLength ' + value);
               return dflt;
            }
            operand = (val[pos] === '-') ? -1 : 1;
            pos++;
            continue;
         }

         if (pos > 0) { val = val.slice(pos); pos = 0; }

         while ((pos < val.length) && (((val[pos] >= '0') && (val[pos] <= '9')) || (val[pos] === '.'))) pos++;

         const v = parseFloat(val.slice(0, pos));
         if (!Number.isFinite(v)) {
            console.log(`Fail to parse RPadLength ${value}`);
            return Math.round(dflt*sizepx);
         }

         val = val.slice(pos);
         pos = 0;
         if (!operand) operand = 1;
         if (val && (val[0] === '%')) {
            val = val.slice(1);
            norm += operand*v*0.01;
         } else if ((val.length > 1) && (val[0] === 'p') && (val[1] === 'x')) {
            val = val.slice(2);
            px += operand*v;
         } else
            norm += operand*v;

         operand = 0;
      }

      return Math.round(norm*sizepx + px);
   }

   /** @summary Evaluate RColor using attribute storage and configured RStyle */
   v7EvalColor(name, dflt) {
      let val = this.v7EvalAttr(name, '');
      if (!val || !isStr(val)) return dflt;

      if (val === 'auto') {
         const pp = this.getPadPainter();
         if (pp?._auto_color_cnt !== undefined) {
            const pal = pp.getHistPalette(),
                  cnt = pp._auto_color_cnt++;
            let num = pp._num_primitives - 1;
            if (num < 2) num = 2;
            val = pal ? pal.getColorOrdinal((cnt % num) / num) : 'blue';
            if (!this._auto_colors) this._auto_colors = {};
            this._auto_colors[name] = val;
         } else if (this._auto_colors && this._auto_colors[name])
            val = this._auto_colors[name];
         else {
            console.error(`Autocolor ${name} not defined yet - please check code`);
            val = '';
         }
      } else if (val[0] === '[') {
         const ordinal = parseFloat(val.slice(1, val.length-1));
         val = 'black';
         if (Number.isFinite(ordinal)) {
             const pal = this.getPadPainter()?.getHistPalette();
             if (pal) val = pal.getColorOrdinal(ordinal);
         }
      }
      return val;
   }

   /** @summary Evaluate RAttrText properties
     * @return {Object} FontHandler, can be used directly for the text drawing */
   v7EvalFont(name, dflts, fontScale) {
      if (!dflts)
         dflts = {};
      else if (typeof dflts === 'number')
         dflts = { size: dflts };

      const pp = this.getPadPainter(),
            rfont = pp?._dfltRFont || { fFamily: 'Arial', fStyle: '', fWeight: '' },
            text_angle = this.v7EvalAttr(name + '_angle', 0),
            text_align = this.v7EvalAttr(name + '_align', dflts.align || 'none'),
            text_color = this.v7EvalColor(name + '_color', dflts.color || 'none'),
            font_family = this.v7EvalAttr(name + '_font_family', rfont.fFamily || 'Arial'),
            font_style = this.v7EvalAttr(name + '_font_style', rfont.fStyle || ''),
            font_weight = this.v7EvalAttr(name + '_font_weight', rfont.fWeight || '');
       let text_size = this.v7EvalAttr(name + '_size', dflts.size || 12);

       if (isStr(text_size)) text_size = parseFloat(text_size);
       if (!Number.isFinite(text_size) || (text_size <= 0)) text_size = 12;
       if (!fontScale) fontScale = pp?.getPadHeight() || 100;

       const handler = new FontHandler(null, text_size, fontScale);
       handler.setNameStyleWeight(font_family, font_style, font_weight);

       if (text_angle) handler.setAngle(360 - text_angle);
       if (text_align !== 'none') handler.setAlign(text_align);
       if (text_color !== 'none') handler.setColor(text_color);

       return handler;
    }

   /** @summary Create this.fillatt object based on v7 fill attributes */
   createv7AttFill(prefix) {
      if (!prefix || !isStr(prefix)) prefix = 'fill_';

      const color = this.v7EvalColor(prefix + 'color', ''),
            pattern = this.v7EvalAttr(prefix + 'style', 0);

      this.createAttFill({ pattern, color, color_as_svg: true });
   }

   /** @summary Create this.lineatt object based on v7 line attributes */
   createv7AttLine(prefix) {
      if (!prefix || !isStr(prefix)) prefix = 'line_';

      const color = this.v7EvalColor(prefix + 'color', 'black'),
            width = this.v7EvalAttr(prefix + 'width', 1),
            style = this.v7EvalAttr(prefix + 'style', 1),
            pattern = this.v7EvalAttr(prefix + 'pattern');

      this.createAttLine({ color, width, style, pattern });

      if (prefix === 'border_')
         this.lineatt.setBorder(this.v7EvalAttr(prefix + 'rx', 0), this.v7EvalAttr(prefix + 'ry', 0));
   }

    /** @summary Create this.markeratt object based on v7 attributes */
   createv7AttMarker(prefix) {
      if (!prefix || !isStr(prefix)) prefix = 'marker_';

      const color = this.v7EvalColor(prefix + 'color', 'black'),
            size = this.v7EvalAttr(prefix + 'size', 0.01),
            style = this.v7EvalAttr(prefix + 'style', 1),
            refsize = (size >= 1) ? 1 : (this.getPadPainter()?.getPadHeight() || 100);

      this.createAttMarker({ color, size, style, refsize });
   }

   /** @summary Create RChangeAttr, which can be applied on the server side
     * @private */
   v7AttrChange(req, name, value, kind) {
      if (!this.snapid)
         return false;

      if (!req._typename) {
         req._typename = `${nsREX}RChangeAttrRequest`;
         req.ids = [];
         req.names = [];
         req.values = [];
         req.update = true;
      }

      if (this.cssprefix) name = this.cssprefix + name;
      req.ids.push(this.snapid);
      req.names.push(name);
      let obj = null;

      if ((value === null) || (value === undefined)) {
        if (!kind) kind = 'none';
        if (kind !== 'none') console.error(`Trying to set ${kind} for none value`);
      }

      if (!kind) {
         switch (typeof value) {
            case 'number': kind = 'double'; break;
            case 'boolean': kind = 'boolean'; break;
         }
      }

      obj = { _typename: `${nsREX}RAttrMap::` };
      switch (kind) {
         case 'none': obj._typename += 'NoValue_t'; break;
         case 'boolean': obj._typename += 'BoolValue_t'; obj.v = !!value; break;
         case 'int': obj._typename += 'IntValue_t'; obj.v = parseInt(value); break;
         case 'double': obj._typename += 'DoubleValue_t'; obj.v = parseFloat(value); break;
         default: obj._typename += 'StringValue_t'; obj.v = isStr(value) ? value : JSON.stringify(value); break;
      }

      req.values.push(obj);
      return true;
   }

   /** @summary Sends accumulated attribute changes to server */
   v7SendAttrChanges(req, do_update) {
      const canp = this.getCanvPainter();
      if (canp && req?._typename) {
         if (do_update !== undefined)
            req.update = !!do_update;
         canp.v7SubmitRequest('', req);
      }
   }

   /** @summary Submit request to server-side drawable
    * @param kind defines request kind, only single request a time can be submitted
    * @param req is object derived from DrawableRequest, including correct _typename
    * @param method is method of painter object which will be called when getting reply */
   v7SubmitRequest(kind, req, method) {
      const canp = this.getCanvPainter();
      if (!isFunc(canp?.submitDrawableRequest)) return null;

      // special situation when snapid not yet assigned - just keep ref until snapid is there
      // maybe keep full list - for now not clear if really needed
      if (!this.snapid) {
         this._pending_request = { kind, req, method };
         return req;
      }

      return canp.submitDrawableRequest(kind, req, this, method);
   }

   /** @summary Assign snapid to the painter
     * @desc Overwrite default method */
   assignSnapId(id) {
      this.snapid = id;
      if (this.snapid && this._pending_request) {
         const p = this._pending_request;
         this.v7SubmitRequest(p.kind, p.req, p.method);
         delete this._pending_request;
      }
   }

   /** @summary Return communication mode with the server
     * @desc
     * kOffline means no server there,
     * kLessTraffic advise not to send commands if offline functionality available
     * kNormal is standard functionality with RCanvas on server side */
   v7CommMode() {
      const canp = this.getCanvPainter();
      if (!canp || !canp.submitDrawableRequest || !canp._websocket)
         return kOffline;

      return kNormal;
   }

   v7NormalMode() { return this.v7CommMode() === kNormal; }

   v7OfflineMode() { return this.v7CommMode() === kOffline; }

} // class RObjectPainter

export { RObjectPainter };
