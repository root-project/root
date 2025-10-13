import { isStr, clTLatex } from '../core.mjs';
import { THREE, getMaterialArgs, getHelveticaFont } from './base3d.mjs';
import { isPlainText, translateLaTeX, produceLatex } from './latex.mjs';
import { ObjectPainter } from './ObjectPainter.mjs';

class TextParseWrapper {

   constructor(kind, parent, font_size) {
      this.kind = kind ?? 'g';
      this.childs = [];
      this.x = 0;
      this.y = 0;
      this.font_size = parent?.font_size ?? font_size;
      this.stroke_width = parent?.stroke_width ?? 5;
      parent?.childs.push(this);
   }

   append(kind) {
      if (kind === 'svg:g')
         return new TextParseWrapper('g', this);
      if (kind === 'svg:text')
         return new TextParseWrapper('text', this);
      if (kind === 'svg:path')
         return new TextParseWrapper('path', this);
      console.warn('missing handle for svg', kind);
   }

   style(name, value) {
      if ((name === 'stroke-width') && value)
         this.stroke_width = Number.parseInt(value);
      return this;
   }

   property(name, value) {
      if (value === undefined)
         return this[name];
      this[name] = value;
      return this;
   }

   attr(name, value) {
      const get = () => {
         if (!value)
            return '';
         const res = value[0];
         value = value.slice(1);
         return res;
      }, getN = skip => {
         let p = 0;
         while (((value[p] >= '0') && (value[p] <= '9')) || (value[p] === '-'))
            p++;
         const res = Number.parseInt(value.slice(0, p));
         value = value.slice(p);
         if (skip)
            get();
         return res;
      };

      if ((name === 'font-size') && value)
         this.font_size = Number.parseInt(value);
      else if ((name === 'transform') && isStr(value) && (value.indexOf('translate') === 0)) {
         const arr = value.slice(value.indexOf('(') + 1, value.lastIndexOf(')')).split(',');
         this.x += arr[0] ? Number.parseInt(arr[0]) * 0.01 : 0;
         this.y -= arr[1] ? Number.parseInt(arr[1]) * 0.01 : 0;
      } else if ((name === 'x') && (this.kind === 'text'))
         this.x += Number.parseInt(value) * 0.01;
      else if ((name === 'y') && (this.kind === 'text'))
         this.y -= Number.parseInt(value) * 0.01;
      else if ((name === 'fill') && (this.kind === 'text'))
         this.fill = value;
      else if ((name === 'd') && (this.kind === 'path') && (value !== 'M0,0')) {
         if (get() !== 'M')
            return console.error('Not starts with M');
         let x1 = getN(true), y1 = getN(), next;
         const pnts = [], add_line = (x2, y2) => {
            const angle = Math.atan2(y2 - y1, x2 - x1),
                  dx = 0.5 * this.stroke_width * Math.sin(angle),
                  dy = -0.5 * this.stroke_width * Math.cos(angle);
            // front side
            pnts.push(x1 - dx, y1 - dy, 0, x2 - dx, y2 - dy, 0, x2 + dx, y2 + dy, 0, x1 - dx, y1 - dy, 0, x2 + dx, y2 + dy, 0, x1 + dx, y1 + dy, 0);
            // back side
            pnts.push(x1 - dx, y1 - dy, 0, x2 + dx, y2 + dy, 0, x2 - dx, y2 - dy, 0, x1 - dx, y1 - dy, 0, x1 + dx, y1 + dy, 0, x2 + dx, y2 + dy, 0);
            x1 = x2;
            y1 = y2;
         };

         while ((next = get())) {
            switch (next) {
               case 'L':
                  add_line(getN(true), getN());
                  continue;
               case 'l':
                  add_line(x1 + getN(true), y1 + getN());
                  continue;
               case 'H':
                  add_line(getN(), y1);
                  continue;
               case 'h':
                  add_line(x1 + getN(), y1);
                  continue;
               case 'V':
                  add_line(x1, getN());
                  continue;
               case 'v':
                  add_line(x1, y1 + getN());
                  continue;
               case 'a': {
                  const rx = getN(true), ry = getN(true),
                        angle = getN(true) / 180 * Math.PI, flag1 = getN(true);
                  getN(true); // skip unused flag2
                  const x2 = x1 + getN(true),
                        y2 = y1 + getN(),
                        x0 = x1 + rx * Math.cos(angle),
                        y0 = y1 + ry * Math.sin(angle);
                  let angle2 = Math.atan2(y0 - y2, x0 - x2);
                  if (flag1 && (angle2 < angle))
                     angle2 += 2 * Math.PI;
                  else if (!flag1 && (angle2 > angle))
                     angle2 -= 2 * Math.PI;

                  for (let cnt = 0; cnt < 10; ++cnt) {
                     const a = angle + (angle2 - angle) / 10 * (cnt + 1);
                     add_line(x0 - rx * Math.cos(a), y0 - ry * Math.sin(a));
                  }
                  continue;
               }
               default:
                  console.log('not supported path operator', next);
            }
         }

         if (pnts.length) {
            const pos = new Float32Array(pnts);
            this.geom = new THREE.BufferGeometry();
            this.geom.setAttribute('position', new THREE.BufferAttribute(pos, 3));
            this.geom.scale(0.01, -0.01, 0.01);
            this.geom.computeVertexNormals();
         }
      }
      return this;
   }

   text(v) {
      if (this.kind === 'text')
         this._text = v;
   }

   collect(geoms, geom_args, as_array) {
      if (this._text) {
         geom_args.size = Math.round(0.01 * this.font_size);
         const geom = new THREE.TextGeometry(this._text, geom_args);
         if (as_array) {
            // this is latex parsing
            // while three.js uses full height, make it more like normal fonts
            geom.scale(1, 0.9, 1);
            geom.translate(0, 0.0005 * this.font_size, 0);
         }
         geom.translate(this.x, this.y, 0);
         geom._fill = this.fill;
         geoms.push(geom);
      }
      if (this.geom) {
         this.geom.translate(this.x, this.y, 0);
         this.geom._fill = this.fill;
         geoms.push(this.geom);
      }

      this.childs.forEach(chld => {
         chld.x += this.x;
         chld.y += this.y;
         chld.collect(geoms, geom_args, as_array);
      });
   }

} // class TextParseWrapper


function createLatexGeometry(painter, lbl, size, as_array, use_latex = true) {
   const geom_args = { font: getHelveticaFont(), size, height: 0, curveSegments: 5 },
         font_size = size * 100,
         node = new TextParseWrapper('g', null, font_size),
         arg = { font_size, latex: use_latex ? 1 : 0, x: 0, y: 0, text: lbl, align: ['start', 'top'], fast: true, font: { size: font_size, isMonospace: () => false, aver_width: 0.9 } },
         geoms = [];

   if (THREE.REVISION > 162)
      geom_args.depth = 0;
   else
      geom_args.height = 0;

   if (!isPlainText(lbl)) {
      produceLatex(painter, node, arg);
      node.collect(geoms, geom_args, as_array);
   }

   if (!geoms.length) {
      geom_args.size = size;
      const res = new THREE.TextGeometry(translateLaTeX(lbl), geom_args);
      return as_array ? [res] : res;
   }

   if (as_array)
      return geoms;

   if (geoms.length === 1)
      return geoms[0];

   let total_size = 0;
   geoms.forEach(geom => { total_size += geom.getAttribute('position').array.length; });

   const pos = new Float32Array(total_size),
         norm = new Float32Array(total_size);
   let indx = 0;

   geoms.forEach(geom => {
      const p1 = geom.getAttribute('position').array,
            n1 = geom.getAttribute('normal').array;
      for (let i = 0; i < p1.length; ++i, ++indx) {
         pos[indx] = p1[i];
         norm[indx] = n1[i];
      }
   });

   const fullgeom = new THREE.BufferGeometry();
   fullgeom.setAttribute('position', new THREE.BufferAttribute(pos, 3));
   fullgeom.setAttribute('normal', new THREE.BufferAttribute(norm, 3));
   return fullgeom;
}


/** @summary Build three.js object for the TLatex
  * @private */
function build3dlatex(obj, opt, painter, fp) {
   if (!painter)
      painter = new ObjectPainter(null, obj, opt);
   const handle = painter.createAttText({ attr: obj }),
         valign = handle.align % 10,
         halign = (handle.align - valign) / 10,
         text_size = handle.size > 1 ? handle.size : 2 * handle.size * (fp?.size_z3d || 100),
         arr3d = createLatexGeometry(painter, obj.fTitle, text_size || 10, true, fp || (obj._typename === clTLatex)),
         bb = new THREE.Box3().makeEmpty();

   arr3d.forEach(geom => {
      geom.computeBoundingBox();
      bb.expandByPoint(geom.boundingBox.max);
      bb.expandByPoint(geom.boundingBox.min);
   });

   let dx = 0, dy = 0;
   if (halign === 2)
      dx = 0.5 * (bb.max.x + bb.min.x);
   else if (halign === 3)
      dx = bb.max.x;

   if (valign === 2)
      dy = 0.5 * (bb.max.y + bb.min.y);
   else if (valign === 3)
      dy = bb.max.y;

   const obj3d = new THREE.Object3D(),
         materials = [],
         getMaterial = color => {
            if (!color)
               color = 'black';
            if (!materials[color])
               materials[color] = new THREE.MeshBasicMaterial(getMaterialArgs(color, { vertexColors: false }));
            return materials[color];
         };

   arr3d.forEach(geom => {
      geom.translate(-dx, -dy, 0);
      obj3d.add(new THREE.Mesh(geom, getMaterial(geom._fill || handle.color)));
   });

   return arr3d.length === 1 ? obj3d.children[0] : obj3d;
}

export { createLatexGeometry, build3dlatex };
