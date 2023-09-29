/// CSG library for THREE.js

import { BufferGeometry, BufferAttribute, Mesh } from '../three.mjs';

const EPSILON = 1e-5,
      COPLANAR = 0,
      FRONT = 1,
      BACK = 2,
      SPANNING = FRONT | BACK;

class Vertex {

   constructor(x, y, z, nx, ny, nz) {
      this.x = x;
      this.y = y;
      this.z = z;
      this.nx = nx;
      this.ny = ny;
      this.nz = nz;
   }

   setnormal(nx, ny, nz) {
      this.nx = nx;
      this.ny = ny;
      this.nz = nz;
   }

   clone() {
      return new Vertex(this.x, this.y, this.z, this.nx, this.ny, this.nz);
   }

   add(vertex) {
      this.x += vertex.x;
      this.y += vertex.y;
      this.z += vertex.z;
      return this;
   }

   subtract(vertex) {
      this.x -= vertex.x;
      this.y -= vertex.y;
      this.z -= vertex.z;
      return this;
   }

   // multiplyScalar( scalar ) {
   //   this.x *= scalar;
   //   this.y *= scalar;
   //   this.z *= scalar;
   //   return this;
   // }

   // cross( vertex ) {
   //    let x = this.x, y = this.y, z = this.z,
   //        vx = vertex.x, vy = vertex.y, vz = vertex.z;
   //
   //    this.x = y * vz - z * vy;
   //    this.y = z * vx - x * vz;
   //    this.z = x * vy - y * vx;
   //
   //    return this;
   // }

   cross3(vx, vy, vz) {
      const x = this.x, y = this.y, z = this.z;

      this.x = y * vz - z * vy;
      this.y = z * vx - x * vz;
      this.z = x * vy - y * vx;

      return this;
   }


   normalize() {
      const length = Math.sqrt(this.x**2 + this.y**2 + this.z**2);

      this.x /= length;
      this.y /= length;
      this.z /= length;

      return this;
   }

   dot(vertex) {
      return this.x*vertex.x + this.y*vertex.y + this.z*vertex.z;
   }

   diff(vertex) {
      const dx = (this.x - vertex.x),
          dy = (this.y - vertex.y),
          dz = (this.z - vertex.z),
          len2 = this.x**2 + this.y**2 + this.z**2;

      return (dx**2 + dy**2 + dz**2) / (len2 > 0 ? len2 : 1e-10);
   }

/*
   lerp( a, t ) {
      this.add(
         a.clone().subtract( this ).multiplyScalar( t )
      );

      this.normal.add(
         a.normal.clone().sub( this.normal ).multiplyScalar( t )
      );

      //this.uv.add(
      //   a.uv.clone().sub( this.uv ).multiplyScalar( t )
      //);

      return this;
   };

   interpolate( other, t ) {
      return this.clone().lerp( other, t );
   };
*/

   interpolate(a, t) {
      const t1 = 1 - t;
      return new Vertex(this.x*t1 + a.x*t, this.y*t1 + a.y*t, this.z*t1 + a.z*t,
                        this.nx*t1 + a.nx*t, this.ny*t1 + a.ny*t, this.nz*t1 + a.nz*t);
   }

   applyMatrix4(m) {
      // input: Matrix4 affine matrix

      let x = this.x, y = this.y, z = this.z;
      const e = m.elements;

      this.x = e[0] * x + e[4] * y + e[8] * z + e[12];
      this.y = e[1] * x + e[5] * y + e[9] * z + e[13];
      this.z = e[2] * x + e[6] * y + e[10] * z + e[14];

      x = this.nx; y = this.ny; z = this.nz;

      this.nx = e[0] * x + e[4] * y + e[8] * z;
      this.ny = e[1] * x + e[5] * y + e[9] * z;
      this.nz = e[2] * x + e[6] * y + e[10] * z;

      return this;
   }

} // class Vertex


class Polygon {

   constructor(vertices, parent, more) {
      this.vertices = vertices || [];
      this.nsign = 1;
      if (parent)
         this.copyProperties(parent, more);
      else if (this.vertices.length > 0)
         this.calculateProperties();
   }

   copyProperties(parent, more) {
      this.normal = parent.normal; // .clone();
      this.w = parent.w;
      this.nsign = parent.nsign;
      if (more && (parent.id !== undefined)) {
         this.id = parent.id;
         this.parent = parent;
      }
      return this;
   }

   calculateProperties(force) {
      if (this.normal && !force) return;

      const a = this.vertices[0],
          b = this.vertices[1],
          c = this.vertices[2];

      this.nsign = 1;

      // this.normal = b.clone().subtract(a).cross(c.clone().subtract(a)).normalize();

      this.normal = new Vertex(b.x - a.x, b.y - a.y, b.z - a.z, 0, 0, 0).cross3(c.x - a.x, c.y - a.y, c.z - a.z).normalize();

      this.w = this.normal.dot(a);
      return this;
   }

   clone() {
      const vertice_count = this.vertices.length,
          vertices = [];

      for (let i = 0; i < vertice_count; ++i)
         vertices.push(this.vertices[i].clone());

      return new Polygon(vertices, this);
   }

   flip() {
      /// normal is not changed, only sign variable
      // this.normal.multiplyScalar( -1 );
      // this.w *= -1;

      this.nsign *= -1;

      this.vertices.reverse();

      return this;
   }

   classifyVertex(vertex) {
      const side_value = this.nsign * (this.normal.dot(vertex) - this.w);

      if (side_value < -EPSILON) return BACK;
      if (side_value > EPSILON) return FRONT;
      return COPLANAR;
   }

   classifySide(polygon) {
      let num_positive = 0, num_negative = 0;
      const vertice_count = polygon.vertices.length;

      for (let i = 0; i < vertice_count; ++i) {
         const classification = this.classifyVertex(polygon.vertices[i]);
         if (classification === FRONT)
            ++num_positive;
          else if (classification === BACK)
            ++num_negative;
      }

      if (num_positive > 0 && num_negative === 0) return FRONT;
      if (num_positive === 0 && num_negative > 0) return BACK;
      if (num_positive === 0 && num_negative === 0) return COPLANAR;
      return SPANNING;
   }

   splitPolygon(polygon, coplanar_front, coplanar_back, front, back) {
      const classification = this.classifySide(polygon);

      if (classification === COPLANAR)

         ((this.nsign * polygon.nsign * this.normal.dot(polygon.normal) > 0) ? coplanar_front : coplanar_back).push(polygon);

       else if (classification === FRONT)

         front.push(polygon);

       else if (classification === BACK)

         back.push(polygon);

       else {
         const vertice_count = polygon.vertices.length,
               nnx = this.normal.x,
               nny = this.normal.y,
               nnz = this.normal.z,
               f = [], b = [];
          let i, j, ti, tj, vi, vj, t, v;

         for (i = 0; i < vertice_count; ++i) {
            j = (i + 1) % vertice_count;
            vi = polygon.vertices[i];
            vj = polygon.vertices[j];
            ti = this.classifyVertex(vi);
            tj = this.classifyVertex(vj);

            if (ti !== BACK) f.push(vi);
            if (ti !== FRONT) b.push(vi);
            if ((ti | tj) === SPANNING) {
               // t = (this.w - this.normal.dot(vi))/this.normal.dot(vj.clone().subtract(vi));
               // v = vi.clone().lerp( vj, t );

               t = (this.w - (nnx*vi.x + nny*vi.y + nnz*vi.z)) / (nnx*(vj.x-vi.x) + nny*(vj.y-vi.y) + nnz*(vj.z-vi.z));

               v = vi.interpolate(vj, t);
               f.push(v);
               b.push(v);
            }
         }

         // if ( f.length >= 3 ) front.push(new Polygon(f).calculateProperties());
         // if ( b.length >= 3 ) back.push(new Polygon(b).calculateProperties());
         if (f.length >= 3) front.push(new Polygon(f, polygon, true));
         if (b.length >= 3) back.push(new Polygon(b, polygon, true));
      }
   }

} // class Polygon


class Node {

   constructor(polygons, nodeid) {
      this.polygons = [];
      this.front = this.back = undefined;

      if (!polygons) return;

      this.divider = polygons[0].clone();

      const polygon_count = polygons.length,
          front = [], back = [];

      for (let i = 0; i < polygon_count; ++i) {
         if (nodeid !== undefined) {
            polygons[i].id = nodeid++;
            delete polygons[i].parent;
         }

         // by difinition polygon should be COPLANAR for itself
         if (i === 0)
            this.polygons.push(polygons[0]);
         else
            this.divider.splitPolygon(polygons[i], this.polygons, this.polygons, front, back);
      }

      if (nodeid !== undefined) this.maxnodeid = nodeid;

      if (front.length > 0)
         this.front = new Node(front);

      if (back.length > 0)
         this.back = new Node(back);
   }

   // isConvex(polygons) {
   //   let i, j, len = polygons.length;
   //   for ( i = 0; i < len; ++i )
   //      for ( j = 0; j < len; ++j )
   //         if ( i !== j && polygons[i].classifySide( polygons[j] ) !== BACK ) return false;
   //   return true;
   // }

   build(polygons) {
      const polygon_count = polygons.length,
            front = [], back = [];
      let first = 0;

      if (!this.divider) {
         this.divider = polygons[0].clone();
         this.polygons.push(polygons[0]);
         first = 1;
      }

      for (let i = first; i < polygon_count; ++i)
         this.divider.splitPolygon(polygons[i], this.polygons, this.polygons, front, back);

      if (front.length > 0) {
         if (!this.front) this.front = new Node();
         this.front.build(front);
      }

      if (back.length > 0) {
         if (!this.back) this.back = new Node();
         this.back.build(back);
      }
   }

   collectPolygons(arr) {
      if (arr === undefined)
         arr = [];
      const len = this.polygons.length;
      for (let i = 0; i < len; ++i)
         arr.push(this.polygons[i]);
      this.front?.collectPolygons(arr);
      this.back?.collectPolygons(arr);
      return arr;
   }

   numPolygons() {
      return this.polygons.length + (this.front?.numPolygons() || 0) + (this.back?.numPolygons() || 0);
   }

   clone() {
      const node = new Node();

      node.divider = this.divider?.clone();
      node.polygons = this.polygons.map(polygon => polygon.clone());
      node.front = this.front?.clone();
      node.back = this.back?.clone();

      return node;
   }

   invert() {
      const polygon_count = this.polygons.length;

      for (let i = 0; i < polygon_count; ++i)
         this.polygons[i].flip();

      this.divider.flip();
      if (this.front) this.front.invert();
      if (this.back) this.back.invert();

      const temp = this.front;
      this.front = this.back;
      this.back = temp;

      return this;
   }

   clipPolygons(polygons) {
      if (!this.divider) return polygons.slice();

      const polygon_count = polygons.length;
      let front = [], back = [];

      for (let i = 0; i < polygon_count; ++i)
         this.divider.splitPolygon(polygons[i], front, back, front, back);

      if (this.front) front = this.front.clipPolygons(front);
      if (this.back) back = this.back.clipPolygons(back);
      else back = [];

      return front.concat(back);
   }

   clipTo(node) {
      this.polygons = node.clipPolygons(this.polygons);
      if (this.front) this.front.clipTo(node);
      if (this.back) this.back.clipTo(node);
   }

 } // class Node


function createBufferGeometry(polygons) {
   const polygon_count = polygons.length;
   let i, j, buf_size = 0;

   for (i = 0; i < polygon_count; ++i)
      buf_size += (polygons[i].vertices.length - 2) * 9;

   const positions_buf = new Float32Array(buf_size),
         normals_buf = new Float32Array(buf_size);
   let iii = 0, polygon;

   function CopyVertex(vertex) {
      positions_buf[iii] = vertex.x;
      positions_buf[iii+1] = vertex.y;
      positions_buf[iii+2] = vertex.z;

      normals_buf[iii] = polygon.nsign * vertex.nx;
      normals_buf[iii+1] = polygon.nsign * vertex.ny;
      normals_buf[iii+2] = polygon.nsign * vertex.nz;
      iii+=3;
   }

   for (i = 0; i < polygon_count; ++i) {
      polygon = polygons[i];
      for (j = 2; j < polygon.vertices.length; ++j) {
         CopyVertex(polygon.vertices[0]);
         CopyVertex(polygon.vertices[j-1]);
         CopyVertex(polygon.vertices[j]);
      }
   }

   const geometry = new BufferGeometry();
   geometry.setAttribute('position', new BufferAttribute(positions_buf, 3));
   geometry.setAttribute('normal', new BufferAttribute(normals_buf, 3));

   // geometry.computeVertexNormals();
   return geometry;
}


class Geometry {

   constructor(geometry, transfer_matrix, nodeid, flippedMesh) {
      // Convert BufferGeometry to ThreeBSP

      if (geometry instanceof Mesh) {
         // #todo: add hierarchy support
         geometry.updateMatrix();
         transfer_matrix = this.matrix = geometry.matrix.clone();
         geometry = geometry.geometry;
      } else if (geometry instanceof Node) {
         this.tree = geometry;
         this.matrix = null; // new Matrix4;
         return this;
      } else if (geometry instanceof BufferGeometry) {
         const pos_buf = geometry.getAttribute('position').array,
               norm_buf = geometry.getAttribute('normal').array,
               polygons = [];
         let polygon, vert1, vert2, vert3;

         for (let i=0; i < pos_buf.length; i+=9) {
            polygon = new Polygon();

            vert1 = new Vertex(pos_buf[i], pos_buf[i+1], pos_buf[i+2], norm_buf[i], norm_buf[i+1], norm_buf[i+2]);
            if (transfer_matrix) vert1.applyMatrix4(transfer_matrix);

            vert2 = new Vertex(pos_buf[i+3], pos_buf[i+4], pos_buf[i+5], norm_buf[i+3], norm_buf[i+4], norm_buf[i+5]);
            if (transfer_matrix) vert2.applyMatrix4(transfer_matrix);

            vert3 = new Vertex(pos_buf[i+6], pos_buf[i+7], pos_buf[i+8], norm_buf[i+6], norm_buf[i+7], norm_buf[i+8]);
            if (transfer_matrix) vert3.applyMatrix4(transfer_matrix);

            if (flippedMesh) polygon.vertices.push(vert1, vert3, vert2);
                        else polygon.vertices.push(vert1, vert2, vert3);

            polygon.calculateProperties(true);
            polygons.push(polygon);
         }

         this.tree = new Node(polygons, nodeid);
         if (nodeid !== undefined) this.maxid = this.tree.maxnodeid;
         return this;
      } else if (geometry.polygons && (geometry.polygons[0] instanceof Polygon)) {
         const polygons = geometry.polygons;

         for (let i = 0; i < polygons.length; ++i) {
            const polygon = polygons[i];
            if (transfer_matrix) {
               const new_vertices = [];

               for (let n = 0; n < polygon.vertices.length; ++n)
                  new_vertices.push(polygon.vertices[n].clone().applyMatrix4(transfer_matrix));

               polygon.vertices = new_vertices;
            }

            polygon.calculateProperties(transfer_matrix);
         }

         this.tree = new Node(polygons, nodeid);
         if (nodeid !== undefined) this.maxid = this.tree.maxnodeid;
         return this;
      } else
         throw Error('ThreeBSP: Given geometry is unsupported');


      const polygons = [], nfaces = geometry.faces.length;
      let face, polygon, vertex, normal, useVertexNormals;

      for (let i = 0; i < nfaces; ++i) {
         face = geometry.faces[i];
         normal = face.normal;
         // faceVertexUvs = geometry.faceVertexUvs[0][i];
         polygon = new Polygon();

         useVertexNormals = face.vertexNormals && (face.vertexNormals.length === 3);

         vertex = geometry.vertices[face.a];
         if (useVertexNormals) normal = face.vertexNormals[0];
         // uvs = faceVertexUvs ? new Vector2( faceVertexUvs[0].x, faceVertexUvs[0].y ) : null;
         vertex = new Vertex(vertex.x, vertex.y, vertex.z, normal.x, normal.y, normal.z /* face.normal, uvs */);
         if (transfer_matrix) vertex.applyMatrix4(transfer_matrix);
         polygon.vertices.push(vertex);

         vertex = geometry.vertices[face.b];
         if (useVertexNormals) normal = face.vertexNormals[1];
         // uvs = faceVertexUvs ? new Vector2( faceVertexUvs[1].x, faceVertexUvs[1].y ) : null;
         vertex = new Vertex(vertex.x, vertex.y, vertex.z, normal.x, normal.y, normal.z /* face.normal, uvs */);
         if (transfer_matrix) vertex.applyMatrix4(transfer_matrix);
         polygon.vertices.push(vertex);

         vertex = geometry.vertices[face.c];
         if (useVertexNormals) normal = face.vertexNormals[2];
         // uvs = faceVertexUvs ? new Vector2( faceVertexUvs[2].x, faceVertexUvs[2].y ) : null;
         vertex = new Vertex(vertex.x, vertex.y, vertex.z, normal.x, normal.y, normal.z /* face.normal, uvs */);
         if (transfer_matrix) vertex.applyMatrix4(transfer_matrix);
         polygon.vertices.push(vertex);

         polygon.calculateProperties(true);
         polygons.push(polygon);
      }

      this.tree = new Node(polygons, nodeid);
      if (nodeid !== undefined) this.maxid = this.tree.maxnodeid;
   }

   subtract(other_tree) {
      let a = this.tree.clone();
      const b = other_tree.tree.clone();

      a.invert();
      a.clipTo(b);
      b.clipTo(a);
      b.invert();
      b.clipTo(a);
      b.invert();
      a.build(b.collectPolygons());
      a.invert();
      a = new Geometry(a);
      a.matrix = this.matrix;
      return a;
   }

   union(other_tree) {
      let a = this.tree.clone();
      const b = other_tree.tree.clone();

      a.clipTo(b);
      b.clipTo(a);
      b.invert();
      b.clipTo(a);
      b.invert();
      a.build(b.collectPolygons());
      a = new Geometry(a);
      a.matrix = this.matrix;
      return a;
   }

   intersect(other_tree) {
      let a = this.tree.clone();
      const b = other_tree.tree.clone();

      a.invert();
      b.clipTo(a);
      b.invert();
      a.clipTo(b);
      b.clipTo(a);
      a.build(b.collectPolygons());
      a.invert();
      a = new Geometry(a);
      a.matrix = this.matrix;
      return a;
   }

   tryToCompress(polygons) {
      if (this.maxid === undefined) return;

      const arr = [];
      let parts, foundpair,
          nreduce = 0, n, len = polygons.length,
          p, p1, p2, i1, i2;

      // sort out polygons
      for (n = 0; n < len; ++n) {
         p = polygons[n];
         if (p.id === undefined) continue;
         if (arr[p.id] === undefined) arr[p.id] = [];

         arr[p.id].push(p);
      }

      for (n = 0; n < arr.length; ++n) {
         parts = arr[n];
         if (parts === undefined) continue;

         len = parts.length;

         foundpair = (len > 1);

         while (foundpair) {
            foundpair = false;

            for (i1 = 0; i1 < len-1; ++i1) {
               p1 = parts[i1];
               if (!p1?.parent) continue;
               for (i2 = i1+1; i2 < len; ++i2) {
                  p2 = parts[i2];
                  if (p2 && (p1.parent === p2.parent) && (p1.nsign === p2.nsign)) {
                     if (p1.nsign !== p1.parent.nsign)
                        p1.parent.flip();
                     nreduce++;
                     parts[i1] = p1.parent;
                     parts[i2] = null;
                     if (p1.parent.vertices.length < 3) console.log('something wrong with parent');
                     foundpair = true;
                     break;
                  }
               }
            }
         }
      }

      if (nreduce > 0) {
         polygons.splice(0, polygons.length);

         for (n = 0; n < arr.length; ++n) {
            parts = arr[n];
            if (parts !== undefined) {
               for (i1 = 0, len = parts.length; i1 < len; ++i1)
                  if (parts[i1]) polygons.push(parts[i1]);
            }
         }
      }
   }

   direct_subtract(other_tree) {
      const a = this.tree,
            b = other_tree.tree;
      a.invert();
      a.clipTo(b);
      b.clipTo(a);
      b.invert();
      b.clipTo(a);
      b.invert();
      a.build(b.collectPolygons());
      a.invert();
      return this;
   }

   direct_union(other_tree) {
      const a = this.tree,
            b = other_tree.tree;

      a.clipTo(b);
      b.clipTo(a);
      b.invert();
      b.clipTo(a);
      b.invert();
      a.build(b.collectPolygons());
      return this;
   }

   direct_intersect(other_tree) {
      const a = this.tree,
            b = other_tree.tree;

      a.invert();
      b.clipTo(a);
      b.invert();
      a.clipTo(b);
      b.clipTo(a);
      a.build(b.collectPolygons());
      a.invert();
      return this;
   }

   cut_from_plane(other_tree) {
      // just cut peaces from second geometry, which just simple plane

      const a = this.tree,
            b = other_tree.tree;

      a.invert();
      b.clipTo(a);

      return this;
   }

   scale(x, y, z) {
      // try to scale as BufferGeometry
      const polygons = this.tree.collectPolygons();

      for (let i = 0; i < polygons.length; ++i) {
         const polygon = polygons[i];
         for (let k = 0; k < polygon.vertices.length; ++k) {
            const v = polygon.vertices[k];
            v.x *= x;
            v.y *= y;
            v.z *= z;
         }
         polygon.calculateProperties(true);
      }
   }

   toPolygons() {
      const polygons = this.tree.collectPolygons();

      this.tryToCompress(polygons);

      for (let i = 0; i < polygons.length; ++i) {
         delete polygons[i].id;
         delete polygons[i].parent;
      }

      return polygons;
   }

   toBufferGeometry() {
      return createBufferGeometry(this.toPolygons());
   }

   toMesh(material) {
      const geometry = this.toBufferGeometry(),
            mesh = new Mesh(geometry, material);

      if (this.matrix) {
         mesh.position.setFromMatrixPosition(this.matrix);
         mesh.rotation.setFromRotationMatrix(this.matrix);
      }

      return mesh;
   }

} // class Geometry

/** @summary create geometry to make cut on specified axis
  * @private */
function createNormal(axis_name, pos, size) {
   if (!size || (size < 10000)) size = 10000;

   let vertices;

   switch (axis_name) {
      case 'x':
         vertices = [new Vertex(pos, -3*size, size, 1, 0, 0),
                     new Vertex(pos, size, -3*size, 1, 0, 0),
                     new Vertex(pos, size, size, 1, 0, 0)];
         break;
      case 'y':
         vertices = [new Vertex(-3*size, pos, size, 0, 1, 0),
                     new Vertex(size, pos, size, 0, 1, 0),
                     new Vertex(size, pos, -3*size, 0, 1, 0)];
         break;
      // case 'z':
      default:
         vertices = [new Vertex(-3*size, size, pos, 0, 0, 1),
                     new Vertex(size, -3*size, pos, 0, 0, 1),
                     new Vertex(size, size, pos, 0, 0, 1)];
   }

   const node = new Node([new Polygon(vertices)]);

   return new Geometry(node);
}

export { createBufferGeometry, createNormal, Vertex, Geometry, Polygon };
