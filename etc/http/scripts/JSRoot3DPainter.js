/// @file JSRoot3DPainter.js
/// JavaScript ROOT 3D graphics

(function( factory ) {
   if ( typeof define === "function" && define.amd ) {
      // AMD. Register as an anonymous module.
      define( ['jquery', 'jquery-ui', 'd3', 'JSRootPainter', 'THREE', 'three.extra', 'THREE_ALL', 'JSRootPainter.jquery'], factory );
   } else {

      if (typeof JSROOT == 'undefined')
         throw new Error('JSROOT is not defined', 'JSRoot3DPainter.js');

      if (typeof d3 != 'object')
         throw new Error('This extension requires d3.v3.js', 'JSRoot3DPainter.js');

      if (typeof JSROOT.Painter != 'object')
         throw new Error('JSROOT.Painter is not defined', 'JSRoot3DPainter.js');

      if (typeof THREE == 'undefined')
         throw new Error('THREE is not defined', 'JSRoot3DPainter.js');

      factory(jQuery, jQuery.ui, d3, JSROOT);
   }
} (function($, myui, d3, JSROOT) {

   JSROOT.Painter.add3DInteraction = function() {
      // add 3D mouse interactive functions

      var painter = this;
      var mouseX, mouseY, mouseDowned = false;
      var mouse = {  x : 0, y : 0 }, INTERSECTED;

      var tooltip = function() {
         var id = 'tt';
         var top = 3;
         var left = 3;
         var maxw = 150;
         var speed = 10;
         var timer = 20;
         var endalpha = 95;
         var alpha = 0;
         var tt, t, c, b, h;
         var ie = document.all ? true : false;
         return {
            show : function(v, w) {
               if (tt == null) {
                  tt = document.createElement('div');
                  tt.setAttribute('id', id);
                  t = document.createElement('div');
                  t.setAttribute('id', id + 'top');
                  c = document.createElement('div');
                  c.setAttribute('id', id + 'cont');
                  b = document.createElement('div');
                  b.setAttribute('id', id + 'bot');
                  tt.appendChild(t);
                  tt.appendChild(c);
                  tt.appendChild(b);
                  document.body.appendChild(tt);
                  tt.style.opacity = 0;
                  tt.style.filter = 'alpha(opacity=0)';
                  document.onmousemove = this.pos;
               }
               tt.style.display = 'block';
               c.innerHTML = v;
               tt.style.width = w ? w + 'px' : 'auto';
               tt.style.width = 'auto'; // let it be automatically resizing...
               if (!w && ie) {
                  t.style.display = 'none';
                  b.style.display = 'none';
                  tt.style.width = tt.offsetWidth;
                  t.style.display = 'block';
                  b.style.display = 'block';
               }
               // if (tt.offsetWidth > maxw) { tt.style.width = maxw + 'px'; }
               h = parseInt(tt.offsetHeight) + top;
               clearInterval(tt.timer);
               tt.timer = setInterval(function() { tooltip.fade(1) }, timer);
            },
            pos : function(e) {
               var u = ie ? event.clientY + document.documentElement.scrollTop : e.pageY;
               var l = ie ? event.clientX + document.documentElement.scrollLeft : e.pageX;
               tt.style.top = u + 15 + 'px';// (u - h) + 'px';
               tt.style.left = (l + left) + 'px';
            },
            fade : function(d) {
               var a = alpha;
               if ((a != endalpha && d == 1) || (a != 0 && d == -1)) {
                  var i = speed;
                  if (endalpha - a < speed && d == 1) {
                     i = endalpha - a;
                  } else if (alpha < speed && d == -1) {
                     i = a;
                  }
                  alpha = a + (i * d);
                  tt.style.opacity = alpha * .01;
                  tt.style.filter = 'alpha(opacity=' + alpha + ')';
               } else {
                  clearInterval(tt.timer);
                  if (d == -1) {
                     tt.style.display = 'none';
                  }
               }
            },
            hide : function() {
               if (tt == null)
                  return;
               clearInterval(tt.timer);
               tt.timer = setInterval(function() {
                  tooltip.fade(-1)
               }, timer);
            }
         };
      }();

      var radius = 100;
      var theta = 0;
      var raycaster = new THREE.Raycaster();
      function findIntersection() {
         // find intersections
         if (mouseDowned) {
            if (INTERSECTED) {
               INTERSECTED.material.emissive.setHex(INTERSECTED.currentHex);
               painter.Render3D(0);
            }
            INTERSECTED = null;
            if (JSROOT.gStyle.Tooltip)
               tooltip.hide();
            return;
         }
         raycaster.setFromCamera( mouse, painter.camera );
         var intersects = raycaster.intersectObjects(painter.scene.children, true);
         if (intersects.length > 0) {
            var pick = null;
            for (var i = 0; i < intersects.length; ++i) {
               if ('emissive' in intersects[i].object.material) {
                  pick = intersects[i];
                  break;
               }
            }
            if (pick && INTERSECTED != pick.object) {
               if (INTERSECTED)
                  INTERSECTED.material.emissive.setHex(INTERSECTED.currentHex);
               INTERSECTED = pick.object;
               INTERSECTED.currentHex = INTERSECTED.material.emissive.getHex();
               INTERSECTED.material.emissive.setHex(0x5f5f5f);
               painter.Render3D(0);
               if (JSROOT.gStyle.Tooltip)
                  tooltip.show(INTERSECTED.name.length > 0 ? INTERSECTED.name
                        : INTERSECTED.parent.name, 200);
            }
         } else {
            if (INTERSECTED) {
               INTERSECTED.material.emissive.setHex(INTERSECTED.currentHex);
               painter.Render3D(0);
            }
            INTERSECTED = null;
            if (JSROOT.gStyle.Tooltip)
               tooltip.hide();
         }
      };

      $(painter.renderer.domElement).on('touchstart mousedown', function(e) {
         // var touch = e.changedTouches[0] || {};
         if (JSROOT.gStyle.Tooltip)
            tooltip.hide();
         e.preventDefault();
         var touch = e;
         if ('changedTouches' in e)
            touch = e.changedTouches[0];
         else if ('touches' in e)
            touch = e.touches[0];
         else if ('originalEvent' in e) {
            if ('changedTouches' in e.originalEvent)
               touch = e.originalEvent.changedTouches[0];
            else if ('touches' in e.originalEvent)
               touch = e.originalEvent.touches[0];
         }
         mouseX = touch.pageX;
         mouseY = touch.pageY;
         mouseDowned = true;
      });
      $(painter.renderer.domElement).on('touchmove mousemove',  function(e) {
         if (mouseDowned) {
            var touch = e;
            if ('changedTouches' in e)
               touch = e.changedTouches[0];
            else if ('touches' in e)
               touch = e.touches[0];
            else if ('originalEvent' in e) {
               if ('changedTouches' in e.originalEvent)
                  touch = e.originalEvent.changedTouches[0];
               else if ('touches' in e.originalEvent)
                  touch = e.originalEvent.touches[0];
            }
            var moveX = touch.pageX - mouseX;
            var moveY = touch.pageY - mouseY;
            // limited X rotate in -45 to 135 deg
            if ((moveY > 0 && painter.toplevel.rotation.x < Math.PI * 3 / 4)
                  || (moveY < 0 && painter.toplevel.rotation.x > -Math.PI / 4)) {
               painter.toplevel.rotation.x += moveY * 0.02;
            }
            painter.toplevel.rotation.y += moveX * 0.02;
            painter.Render3D(0);
            mouseX = touch.pageX;
            mouseY = touch.pageY;
         } else {
            e.preventDefault();
            var mouse_x = 'offsetX' in e.originalEvent ? e.originalEvent.offsetX : e.originalEvent.layerX;
            var mouse_y = 'offsetY' in e.originalEvent ? e.originalEvent.offsetY : e.originalEvent.layerY;
            mouse.x = (mouse_x / painter.renderer.domElement.width) * 2 - 1;
            mouse.y = -(mouse_y / painter.renderer.domElement.height) * 2 + 1;
            // enable picking once tootips are available...
            findIntersection();
         }
      });
      $(painter.renderer.domElement).on('touchend mouseup', function(e) {
         mouseDowned = false;
      });

      $(painter.renderer.domElement).on('mousewheel', function(e, d) {
         e.preventDefault();
         painter.camera.position.z += d * 20;
         painter.Render3D(0);
      });

      $(painter.renderer.domElement).on('contextmenu', function(e) {
         e.preventDefault();

         if (JSROOT.gStyle.Tooltip) tooltip.hide();

         return painter.ShowContextMenu("hist", e.originalEvent);
      });
   }

   JSROOT.Painter.HPainter_Create3DScene = function(arg) {

      if ((arg!=null) && (arg<0)) {
         this.clear_3d_canvas();
         delete this.size3d;
         delete this.scene;
         delete this.toplevel;
         delete this.camera;
         delete this.renderer;
         return;
      }

      if ('toplevel' in this) {
         // it is indication that all 3D object created, just replace it with empty

         var newtop = new THREE.Object3D();

         newtop.rotation.x = this.toplevel.rotation.x;
         newtop.rotation.y = this.toplevel.rotation.y;

         this.scene.remove(this.toplevel);

         this.scene.add(newtop);

         this.toplevel = newtop;
         return;
      }

      var size = this.size_for_3d();

      this.size3d = 100;

      // three.js 3D drawing
      this.scene = new THREE.Scene();
      //scene.fog = new THREE.Fog(0xffffff, 500, 3000);

      this.toplevel = new THREE.Object3D();
      this.toplevel.rotation.x = 30 * Math.PI / 180;
      this.toplevel.rotation.y = 30 * Math.PI / 180;
      this.scene.add(this.toplevel);

      this.camera = new THREE.PerspectiveCamera(45, size.width / size.height, 1, 4000);
      var pointLight = new THREE.PointLight(0xcfcfcf);
      this.camera.add( pointLight );
      pointLight.position.set( 10, 10, 10 );
      this.camera.position.set(0, this.size3d / 2, 500);
      this.scene.add( this.camera );

      /**
       * @author alteredq / http://alteredqualia.com/
       * @author mr.doob / http://mrdoob.com/
       */
      var Detector = {
            canvas : !!window.CanvasRenderingContext2D,
            webgl : (function() { try {
                  return !!window.WebGLRenderingContext && !!document.createElement('canvas').getContext('experimental-webgl');
               } catch (e) {
                  return false;
               }
            })(),
            workers : !!window.Worker,
            fileapi : window.File && window.FileReader && window.FileList && window.Blob
      };

      this.renderer = Detector.webgl ? new THREE.WebGLRenderer({ antialias : true, alpha: true }) :
                                       new THREE.CanvasRenderer({ antialias : true, alpha: true  });
      //renderer.setClearColor(0xffffff, 1);
      // renderer.setClearColor(0x0, 0);
      this.renderer.setSize(size.width, size.height);

      this.add_3d_canvas(this.renderer.domElement);

      this['CreateXYZ'] = JSROOT.Painter.HPainter_CreateXYZ;
      this['DrawXYZ'] = JSROOT.Painter.HPainter_DrawXYZ;
      this['Add3DInteraction'] = JSROOT.Painter.add3DInteraction;
      this['Render3D'] =JSROOT.Painter.Render3D;

      this.Add3DInteraction();
   }

   JSROOT.Painter.HPainter_CreateXYZ = function() {

      var xmin = this.xmin, xmax = this.xmax;
      //if (this.zoom_xmin != this.zoom_xmax) {
      //   xmin = this.zoom_xmin;
      //   xmax = this.zoom_xmax;
      //}
      var ymin = this.ymin, ymax = this.ymax;
      //if (this.zoom_ymin != this.zoom_ymax) {
      //   ymin = this.zoom_ymin;
      //   ymax = this.zoom_ymax;
      //}
      var zmin = this.gminbin, zmax = this.gmaxbin;
      if (('zmin' in this) && ('zmax' in this) && (this.Dimension()==3)) {
         zmin = this.zmin;
         zmax = this.zmax;
      } else {
         zmax = Math.ceil(zmax / 100) * 105; // not very nice
      }

      if (this.options.Logx) {
         if (xmin <= 0) xmin = 1e-6*xmax;
         this.tx = d3.scale.log().domain([ xmin, xmax ]).range([ -this.size3d, this.size3d ]);
         this.utx = d3.scale.log().domain([ -this.size3d, this.size3d ]).range([ xmin, xmax ]);
      } else {
         this.tx = d3.scale.linear().domain([ xmin, xmax ]).range([ -this.size3d, this.size3d ]);
         this.utx = d3.scale.linear().domain([ -this.size3d, this.size3d ]).range([ xmin, xmax ]);
      }
      if (this.options.Logy) {
         if (ymin <= 0) ymin = 1e-6*ymax;
         this.ty = d3.scale.log().domain([ ymin, ymax ]).range([ -this.size3d, this.size3d ]);
         this.uty = d3.scale.log().domain([ this.size3d, -this.size3d ]).range([ ymin, ymax ]);
      } else {
         this.ty = d3.scale.linear().domain([ ymin, ymax ]).range([ -this.size3d, this.size3d ]);
         this.uty = d3.scale.linear().domain([ this.size3d, -this.size3d ]).range([ ymin, ymax ]);
      }
      if (this.options.Logz) {
         if (zmin <= 0) zmin = 1e-6*zmax;
         this.tz = d3.scale.log().domain([ zmin, zmax]).range([ 0, this.size3d * 2 ]);
         this.utz = d3.scale.log().domain([ 0, this.size3d * 2 ]).range([ zmin, zmax ]);
      } else {
         this.tz = d3.scale.linear().domain([ zmin, zmax ]).range( [ 0, this.size3d * 2 ]);
         this.utz = d3.scale.linear().domain([ 0, this.size3d * 2 ]).range( [ zmin, zmax ]);
      }
   }

   JSROOT.Painter.HPainter_DrawXYZ = function() {
      // add the calibration vectors and texts

      var textMaterial = new THREE.MeshBasicMaterial({ color : 0x000000 });
      var lineMaterial = new THREE.LineBasicMaterial({ color : 0x000000 });

      var ticks = new Array();
      var imax, istep, len = 3, plen, sin45 = Math.sin(45);
      var text3d, text;
      var xmajors = this.tx.ticks(8);
      var xminors = this.tx.ticks(50);
      for (var i = -this.size3d, j = 0, k = 0; i < this.size3d; ++i) {
         var is_major = (this.utx(i) <= xmajors[j] && this.utx(i + 1) > xmajors[j]) ? true : false;
         var is_minor = (this.utx(i) <= xminors[k] && this.utx(i + 1) > xminors[k]) ? true : false;
         plen = (is_major ? len + 2 : len) * sin45;
         if (is_major) {
            text3d = new THREE.TextGeometry(xmajors[j], { size : 7, height : 0, curveSegments : 10 });
            ++j;

            text3d.computeBoundingBox();
            var centerOffset = 0.5 * (text3d.boundingBox.max.x - text3d.boundingBox.min.x);

            text = new THREE.Mesh(text3d, textMaterial);
            text.position.set(i - centerOffset, -13, this.size3d + plen);
            this.toplevel.add(text);

            text = new THREE.Mesh(text3d, textMaterial);
            text.position.set(i + centerOffset, -13, -this.size3d - plen);
            text.rotation.y = Math.PI;
            this.toplevel.add(text);
         }
         if (is_major || is_minor) {
            ++k;
            var geometry = new THREE.Geometry();
            geometry.vertices.push(new THREE.Vector3(i, 0, this.size3d));
            geometry.vertices.push(new THREE.Vector3(i, -plen, this.size3d + plen));
            this.toplevel.add(new THREE.Line(geometry, lineMaterial));
            ticks.push(geometry);
            geometry = new THREE.Geometry();
            geometry.vertices.push(new THREE.Vector3(i, 0, -this.size3d));
            geometry.vertices.push(new THREE.Vector3(i, -plen, -this.size3d - plen));
            this.toplevel.add(new THREE.Line(geometry, lineMaterial));
            ticks.push(geometry);
         }
      }
      var ymajors = this.ty.ticks(8);
      var yminors = this.ty.ticks(50);
      for (var i = this.size3d, j = 0, k = 0; i > -this.size3d; --i) {
         var is_major = (this.uty(i) <= ymajors[j] && this.uty(i - 1) > ymajors[j]) ? true : false;
         var is_minor = (this.uty(i) <= yminors[k] && this.uty(i - 1) > yminors[k]) ? true : false;
         plen = (is_major ? len + 2 : len) * sin45;
         if (is_major) {
            text3d = new THREE.TextGeometry(ymajors[j], { size : 7, height : 0, curveSegments : 10 });
            ++j;

            text3d.computeBoundingBox();
            var centerOffset = 0.5 * (text3d.boundingBox.max.x - text3d.boundingBox.min.x);

            text = new THREE.Mesh(text3d, textMaterial);
            text.position.set(this.size3d + plen, -13, i + centerOffset);
            text.rotation.y = Math.PI / 2;
            this.toplevel.add(text);

            text = new THREE.Mesh(text3d, textMaterial);
            text.position.set(-this.size3d - plen, -13, i - centerOffset);
            text.rotation.y = -Math.PI / 2;
            this.toplevel.add(text);
         }
         if (is_major || is_minor) {
            ++k;
            var geometry = new THREE.Geometry();
            geometry.vertices.push(new THREE.Vector3(this.size3d, 0, i));
            geometry.vertices.push(new THREE.Vector3(this.size3d + plen, -plen, i));
            this.toplevel.add(new THREE.Line(geometry, lineMaterial));
            ticks.push(geometry);
            geometry = new THREE.Geometry();
            geometry.vertices.push(new THREE.Vector3(-this.size3d, 0, i));
            geometry.vertices.push(new THREE.Vector3(-this.size3d - plen, -plen, i));
            this.toplevel.add(new THREE.Line(geometry, lineMaterial));
            ticks.push(geometry);
         }
      }
      var zmajors = this.tz.ticks(8);
      var zminors = this.tz.ticks(50);
      for (var i = 0, j = 0, k = 0; i < 2*this.size3d; ++i) {
         var is_major = (this.utz(i) <= zmajors[j] && this.utz(i + 1) > zmajors[j]) ? true : false;
         var is_minor = (this.utz(i) <= zminors[k] && this.utz(i + 1) > zminors[k]) ? true : false;
         plen = (is_major ? len + 2 : len) * sin45;
         if (is_major) {
            text3d = new THREE.TextGeometry(zmajors[j], { size : 7, height : 0, curveSegments : 10 });
            ++j;

            text3d.computeBoundingBox();
            var offset = 0.8 * (text3d.boundingBox.max.x - text3d.boundingBox.min.x);

            text = new THREE.Mesh(text3d, textMaterial);
            text.position.set(this.size3d + offset + 5, i - 2.5, this.size3d + offset + 5);
            text.rotation.y = Math.PI * 3 / 4;
            this.toplevel.add(text);

            text = new THREE.Mesh(text3d, textMaterial);
            text.position.set(this.size3d + offset + 5, i - 2.5, -this.size3d - offset - 5);
            text.rotation.y = -Math.PI * 3 / 4;
            this.toplevel.add(text);

            text = new THREE.Mesh(text3d, textMaterial);
            text.position.set(-this.size3d - offset - 5, i - 2.5, this.size3d + offset + 5);
            text.rotation.y = Math.PI / 4;
            this.toplevel.add(text);

            text = new THREE.Mesh(text3d, textMaterial);
            text.position.set(-this.size3d - offset - 5, i - 2.5, -this.size3d - offset - 5);
            text.rotation.y = -Math.PI / 4;
            this.toplevel.add(text);
         }
         if (is_major || is_minor) {
            ++k;
            var geometry = new THREE.Geometry();
            geometry.vertices.push(new THREE.Vector3(this.size3d, i, this.size3d));
            geometry.vertices.push(new THREE.Vector3(this.size3d + plen, i, this.size3d + plen));
            this.toplevel.add(new THREE.Line(geometry, lineMaterial));
            ticks.push(geometry);
            geometry = new THREE.Geometry();
            geometry.vertices.push(new THREE.Vector3(this.size3d, i, -this.size3d));
            geometry.vertices.push(new THREE.Vector3(this.size3d + plen, i, -this.size3d - plen));
            this.toplevel.add(new THREE.Line(geometry, lineMaterial));
            ticks.push(geometry);
            geometry = new THREE.Geometry();
            geometry.vertices.push(new THREE.Vector3(-this.size3d, i, this.size3d));
            geometry.vertices.push(new THREE.Vector3(-this.size3d - plen, i, this.size3d + plen));
            this.toplevel.add(new THREE.Line(geometry, lineMaterial));
            ticks.push(geometry);
            geometry = new THREE.Geometry();
            geometry.vertices.push(new THREE.Vector3(-this.size3d, i, -this.size3d));
            geometry.vertices.push(new THREE.Vector3(-this.size3d - plen, i, -this.size3d - plen));
            this.toplevel.add(new THREE.Line(geometry, lineMaterial));
            ticks.push(geometry);
         }
      }

      for (var t=0; t < ticks.length; ++t)
         ticks[t].dispose();

      var wireMaterial = new THREE.MeshBasicMaterial({
         color : 0x000000,
         wireframe : true,
         wireframeLinewidth : 0.5,
         side : THREE.DoubleSide
      });

      // create a new mesh with cube geometry
      var cube = new THREE.Mesh(new THREE.BoxGeometry(this.size3d * 2, this.size3d * 2, this.size3d * 2), wireMaterial);
      //cube.position.y = size;

      var helper = new THREE.BoxHelper(cube);
      helper.material.color.set(0x000000);

      var box = new THREE.Object3D();
      box.add(helper);
      box.position.y = this.size3d;

      // add the cube to the scene
      this.toplevel.add(box);

      this.camera.lookat = cube;
   }

   JSROOT.Painter.TH2Painter_Draw3DBins = function() {
      var constx = (this.size3d * 2 / this.nbinsx) / this.gmaxbin;
      var consty = (this.size3d * 2 / this.nbinsy) / this.gmaxbin;

      var fcolor = d3.rgb(JSROOT.Painter.root_colors[this.histo['fFillColor']]);

      var local_bins = this.CreateDrawBins(100, 100, 2, (JSROOT.gStyle.Tooltip ? 1 : 0));

      // create the bin cubes
      var fillcolor = new THREE.Color(0xDDDDDD);
      fillcolor.setRGB(fcolor.r / 255, fcolor.g / 255, fcolor.b / 255);

      for (var i = 0; i < local_bins.length; ++i) {
         var hh = local_bins[i];
         var wei = this.tz(hh.z);

         // create a new mesh with cube geometry
         var bin = new THREE.Mesh(new THREE.BoxGeometry(2 * this.size3d / this.nbinsx, wei, 2 * this.size3d / this.nbinsy),
                                  new THREE.MeshLambertMaterial({ color : fillcolor.getHex() }));

         bin.position.x = this.tx(hh.x);
         bin.position.y = wei / 2;
         bin.position.z = -(this.ty(hh.y));

         if (JSROOT.gStyle.Tooltip && ('tip' in hh))
            bin.name = hh.tip.replace(/(?:\r\n|\r|\n)/g, '<br/>');
         this.toplevel.add(bin);

         var helper = new THREE.BoxHelper(bin);
         helper.material.color.set(0x000000);
         helper.material.linewidth = 1.0;
         this.toplevel.add(helper);
      }

      delete local_bins;
      local_bins = null;
   }

   JSROOT.Painter.Render3D = function(tmout) {
      if (tmout === undefined) tmout = 5; // by default, rendering happens with timeout

      if (tmout <= 0) {
         if ('render_tmout' in this)
            clearTimeout(this['render_tmout']);

         // do rendering, most consuming time
         this.renderer.render(this.scene, this.camera);

         delete this['render_tmout'];

         return;
      }

      // no need to shoot rendering once again
      if ('render_tmout' in this) return;

      this['render_tmout'] = setTimeout(this.Render3D.bind(this,0), tmout);
   }

   JSROOT.Painter.TH2Painter_Draw3D = function(call_back) {

      // function called with this as painter

      this.Create3DScene();

      this.CreateXYZ();

      this.DrawXYZ();

      this.Draw3DBins();

      this.Render3D();

      JSROOT.CallBack(call_back);
   }

   // ==============================================================================


   JSROOT.TH3Painter = function(histo) {
      JSROOT.THistPainter.call(this, histo);

      this['Create3DScene'] = JSROOT.Painter.HPainter_Create3DScene;
   }

   JSROOT.TH3Painter.prototype = Object.create(JSROOT.THistPainter.prototype);

   JSROOT.TH3Painter.prototype.ScanContent = function() {
      this.nbinsx = this.histo['fXaxis']['fNbins'];
      this.nbinsy = this.histo['fYaxis']['fNbins'];
      this.nbinsz = this.histo['fZaxis']['fNbins'];

      this.xmin = this.histo['fXaxis']['fXmin'];
      this.xmax = this.histo['fXaxis']['fXmax'];

      this.ymin = this.histo['fYaxis']['fXmin'];
      this.ymax = this.histo['fYaxis']['fXmax'];

      this.zmin = this.histo['fZaxis']['fXmin'];
      this.zmax = this.histo['fZaxis']['fXmax'];

      // global min/max, used at the moment in 3D drawing

      this.gminbin = this.gmaxbin = this.histo.getBinContent(1, 1, 1);
      for (var i = 0; i < this.nbinsx; ++i)
         for (var j = 0; j < this.nbinsy; ++j)
            for (var k = 0; k < this.nbinsz; ++k) {
               var bin_content = this.histo.getBinContent(i + 1, j + 1, k + 1);
               if (bin_content < this.gminbin) this.gminbin = bin_content; else
               if (bin_content > this.gmaxbin) this.gmaxbin = bin_content;
            }

      this.draw_content = this.gmaxbin > 0;
   }

   JSROOT.TH3Painter.prototype.CountStat = function() {
      var stat_sum0 = 0, stat_sumx1 = 0, stat_sumy1 = 0, stat_sumz1 = 0, stat_sumx2 = 0, stat_sumy2 = 0, stat_sumz2 = 0;

      var res = { entries: 0, integral: 0, meanx: 0, meany: 0, meanz: 0, rmsx: 0, rmsy: 0, rmsz: 0 };

      for (var xi = 0; xi < this.nbinsx+2; ++xi) {

         var xx = this.xmin + (xi - 0.5) / this.nbinsx * (this.xmax - this.xmin);
         var xside = (xi === 0) ? 0 : (xi === this.nbinsx+1 ? 2 : 1);

         for (var yi = 0; yi < this.nbinsy+2; ++yi) {

            var yy = this.ymin + (yi - 0.5) / this.nbinsy * (this.ymax - this.ymin);
            var yside = (yi === 0) ? 0 : (yi === this.nbinsy+1 ? 2 : 1);

            for (var zi = 0; zi < this.nbinsz+2; ++zi) {

               var zz = this.zmin + (zi - 0.5) / this.nbinsz * (this.zmax - this.zmin);
               var zside = (zi === 0) ? 0 : (zi === this.nbinsz+1 ? 2 : 1);

               var cont = this.histo.getBinContent(xi, yi, zi);
               res.entries += cont;

               if ((xside==1) && (yside==1) && (zside==1)) {
                  stat_sum0 += cont;
                  stat_sumx1 += xx * cont;
                  stat_sumy1 += yy * cont;
                  stat_sumz1 += zz * cont;
                  stat_sumx2 += xx * xx * cont;
                  stat_sumy2 += yy * yy * cont;
                  stat_sumz2 += zz * zz * cont;
               }
            }
         }
      }

      if (this.histo.fTsumw>0) {
         stat_sum0 = this.histo.fTsumw;
         stat_sumx1 = this.histo.fTsumwx;
         stat_sumx2 = this.histo.fTsumwx2;
         stat_sumy1 = this.histo.fTsumwy;
         stat_sumy2 = this.histo.fTsumwy2;
         stat_sumz1 = this.histo.fTsumwz;
         stat_sumz2 = this.histo.fTsumwz2;
      }

      if (stat_sum0 > 0) {
         res.meanx = stat_sumx1 / stat_sum0;
         res.meany = stat_sumy1 / stat_sum0;
         res.meanz = stat_sumz1 / stat_sum0;
         res.rmsx = Math.sqrt(stat_sumx2 / stat_sum0 - res.meanx * res.meanx);
         res.rmsy = Math.sqrt(stat_sumy2 / stat_sum0 - res.meany * res.meany);
         res.rmsz = Math.sqrt(stat_sumz2 / stat_sum0 - res.meanz * res.meanz);
      }

      res.integral = stat_sum0;

      if (this.histo.fEntries > 1) res.entries = this.histo.fEntries;

      return res;
   }

   JSROOT.TH3Painter.prototype.FillStatistic = function(stat, dostat, dofit) {
      if (!this.histo) return false;

      var data = this.CountStat();

      var print_name = dostat % 10;
      var print_entries = Math.floor(dostat / 10) % 10;
      var print_mean = Math.floor(dostat / 100) % 10;
      var print_rms = Math.floor(dostat / 1000) % 10;
      var print_under = Math.floor(dostat / 10000) % 10;
      var print_over = Math.floor(dostat / 100000) % 10;
      var print_integral = Math.floor(dostat / 1000000) % 10;
      //var print_skew = Math.floor(dostat / 10000000) % 10;
      //var print_kurt = Math.floor(dostat / 100000000) % 10;

      if (print_name > 0)
         stat.AddLine(this.histo['fName']);

      if (print_entries > 0)
         stat.AddLine("Entries = " + stat.Format(data.entries,"entries"));

      if (print_mean > 0) {
         stat.AddLine("Mean x = " + stat.Format(data.meanx));
         stat.AddLine("Mean y = " + stat.Format(data.meany));
         stat.AddLine("Mean z = " + stat.Format(data.meanz));
      }

      if (print_rms > 0) {
         stat.AddLine("Std Dev x = " + stat.Format(data.rmsx));
         stat.AddLine("Std Dev y = " + stat.Format(data.rmsy));
         stat.AddLine("Std Dev z = " + stat.Format(data.rmsz));
      }

      if (print_integral > 0) {
         stat.AddLine("Integral = " + stat.Format(data.integral,"entries"));
      }

      // adjust the size of the stats box with the number of lines
      var nlines = stat.pavetext['fLines'].arr.length;
      var stath = nlines * JSROOT.gStyle.StatFontSize;
      if (stath <= 0 || 3 == (JSROOT.gStyle.StatFont % 10)) {
         stath = 0.25 * nlines * JSROOT.gStyle.StatH;
         stat.pavetext['fY1NDC'] = 0.93 - stath;
         stat.pavetext['fY2NDC'] = 0.93;
      }

      return true;
   }

   JSROOT.TH3Painter.prototype.CreateBins = function() {
      var bins = [];

      var name = this.GetItemName();
      if ((name==null) || (name=="")) name = this.histo.fName;
      if (name.length > 0) name += "<br/>";

      for (var i = 0; i < this.nbinsx; ++i)
         for (var j = 0; j < this.nbinsy; ++j)
            for (var k = 0; k < this.nbinsz; ++k) {
               var bin_content = this.histo.getBinContent(i + 1, j + 1, k + 1);
               if (bin_content <= this.gminbin) continue;

               var bin = {
                     x : this.xmin + (i + 0.5) / this.nbinsx * (this.xmax - this.xmin),
                     y : this.ymin + (j + 0.5) / this.nbinsy * (this.ymax - this.ymin),
                     z : this.zmin + (k + 0.5) / this.nbinsz * (this.zmax - this.zmin),
                     n : bin_content
                  };

               if (JSROOT.gStyle.Tooltip)
                  bin.tip = name + 'x=' + JSROOT.FFormat(bin.x,"6.4g") + ' bin=' + (i+1) + '<br/>'
                                 + 'y=' + JSROOT.FFormat(bin.y,"6.4g") + ' bin=' + (j+1) + '<br/>'
                                 + 'z=' + JSROOT.FFormat(bin.z,"6.4g") + ' bin=' + (k+1) + '<br/>'
                                 + 'entries=' + JSROOT.FFormat(bin.n, "7.0g");

               bins.push(bin);
            }

      return bins;
   }

   JSROOT.TH3Painter.prototype.Draw3DBins = function() {
      if (!this.draw_content) return;

      var bins = this.CreateBins();

      // create the bin cubes
      var constx = (this.size3d * 2 / this.nbinsx) / this.gmaxbin;
      var consty = (this.size3d * 2 / this.nbinsy) / this.gmaxbin;
      var constz = (this.size3d * 2 / this.nbinsz) / this.gmaxbin;

      var fcolor = d3.rgb(JSROOT.Painter.root_colors[this.histo['fFillColor']]);
      var fillcolor = new THREE.Color(0xDDDDDD);
      fillcolor.setRGB(fcolor.r / 255, fcolor.g / 255,  fcolor.b / 255);
      var bin, wei;
      for (var i = 0; i < bins.length; ++i) {
         wei = (this.options.Color > 0 ? this.gmaxbin : bins[i].n);
         if (this.options.Box == 11) {
            bin = new THREE.Mesh(new THREE.SphereGeometry(0.5 * wei * constx),
                                 new THREE.MeshPhongMaterial({ color : fillcolor.getHex(), specular : 0x4f4f4f }));
         } else {
            bin = new THREE.Mesh(new THREE.BoxGeometry(wei * constx, wei * constz, wei * consty),
                                 new THREE.MeshLambertMaterial({ color : fillcolor.getHex() }));
         }
         bin.position.x = this.tx(bins[i].x);
         bin.position.y = this.tz(bins[i].z);
         bin.position.z = -(this.ty(bins[i].y));
         if ('tip' in bins[i])
           bin.name = bins[i].tip;

         this.toplevel.add(bin);

         if (this.options.Box != 11) {
            var helper = new THREE.BoxHelper(bin);
            helper.material.color.set(0x000000);
            helper.material.linewidth = 1.0;
            this.toplevel.add(helper);
         }
      }
   }

   JSROOT.TH3Painter.prototype.Redraw = function() {
      this.Create3DScene();
      this.CreateXYZ();
      this.DrawXYZ();
      this.Draw3DBins();
      this.Render3D();
   }

   JSROOT.Painter.drawHistogram3D = function(divid, histo, opt) {
      // when called, *this* set to painter instance

      // create painter and add it to canvas
      JSROOT.extend(this, new JSROOT.TH3Painter(histo));

      this.SetDivId(divid, 4);

      this.options = this.DecodeOptions(opt);

      this.CheckPadOptions();

      this.ScanContent();

      this.Redraw();

      if (this.create_canvas) this.DrawTitle();

      if (JSROOT.gStyle.AutoStat && this.create_canvas) {
         var stats = this.CreateStat();
         if (stats) JSROOT.draw(this.divid, stats, "");
      }

      return this.DrawingReady();
   }

   JSROOT.Painter.drawPolyMarker3D = function(divid, poly, opt) {
      // when called, *this* set to painter instance

      this.SetDivId(divid);

      var main = this.main_painter();

      if ((main == null) || !('renderer' in main)) return this.DrawingReady();

      var cnt = poly.fP.length;
      var step = 3;

      if ((JSROOT.gStyle.OptimizeDraw > 0) && (cnt > 300*3)) {
         step = Math.floor(cnt / 300 / 3) * 3;
         if (step <= 6) step = 6;
      }

      var fcolor = d3.rgb(JSROOT.Painter.root_colors[poly.fMarkerColor]);
      var fillcolor = new THREE.Color(0xDDDDDD);
      fillcolor.setRGB(fcolor.r / 255, fcolor.g / 255,  fcolor.b / 255);

      for (var n=0; n<cnt; n+=step) {
         var bin = new THREE.Mesh(new THREE.SphereGeometry(1),
                                  new THREE.MeshPhongMaterial({ color : fillcolor.getHex(), specular : 0x4f4f4f}));
         bin.position.x = main.tx(poly.fP[n]);
         bin.position.y = main.tz(poly.fP[n+2]);
         bin.position.z = -main.ty(poly.fP[n+1]);
         main.toplevel.add(bin);
      }

      main.Render3D();

      return this.DrawingReady();
   }

   return JSROOT.Painter;

}));

