/// @file JSRoot3DPainter.js
/// JavaScript ROOT 3D graphics

(function( factory ) {
   if ( typeof define === "function" && define.amd ) {
      // AMD. Register as an anonymous module.
      define( ['jquery','jquery-ui', 'd3', 'JSRootPainter', 'THREE', 'jquery.mousewheel'], factory );
   } else {

      if (typeof JSROOT == 'undefined') {
         var e1 = new Error('JSROOT is not defined');
         e1.source = 'JSRoot3DPainter.js';
         throw e1;
      }

      if (typeof d3 != 'object') {
         var e1 = new Error('This extension requires d3.v3.js');
         e1.source = 'JSRoot3DPainter.js';
         throw e1;
      }

      if (typeof JSROOT.Painter != 'object') {
         var e1 = new Error('JSROOT.Painter is not defined');
         e1.source = 'JSRoot3DPainter.js';
         throw e1;
      }

      if (typeof THREE == 'undefined') {
         var e1 = new Error('THREE is not defined');
         e1.source = 'JSRoot3DPainter.js';
         throw e1;
      }

      factory(jQuery, jQuery.ui, d3, JSROOT);
   }
} (function($, myui, d3, JSROOT) {

   JSROOT.Painter.add3DInteraction = function(renderer, scene, camera, toplevel, painter) {
      // add 3D mouse interactive functions
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
      var projector = new THREE.Projector();
      function findIntersection() {
         // find intersections
         if (mouseDowned) {
            if (INTERSECTED) {
               INTERSECTED.material.emissive.setHex(INTERSECTED.currentHex);
               renderer.render(scene, camera);
            }
            INTERSECTED = null;
            if (JSROOT.gStyle.Tooltip)
               tooltip.hide();
            return;
         }
         var vector = new THREE.Vector3(mouse.x, mouse.y, 1);
         projector.unprojectVector(vector, camera);
         var raycaster = new THREE.Raycaster(camera.position, vector.sub(
               camera.position).normalize());
         var intersects = raycaster.intersectObjects(scene.children, true);
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
               renderer.render(scene, camera);
               if (JSROOT.gStyle.Tooltip)
                  tooltip.show(INTERSECTED.name.length > 0 ? INTERSECTED.name
                        : INTERSECTED.parent.name, 200);
            }
         } else {
            if (INTERSECTED) {
               INTERSECTED.material.emissive.setHex(INTERSECTED.currentHex);
               renderer.render(scene, camera);
            }
            INTERSECTED = null;
            if (JSROOT.gStyle.Tooltip)
               tooltip.hide();
         }
      }
      ;

      $(renderer.domElement).on('touchstart mousedown', function(e) {
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
      $(renderer.domElement).on('touchmove mousemove',  function(e) {
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
            if ((moveY > 0 && toplevel.rotation.x < Math.PI * 3 / 4)
                  || (moveY < 0 && toplevel.rotation.x > -Math.PI / 4)) {
               toplevel.rotation.x += moveY * 0.02;
            }
            toplevel.rotation.y += moveX * 0.02;
            renderer.render(scene, camera);
            mouseX = touch.pageX;
            mouseY = touch.pageY;
         } else {
            e.preventDefault();
            var mouse_x = 'offsetX' in e.originalEvent ? e.originalEvent.offsetX : e.originalEvent.layerX;
            var mouse_y = 'offsetY' in e.originalEvent ? e.originalEvent.offsetY : e.originalEvent.layerY;
            mouse.x = (mouse_x / renderer.domElement.width) * 2 - 1;
            mouse.y = -(mouse_y / renderer.domElement.height) * 2 + 1;
            // enable picking once tootips are available...
            findIntersection();
         }
      });
      $(renderer.domElement).on('touchend mouseup', function(e) {
         mouseDowned = false;
      });

      $(renderer.domElement).on('mousewheel', function(e, d) {
         e.preventDefault();
         camera.position.z += d * 20;
         renderer.render(scene, camera);
      });

      $(renderer.domElement).on('contextmenu', function(e) {
         e.preventDefault();

         if (JSROOT.gStyle.Tooltip) tooltip.hide();

         JSROOT.Painter.createMenu(function(menu) {
            if (painter)
               menu.add("header:"+ painter.histo['fName']);

            menu.add(JSROOT.gStyle.Tooltip ? "Disable tooltip" : "Enable tooltip", function() {
               JSROOT.gStyle.Tooltip = !JSROOT.gStyle.Tooltip;
               tooltip.hide();
            });

            if (painter)
               menu.add("Switch to 2D", function() {
                  $(painter.svg_pad().node()).show().parent().find(renderer.domElement).remove();
                  tooltip.hide();
                  painter.Draw2D();
               });
            menu.add("Close");

            menu.show(e.originalEvent);
         });

      });
   }

   JSROOT.Painter.real_drawHistogram2D = function(painter) {

      var w = painter.pad_width(), h = painter.pad_height(), size = 100;

      var xmin = painter.xmin, xmax = painter.xmax;
      if (painter.zoom_xmin != painter.zoom_xmax) {
         xmin = painter.zoom_xmin;
         xmax = painter.zoom_xmax;
      }
      var ymin = painter.ymin, ymax = painter.ymax;
      if (painter.zoom_ymin != painter.zoom_ymax) {
         ymin = painter.zoom_ymin;
         ymax = painter.zoom_ymax;
      }

      var tx, utx, ty, uty, tz, utz;

      if (painter.options.Logx) {
         tx = d3.scale.log().domain([ xmin, xmax ]).range([ -size, size ]);
         utx = d3.scale.log().domain([ -size, size ]).range([ xmin, xmax ]);
      } else {
         tx = d3.scale.linear().domain([ xmin, xmax ]).range([ -size, size ]);
         utx = d3.scale.linear().domain([ -size, size ]).range([ xmin, xmax ]);
      }
      if (painter.options.Logy) {
         ty = d3.scale.log().domain([ ymin, ymax ]).range([ -size, size ]);
         uty = d3.scale.log().domain([ size, -size ]).range([ ymin, ymax ]);
      } else {
         ty = d3.scale.linear().domain([ ymin, ymax ]).range([ -size, size ]);
         uty = d3.scale.linear().domain([ size, -size ]).range([ ymin, ymax ]);
      }
      if (painter.options.Logz) {
         tz = d3.scale.log().domain([ painter.gminbin, Math.ceil(painter.gmaxbin / 100) * 105 ]).range([ 0, size * 2 ]);
         utz = d3.scale.log().domain([ 0, size * 2 ]).range([ painter.gminbin, Math.ceil(painter.gmaxbin / 100) * 105 ]);
      } else {
         tz = d3.scale.linear().domain([ painter.gminbin, Math.ceil(painter.gmaxbin / 100) * 105 ]).range( [ 0, size * 2 ]);
         utz = d3.scale.linear().domain([ 0, size * 2 ]).range( [ painter.gminbin, Math.ceil(painter.gmaxbin / 100) * 105 ]);
      }

      var constx = (size * 2 / painter.nbinsx) / painter.gmaxbin;
      var consty = (size * 2 / painter.nbinsy) / painter.gmaxbin;

      var colorFlag = (painter.options.Color > 0);
      var fcolor = d3.rgb(JSROOT.Painter.root_colors[painter.histo['fFillColor']]);

      var local_bins = painter.CreateDrawBins(100, 100, 2, (JSROOT.gStyle.Tooltip ? 1 : 0));

      // three.js 3D drawing
      var scene = new THREE.Scene();

      var toplevel = new THREE.Object3D();
      toplevel.rotation.x = 30 * Math.PI / 180;
      toplevel.rotation.y = 30 * Math.PI / 180;
      scene.add(toplevel);

      var wireMaterial = new THREE.MeshBasicMaterial({
         color : 0x000000,
         wireframe : true,
         wireframeLinewidth : 0.5,
         side : THREE.DoubleSide
      });

      // create a new mesh with cube geometry
      var cube = new THREE.Mesh(new THREE.BoxGeometry(size * 2, size * 2, size * 2), wireMaterial);
      //cube.position.y = size;

      var helper = new THREE.BoxHelper(cube);
      helper.material.color.set(0x000000);

      var box = new THREE.Object3D();
      box.add(helper);
      box.position.y = size;

      // add the cube to the scene
      toplevel.add(box);

      var textMaterial = new THREE.MeshBasicMaterial({ color : 0x000000 });

      // add the calibration vectors and texts
      var geometry = new THREE.Geometry();
      var imax, istep, len = 3, plen, sin45 = Math.sin(45);
      var text3d, text;
      var xmajors = tx.ticks(8);
      var xminors = tx.ticks(50);
      for (var i = -size, j = 0, k = 0; i < size; ++i) {
         var is_major = (utx(i) <= xmajors[j] && utx(i + 1) > xmajors[j]) ? true : false;
         var is_minor = (utx(i) <= xminors[k] && utx(i + 1) > xminors[k]) ? true : false;
         plen = (is_major ? len + 2 : len) * sin45;
         if (is_major) {
            text3d = new THREE.TextGeometry(xmajors[j], { size : 7, height : 0, curveSegments : 10 });
            ++j;

            text3d.computeBoundingBox();
            var centerOffset = 0.5 * (text3d.boundingBox.max.x - text3d.boundingBox.min.x);

            text = new THREE.Mesh(text3d, textMaterial);
            text.position.set(i - centerOffset, -13, size + plen);
            toplevel.add(text);

            text = new THREE.Mesh(text3d, textMaterial);
            text.position.set(i + centerOffset, -13, -size - plen);
            text.rotation.y = Math.PI;
            toplevel.add(text);
         }
         if (is_major || is_minor) {
            ++k;
            geometry.vertices.push(new THREE.Vector3(i, 0, size));
            geometry.vertices.push(new THREE.Vector3(i, -plen, size + plen));
            geometry.vertices.push(new THREE.Vector3(i, 0, -size));
            geometry.vertices.push(new THREE.Vector3(i, -plen, -size - plen));
         }
      }
      var ymajors = ty.ticks(8);
      var yminors = ty.ticks(50);
      for (var i = size, j = 0, k = 0; i > -size; --i) {
         var is_major = (uty(i) <= ymajors[j] && uty(i - 1) > ymajors[j]) ? true : false;
         var is_minor = (uty(i) <= yminors[k] && uty(i - 1) > yminors[k]) ? true : false;
         plen = (is_major ? len + 2 : len) * sin45;
         if (is_major) {
            text3d = new THREE.TextGeometry(ymajors[j], { size : 7, height : 0, curveSegments : 10 });
            ++j;

            text3d.computeBoundingBox();
            var centerOffset = 0.5 * (text3d.boundingBox.max.x - text3d.boundingBox.min.x);

            text = new THREE.Mesh(text3d, textMaterial);
            text.position.set(size + plen, -13, i + centerOffset);
            text.rotation.y = Math.PI / 2;
            toplevel.add(text);

            text = new THREE.Mesh(text3d, textMaterial);
            text.position.set(-size - plen, -13, i - centerOffset);
            text.rotation.y = -Math.PI / 2;
            toplevel.add(text);
         }
         if (is_major || is_minor) {
            ++k;
            geometry.vertices.push(new THREE.Vector3(size, 0, i));
            geometry.vertices.push(new THREE.Vector3(size + plen, -plen, i));
            geometry.vertices.push(new THREE.Vector3(-size, 0, i));
            geometry.vertices.push(new THREE.Vector3(-size - plen, -plen, i));
         }
      }
      var zmajors = tz.ticks(8);
      var zminors = tz.ticks(50);
      for (var i = 0, j = 0, k = 0; i < (size * 2); ++i) {
         var is_major = (utz(i) <= zmajors[j] && utz(i + 1) > zmajors[j]) ? true : false;
         var is_minor = (utz(i) <= zminors[k] && utz(i + 1) > zminors[k]) ? true : false;
         plen = (is_major ? len + 2 : len) * sin45;
         if (is_major) {
            text3d = new THREE.TextGeometry(zmajors[j], { size : 7, height : 0, curveSegments : 10 });
            ++j;

            text3d.computeBoundingBox();
            var offset = 0.8 * (text3d.boundingBox.max.x - text3d.boundingBox.min.x);

            text = new THREE.Mesh(text3d, textMaterial);
            text.position.set(size + offset + 5, i - 2.5, size + offset + 5);
            text.rotation.y = Math.PI * 3 / 4;
            toplevel.add(text);

            text = new THREE.Mesh(text3d, textMaterial);
            text.position.set(size + offset + 5, i - 2.5, -size - offset - 5);
            text.rotation.y = -Math.PI * 3 / 4;
            toplevel.add(text);

            text = new THREE.Mesh(text3d, textMaterial);
            text.position.set(-size - offset - 5, i - 2.5, size + offset + 5);
            text.rotation.y = Math.PI / 4;
            toplevel.add(text);

            text = new THREE.Mesh(text3d, textMaterial);
            text.position.set(-size - offset - 5, i - 2.5, -size - offset - 5);
            text.rotation.y = -Math.PI / 4;
            toplevel.add(text);
         }
         if (is_major || is_minor) {
            ++k;
            geometry.vertices.push(new THREE.Vector3(size, i, size));
            geometry.vertices.push(new THREE.Vector3(size + plen, i, size + plen));
            geometry.vertices.push(new THREE.Vector3(size, i, -size));
            geometry.vertices.push(new THREE.Vector3(size + plen, i, -size - plen));
            geometry.vertices.push(new THREE.Vector3(-size, i, size));
            geometry.vertices.push(new THREE.Vector3(-size - plen, i, size + plen));
            geometry.vertices.push(new THREE.Vector3(-size, i, -size));
            geometry.vertices.push(new THREE.Vector3(-size - plen, i, -size - plen));
         }
      }

      // add the calibration lines
      var lineMaterial = new THREE.LineBasicMaterial({ color : 0x000000 });
      var line = new THREE.Line(geometry, lineMaterial);
      line.type = THREE.LinePieces;
      toplevel.add(line);

      // create the bin cubes

      var fillcolor = new THREE.Color(0xDDDDDD);
      fillcolor.setRGB(fcolor.r / 255, fcolor.g / 255, fcolor.b / 255);
      var bin, wei, hh;

      for (var i = 0; i < local_bins.length; ++i) {
         hh = local_bins[i];
         wei = tz(hh.z);

         bin = THREE.SceneUtils.createMultiMaterialObject(
               new THREE.BoxGeometry(2 * size / painter.nbinsx, wei, 2 * size / painter.nbinsy),
               [ new THREE.MeshLambertMaterial({ color : fillcolor.getHex(), shading : THREE.NoShading }), wireMaterial ]);
         bin.position.x = tx(hh.x);
         bin.position.y = wei / 2;
         bin.position.z = -(ty(hh.y));

         if (JSROOT.gStyle.Tooltip)
            bin.name = hh.tip;
         toplevel.add(bin);
      }

      delete local_bins;
      local_bins = null;

      // create a point light
      var pointLight = new THREE.PointLight(0xcfcfcf);
      pointLight.position.set(0, 50, 250);
      scene.add(pointLight);

      // var directionalLight = new THREE.DirectionalLight(
            // 0x7f7f7f );
      // directionalLight.position.set( 0, -70, 100
      // ).normalize();
      // scene.add( directionalLight );

      var camera = new THREE.PerspectiveCamera(45, w / h, 1, 1000);
      camera.position.set(0, size / 2, 500);
      camera.lookat = cube;

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

      var renderer = Detector.webgl ? new THREE.WebGLRenderer({ antialias : true }) :
                                      new THREE.CanvasRenderer({ antialias : true });
      renderer.setClearColor(0xffffff, 1);
      renderer.setSize(w, h);
      $(painter.svg_pad().node()).hide().parent().append(renderer.domElement);
      renderer.render(scene, camera);

      JSROOT.Painter.add3DInteraction(renderer, scene, camera, toplevel, painter);
   }

   JSROOT.Painter.drawHistogram3D = function(divid, histo, opt, painter) {

      var logx = false, logy = false, logz = false, gridx = false, gridy = false, gridz = false;

      painter.SetDivId(divid, -1);
      var pad = painter.root_pad();

      var render_to;
      if (!painter.svg_pad().empty())
         render_to = $(painter.svg_pad().node()).hide().parent();
      else
         render_to = $("#" + divid);

      var opt = histo['fOption'].toLowerCase();
      // if (opt=="") opt = "colz";

      if (pad) {
         logx = pad['fLogx'];
         logy = pad['fLogy'];
         logz = pad['fLogz'];
         gridx = pad['fGridx'];
         gridy = pad['fGridy'];
         gridz = pad['fGridz'];
      }

      var fillcolor = JSROOT.Painter.root_colors[histo['fFillColor']];
      var linecolor = JSROOT.Painter.root_colors[histo['fLineColor']];
      if (histo['fFillColor'] == 0) {
         fillcolor = '#4572A7';
      }
      if (histo['fLineColor'] == 0) {
         linecolor = '#4572A7';
      }
      var nbinsx = histo['fXaxis']['fNbins'];
      var nbinsy = histo['fYaxis']['fNbins'];
      var nbinsz = histo['fZaxis']['fNbins'];
      var scalex = (histo['fXaxis']['fXmax'] - histo['fXaxis']['fXmin']) / histo['fXaxis']['fNbins'];
      var scaley = (histo['fYaxis']['fXmax'] - histo['fYaxis']['fXmin']) / histo['fYaxis']['fNbins'];
      var scalez = (histo['fZaxis']['fXmax'] - histo['fZaxis']['fXmin']) / histo['fZaxis']['fNbins'];
      var maxbin = -1e32, minbin = 1e32;
      maxbin = d3.max(histo['fArray']);
      minbin = d3.min(histo['fArray']);
      var bins = new Array();
      for (var i = 0; i <= nbinsx + 2; ++i) {
         for (var j = 0; j < nbinsy + 2; ++j) {
            for (var k = 0; k < nbinsz + 2; ++k) {
               var bin_content = histo.getBinContent(i, j, k);
               if (bin_content > minbin) {
                  var point = {
                        x : histo['fXaxis']['fXmin'] + (i * scalex),
                        y : histo['fYaxis']['fXmin'] + (j * scaley),
                        z : histo['fZaxis']['fXmin'] + (k * scalez),
                        n : bin_content
                  };
                  bins.push(point);
               }
            }
         }
      }
      var w = render_to.width(), h = render_to.height(), size = 100;
      if (h<10) { render_to.height(0.66*w); h = render_to.height(); }

      if (logx) {
         var tx = d3.scale.log().domain([ histo['fXaxis']['fXmin'],  histo['fXaxis']['fXmax'] ]).range( [ -size, size ]);
         var utx = d3.scale.log().domain([ -size, size ]).range([ histo['fXaxis']['fXmin'], histo['fXaxis']['fXmax'] ]);
      } else {
         var tx = d3.scale.linear().domain( [ histo['fXaxis']['fXmin'], histo['fXaxis']['fXmax'] ]).range( [ -size, size ]);
         var utx = d3.scale.linear().domain([ -size, size ]).range([ histo['fXaxis']['fXmin'], histo['fXaxis']['fXmax'] ]);
      }
      if (logy) {
         var ty = d3.scale.log().domain([ histo['fYaxis']['fXmin'], histo['fYaxis']['fXmax'] ]).range( [ -size, size ]);
         var uty = d3.scale.log().domain([ size, -size ]).range([ histo['fYaxis']['fXmin'], histo['fYaxis']['fXmax'] ]);
      } else {
         var ty = d3.scale.linear().domain( [ histo['fYaxis']['fXmin'], histo['fYaxis']['fXmax'] ]).range([ -size, size ]);
         var uty = d3.scale.linear().domain([ size, -size ]).range([ histo['fYaxis']['fXmin'], histo['fYaxis']['fXmax'] ]);
      }
      if (logz) {
         var tz = d3.scale.log().domain([ histo['fZaxis']['fXmin'], histo['fZaxis']['fXmax'] ]).range([ -size, size ]);
         var utz = d3.scale.log().domain([ -size, size ]).range([ histo['fZaxis']['fXmin'], histo['fZaxis']['fXmax'] ]);
      } else {
         var tz = d3.scale.linear().domain([ histo['fZaxis']['fXmin'], histo['fZaxis']['fXmax'] ]).range([ -size, size ]);
         var utz = d3.scale.linear().domain([ -size, size ]).range([ histo['fZaxis']['fXmin'], histo['fZaxis']['fXmax'] ]);
      }

      // three.js 3D drawing
      var scene = new THREE.Scene();

      var toplevel = new THREE.Object3D();
      toplevel.rotation.x = 30 * Math.PI / 180;
      toplevel.rotation.y = 30 * Math.PI / 180;
      scene.add(toplevel);

      var wireMaterial = new THREE.MeshBasicMaterial({
         color : 0x000000,
         wireframe : true,
         wireframeLinewidth : 0.5,
         side : THREE.DoubleSide
      });

      // create a new mesh with cube geometry
      var cube = new THREE.Mesh(new THREE.BoxGeometry(size * 2, size * 2, size * 2), wireMaterial);

      var helper = new THREE.BoxHelper(cube);
      helper.material.color.set(0x000000);

      // add the cube to the scene
      toplevel.add(helper);

      var textMaterial = new THREE.MeshBasicMaterial({ color : 0x000000 });

      // add the calibration vectors and texts
      var geometry = new THREE.Geometry();
      var imax, istep, len = 3, plen, sin45 = Math.sin(45);
      var text3d, text;
      var xmajors = tx.ticks(5);
      var xminors = tx.ticks(25);
      for (var i = -size, j = 0, k = 0; i <= size; ++i) {
         var is_major = (utx(i) <= xmajors[j] && utx(i + 1) > xmajors[j]) ? true : false;
         var is_minor = (utx(i) <= xminors[k] && utx(i + 1) > xminors[k]) ? true : false;
         plen = (is_major ? len + 2 : len) * sin45;
         if (is_major) {
            text3d = new THREE.TextGeometry(xmajors[j], { size : 7, height : 0, curveSegments : 10 });
            ++j;

            text3d.computeBoundingBox();
            var centerOffset = 0.5 * (text3d.boundingBox.max.x - text3d.boundingBox.min.x);

            text = new THREE.Mesh(text3d, textMaterial);
            text.position.set(i - centerOffset, -size - 13, size + plen);
            toplevel.add(text);

            text = new THREE.Mesh(text3d, textMaterial);
            text.position.set(i + centerOffset, -size - 13, -size - plen);
            text.rotation.y = Math.PI;
            toplevel.add(text);
         }
         if (is_major || is_minor) {
            ++k;
            geometry.vertices.push(new THREE.Vector3(i, -size, size));
            geometry.vertices.push(new THREE.Vector3(i, -size - plen, size + plen));
            geometry.vertices.push(new THREE.Vector3(i, -size, -size));
            geometry.vertices.push(new THREE.Vector3(i, -size - plen, -size - plen));
         }
      }
      var ymajors = ty.ticks(5);
      var yminors = ty.ticks(25);
      for (var i = size, j = 0, k = 0; i > -size; --i) {
         var is_major = (uty(i) <= ymajors[j] && uty(i - 1) > ymajors[j]) ? true : false;
         var is_minor = (uty(i) <= yminors[k] && uty(i - 1) > yminors[k]) ? true : false;
         plen = (is_major ? len + 2 : len) * sin45;
         if (is_major) {
            text3d = new THREE.TextGeometry(ymajors[j], { size : 7, height : 0, curveSegments : 10 });
            ++j;

            text3d.computeBoundingBox();
            var centerOffset = 0.5 * (text3d.boundingBox.max.x - text3d.boundingBox.min.x);

            text = new THREE.Mesh(text3d, textMaterial);
            text.position.set(size + plen, -size - 13, i + centerOffset);
            text.rotation.y = Math.PI / 2;
            toplevel.add(text);

            text = new THREE.Mesh(text3d, textMaterial);
            text.position.set(-size - plen, -size - 13, i - centerOffset);
            text.rotation.y = -Math.PI / 2;
            toplevel.add(text);
         }
         if (is_major || is_minor) {
            ++k;
            geometry.vertices.push(new THREE.Vector3(size, -size, i));
            geometry.vertices.push(new THREE.Vector3(size + plen, -size - plen, i));
            geometry.vertices.push(new THREE.Vector3(-size, -size, i));
            geometry.vertices.push(new THREE.Vector3(-size - plen, -size - plen, i));
         }
      }
      var zmajors = tz.ticks(5);
      var zminors = tz.ticks(25);
      for (var i = -size, j = 0, k = 0; i <= size; ++i) {
         var is_major = (utz(i) <= zmajors[j] && utz(i + 1) > zmajors[j]) ? true : false;
         var is_minor = (utz(i) <= zminors[k] && utz(i + 1) > zminors[k]) ? true : false;
         plen = (is_major ? len + 2 : len) * sin45;
         if (is_major) {
            text3d = new THREE.TextGeometry(zmajors[j], { size : 7, height : 0, curveSegments : 10 });
            ++j;

            text3d.computeBoundingBox();
            var offset = 0.6 * (text3d.boundingBox.max.x - text3d.boundingBox.min.x);

            text = new THREE.Mesh(text3d, textMaterial);
            text.position.set(size + offset + 7, i - 2.5, size + offset + 7);
            text.rotation.y = Math.PI * 3 / 4;
            toplevel.add(text);

            text = new THREE.Mesh(text3d, textMaterial);
            text.position.set(size + offset + 7, i - 2.5, -size - offset - 7);
            text.rotation.y = -Math.PI * 3 / 4;
            toplevel.add(text);

            text = new THREE.Mesh(text3d, textMaterial);
            text.position.set(-size - offset - 7, i - 2.5, size + offset + 7);
            text.rotation.y = Math.PI / 4;
            toplevel.add(text);

            text = new THREE.Mesh(text3d, textMaterial);
            text.position.set(-size - offset - 7, i - 2.5, -size - offset - 7);
            text.rotation.y = -Math.PI / 4;
            toplevel.add(text);
         }
         if (is_major || is_minor) {
            ++k;
            geometry.vertices.push(new THREE.Vector3(size, i, size));
            geometry.vertices.push(new THREE.Vector3(size + plen, i, size + plen));
            geometry.vertices.push(new THREE.Vector3(size, i, -size));
            geometry.vertices.push(new THREE.Vector3(size + plen, i, -size - plen));
            geometry.vertices.push(new THREE.Vector3(-size, i, size));
            geometry.vertices.push(new THREE.Vector3(-size - plen, i, size + plen));
            geometry.vertices.push(new THREE.Vector3(-size, i, -size));
            geometry.vertices.push(new THREE.Vector3(-size - plen, i, -size - plen));
         }
      }

      // add the calibration lines
      var lineMaterial = new THREE.LineBasicMaterial({ color : 0x000000 });
      var line = new THREE.Line(geometry, lineMaterial);
      line.type = THREE.LinePieces;
      toplevel.add(line);

      // create the bin cubes
      var constx = (size * 2 / histo['fXaxis']['fNbins']) / maxbin;
      var consty = (size * 2 / histo['fYaxis']['fNbins']) / maxbin;
      var constz = (size * 2 / histo['fZaxis']['fNbins']) / maxbin;

      var optFlag = (opt.indexOf('colz') != -1 || opt.indexOf('col') != -1);
      var fcolor = d3.rgb(JSROOT.Painter.root_colors[histo['fFillColor']]);
      var fillcolor = new THREE.Color(0xDDDDDD);
      fillcolor.setRGB(fcolor.r / 255, fcolor.g / 255,  fcolor.b / 255);
      var bin, wei;
      for (var i = 0; i < bins.length; ++i) {
         wei = (optFlag ? maxbin : bins[i].n);
         if (opt.indexOf('box1') != -1) {
            bin = new THREE.Mesh(new THREE.SphereGeometry(0.5 * wei * constx /* , 16, 16 */),
                  new THREE.MeshPhongMaterial({  color : fillcolor.getHex(), specular : 0xbfbfbf/* , shading: THREE.NoShading */}));
         } else {
            bin = THREE.SceneUtils.createMultiMaterialObject(
                  new THREE.BoxGeometry(wei * constx, wei * constz, wei * consty),
                  [ new THREE.MeshLambertMaterial({ color : fillcolor.getHex(), shading : THREE.NoShading }), wireMaterial ]);
         }
         bin.position.x = tx(bins[i].x - (scalex / 2));
         bin.position.y = tz(bins[i].z - (scalez / 2));
         bin.position.z = -(ty(bins[i].y - (scaley / 2)));
         bin.name = "x: [" + bins[i].x.toPrecision(4) + ", "
                   + (bins[i].x + scalex).toPrecision(4) + "]<br/>"
                   + "y: [" + bins[i].y.toPrecision(4) + ", "
                   + (bins[i].y + scaley).toPrecision(4) + "]<br/>"
                   + "z: [" + bins[i].z.toPrecision(4) + ", "
                   + (bins[i].z + scalez).toPrecision(4) + "]<br/>"
                   + "entries: " + bins[i].n.toFixed();
         toplevel.add(bin);
      }
      // create a point light
      var pointLight = new THREE.PointLight(0xcfcfcf);
      pointLight.position.set(0, 50, 250);
      scene.add(pointLight);

      // var directionalLight = new THREE.DirectionalLight( 0x7f7f7f );
      // directionalLight.position.set( 0, -70, 100).normalize();
      // scene.add( directionalLight );

      var camera = new THREE.PerspectiveCamera(45, w / h, 1, 1000);
      camera.position.set(0, 0, 500);
      camera.lookat = cube;

      /**
       * @author alteredq / http://alteredqualia.com/
       * @author mr.doob / http://mrdoob.com/
       */
      var Detector = {
            canvas : !!window.CanvasRenderingContext2D,
            webgl : (function() {
               try {
                  return !!window.WebGLRenderingContext
                  && !!document.createElement('canvas')
                  .getContext('experimental-webgl');
               } catch (e) {
                  return false;
               }
            })(),
            workers : !!window.Worker,
            fileapi : window.File && window.FileReader
            && window.FileList && window.Blob
      };

      var renderer = Detector.webgl ?
                       new THREE.WebGLRenderer({ antialias : true }) :
                       new THREE.CanvasRenderer({antialias : true });
      renderer.setClearColor(0xffffff, 1);
      renderer.setSize(w, h);
      render_to.append(renderer.domElement);
      renderer.render(scene, camera);

      JSROOT.Painter.add3DInteraction(renderer, scene, camera, toplevel, null);

      return painter.DrawingReady();
   }

   return JSROOT.Painter;

}));

