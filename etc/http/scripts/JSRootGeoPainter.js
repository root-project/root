/** JavaScript ROOT 3D geometry painter
 * @file JSRootGeoPainter.js */

(function( factory ) {
   if ( typeof define === "function" && define.amd ) {
      // AMD. Register as an anonymous module.
      define( [ 'd3', 'JSRootPainter', 'threejs', 'JSRoot3DPainter', 'JSRootGeoBase' ], factory );
   } else {

      if (typeof JSROOT == 'undefined')
         throw new Error('JSROOT is not defined', 'JSRootGeoPainter.js');

      if (typeof JSROOT.Painter != 'object')
         throw new Error('JSROOT.Painter is not defined', 'JSRootGeoPainter.js');

      if (typeof d3 == 'undefined')
         throw new Error('d3 is not defined', 'JSRootGeoPainter.js');

      if (typeof THREE == 'undefined')
         throw new Error('THREE is not defined', 'JSRootGeoPainter.js');

      factory( d3, JSROOT, THREE );
   }
} (function( d3, JSROOT, THREE ) {


   if ( typeof define === "function" && define.amd )
      JSROOT.loadScript('$$$style/JSRootGeoPainter.css');

   if (typeof JSROOT.GEO !== 'object')
      console.error('JSROOT.GEO namespace is not defined')

   JSROOT.Toolbar = function(container, buttons) {
      if ((container !== undefined) && (typeof container.append == 'function'))  {
         this.element = container.append("div").attr('class','jsroot');
         this.addButtons(buttons);
      }
   }

   JSROOT.Toolbar.prototype.addButtons = function(buttons) {
      var pthis = this;

      this.buttonsNames = [];
      buttons.forEach(function(buttonGroup) {
         var group = pthis.element.append('div').attr('class', 'toolbar-group');

         buttonGroup.forEach(function(buttonConfig) {
            var buttonName = buttonConfig.name;
            if (!buttonName) {
               throw new Error('must provide button \'name\' in button config');
            }
            if (pthis.buttonsNames.indexOf(buttonName) !== -1) {
               throw new Error('button name \'' + buttonName + '\' is taken');
            }
            pthis.buttonsNames.push(buttonName);

            pthis.createButton(group, buttonConfig);
         });
      });
   };

   JSROOT.Toolbar.prototype.createButton = function(group, config) {

      var title = config.title;
      if (title === undefined) title = config.name;

      if (typeof config.click !== 'function')
         throw new Error('must provide button \'click\' function in button config');

      var button = group.append('a')
                        .attr('class','toolbar-btn')
                        .attr('rel', 'tooltip')
                        .attr('data-title', title)
                        .on('click', config.click);

      this.createIcon(button, config.icon || JSROOT.ToolbarIcons.question);
   };

   JSROOT.Toolbar.prototype.createIcon = function(button, thisIcon) {
      var size = thisIcon.size || 512,
          scale = thisIcon.scale || 1,
          svg = button.append("svg:svg")
                      .attr('height', '1em')
                      .attr('width', '1em')
                      .attr('viewBox', [0, 0, size, size].join(' '));

      if ('recs' in thisIcon) {
          var rec = {};
          for (var n=0;n<thisIcon.recs.length;++n) {
             JSROOT.extend(rec, thisIcon.recs[n]);
             svg.append('rect').attr("x", rec.x).attr("y", rec.y)
                               .attr("width", rec.w).attr("height", rec.h)
                               .attr("fill", rec.f);
          }
       } else {
          var elem = svg.append('svg:path').attr('d',thisIcon.path);
          if (scale !== 1)
             elem.attr('transform', 'scale(' + scale + ' ' + scale +')');
       }
   };

   JSROOT.Toolbar.prototype.removeAllButtons = function() {
      this.element.remove();
   };

   /**
    * @class JSROOT.TGeoPainter Holder of different functions and classes for drawing geometries
    */

   JSROOT.TGeoPainter = function( obj, is_manager ) {
      if (obj && (obj._typename.indexOf('TGeoVolume') === 0))
         obj = { _typename:"TGeoNode", fVolume: obj, fName: obj.fName, $geoh: obj.$geoh, _proxy: true };

      JSROOT.TObjectPainter.call(this, obj);

      this.no_default_title = true; // do not set title to main DIV

      this.is_geo_manager = is_manager; // only in manager name of top volume used in the item name

      this.Cleanup(true);
   }

   JSROOT.TGeoPainter.prototype = Object.create( JSROOT.TObjectPainter.prototype );

   JSROOT.TGeoPainter.prototype.CreateToolbar = function(args) {
      if (this._toolbar) return;
      var painter = this;
      var buttonList = [{
         name: 'toImage',
         title: 'Save as PNG',
         icon: JSROOT.ToolbarIcons.camera,
         click: function() {
            painter.Render3D(0);
            var dataUrl = painter._renderer.domElement.toDataURL("image/png");
            dataUrl.replace("image/png", "image/octet-stream");
            var link = document.createElement('a');
            if (typeof link.download === 'string') {
               document.body.appendChild(link); //Firefox requires the link to be in the body
               link.download = "geometry.png";
               link.href = dataUrl;
               link.click();
               document.body.removeChild(link); //remove the link when done
            }
         }
      }];

      if (JSROOT.gStyle.ContextMenu)
      buttonList.push({
         name: 'menu',
         title: 'Show context menu',
         icon: JSROOT.ToolbarIcons.question,
         click: function() {

            var evnt = d3.event;

            d3.event.preventDefault();
            d3.event.stopPropagation();

            JSROOT.Painter.createMenu(painter, function(menu) {
               menu.painter.FillContextMenu(menu);
               menu.show(evnt);
            });
         }
      });

      this._toolbar = new JSROOT.Toolbar( this.select_main(), [buttonList] );
   }

   JSROOT.TGeoPainter.prototype.ModifyVisisbility = function(name, sign) {
      if (JSROOT.GEO.NodeKind(this.GetObject()) !== 0) return;

      if (name == "")
         return JSROOT.GEO.SetBit(this.GetObject().fVolume, JSROOT.GEO.BITS.kVisThis, (sign === "+"));

      var regexp, exact = false;

      //arg.node.fVolume
      if (name.indexOf("*") < 0) {
         regexp = new RegExp("^"+name+"$");
         exact = true;
      } else {
         regexp = new RegExp("^" + name.split("*").join(".*") + "$");
         exact = false;
      }

      this.FindNodeWithVolume(regexp, function(arg) {
         JSROOT.GEO.InvisibleAll.call(arg.node.fVolume, (sign !== "+"));
         return exact ? arg : null; // continue search if not exact expression provided
      });
   }

   JSROOT.TGeoPainter.prototype.decodeOptions = function(opt) {
      var res = { _grid: false, _bound: false, _debug: false,
                  _full: false, _axis:false, _count:false, wireframe: false,
                   scale: new THREE.Vector3(1,1,1),
                   more: 1, maxlimit: 100000, maxnodeslimit: 3000,
                   use_worker: false, update_browser: true, show_controls: false,
                   highlight: false, select_in_view: false,
                   clipx: false, clipy: false, clipz: false, ssao: false,
                   script_name: "", transparancy: 1, autoRotate: false };

      var _opt = JSROOT.GetUrlOption('_grid');
      if (_opt !== null && _opt == "true") res._grid = true;
      var _opt = JSROOT.GetUrlOption('_debug');
      if (_opt !== null && _opt == "true") { res._debug = true; res._grid = true; }
      if (_opt !== null && _opt == "bound") { res._debug = true; res._grid = true; res._bound = true; }
      if (_opt !== null && _opt == "full") { res._debug = true; res._grid = true; res._full = true; res._bound = true; }

      var macro = opt.indexOf("macro:");
      if (macro>=0) {
         var separ = opt.indexOf(";", macro+6);
         if (separ<0) separ = opt.length;
         res.script_name = opt.substr(macro+6,separ-macro-6);
         opt = opt.substr(0, macro) + opt.substr(separ+1);
         console.log('script', res.script_name, 'rest', opt);
      }

      while (true) {
         var pp = opt.indexOf("+"), pm = opt.indexOf("-");
         if ((pp<0) && (pm<0)) break;
         var p1 = pp, sign = "+";
         if ((p1<0) || ((pm>=0) && (pm<pp))) { p1 = pm; sign = "-"; }

         var p2 = p1+1, regexp = new RegExp('[,; .]');
         while ((p2<opt.length) && !regexp.test(opt[p2]) && (opt[p2]!='+') && (opt[p2]!='-')) p2++;

         var name = opt.substring(p1+1, p2);
         opt = opt.substr(0,p1) + opt.substr(p2);
         // console.log("Modify visibility", sign,':',name);

         this.ModifyVisisbility(name, sign);
      }

      opt = opt.toLowerCase();

      function check(name) {
         var indx = opt.indexOf(name);
         if (indx<0) return false;
         opt = opt.substr(0, indx) + opt.substr(indx+name.length);
         return true;
      }

      function checkval(name, dflt) {
         var indx = opt.indexOf(name);
         if (indx<0) return dflt;
         opt = opt.substr(0, indx) + opt.substr(indx+name.length);
         var indx2 = indx;
         while ((indx2<opt.length) && (opt[indx2].match(/[0-9]/))) indx2++;
         if (indx2>indx) dflt = parseInt(opt.substr(indx, indx2-indx));
         opt = opt.substr(0,indx) + opt.substr(indx2);
         return dflt;
      }

      if (check("more3")) res.more = 3;
      if (check("more")) res.more = 2;
      if (check("all")) res.more = 100;

      if (check("invx") || check("invertx")) res.scale.x = -1;

      if (check("controls") || check("ctrl")) res.show_controls = true;

      if (check("clipxyz")) res.clipx = res.clipy = res.clipz = true;
      if (check("clipx")) res.clipx = true;
      if (check("clipy")) res.clipy = true;
      if (check("clipz")) res.clipz = true;
      if (check("clip")) res.clipx = res.clipy = res.clipz = true;

      if (check("dflt_colors")) this.SetRootDefaultColors();
      if (check("ssao")) res.ssao = true;

      if (check("noworker")) res.use_worker = -1;
      if (check("worker")) res.use_worker = 1;

      if (check("highlight")) res.highlight = true;

      if (check("wire")) res.wireframe = true;
      if (check("rotate")) res.autoRotate = true;

      if (check("invy")) res.scale.y = -1;
      if (check("invz")) res.scale.z = -1;

      if (check("count")) res._count = true;

      res.transparancy = checkval('transp', 100)/100;

      if (check("axis") || check("a")) { res._axis = true; res._yup = false; }

      if (check("d")) res._debug = true;
      if (check("g")) res._grid = true;
      if (check("b")) res._bound = true;
      if (check("w")) res.wireframe = true;
      if (check("f")) res._full = true;
      if (check("y")) res._yup = true;
      if (check("z")) res._yup = false;

      return res;
   }

   JSROOT.TGeoPainter.prototype.ActiavteInBrowser = function(names, force) {
      // if (this.GetItemName() === null) return;

      if (typeof names == 'string') names = [ names ];

      if (JSROOT.hpainter) {
         // show browser if it not visible
         JSROOT.hpainter.actiavte(names, force);

         // if highlight in the browser disabled, suppress in few seconds
         if (!this.options.update_browser)
            setTimeout(function() { JSROOT.hpainter.actiavte([]); }, 2000);
      }
   }

   JSROOT.TGeoPainter.prototype.TestMatrixes = function() {
      // method can be used to check matrix calculations with current three.js model

      var painter = this, errcnt = 0, totalcnt = 0, totalmax = 0;

      var arg = {
            domatrix: true,
            func: function(node) {

               var m2 = this.getmatrix();

               var entry = this.CopyStack();

               var mesh = painter._clones.CreateObject3D(entry.stack, painter._toplevel, 'mesh');

               if (!mesh) return true;

               totalcnt++;

               var m1 = mesh.matrixWorld, flip, origm2;

               if (m1.equals(m2)) return true
               if ((m1.determinant()>0) && (m2.determinant()<-0.9)) {
                  flip = THREE.Vector3(1,1,-1);
                  origm2 = m2;
                  m2 = m2.clone().scale(flip);
                  if (m1.equals(m2)) return true;
               }

               var max = 0;
               for (var k=0;k<16;++k)
                  max = Math.max(max, Math.abs(m1.elements[k] - m2.elements[k]));

               totalmax = Math.max(max, totalmax);

               if (max < 1e-4) return true;

               console.log(painter._clones.ResolveStack(entry.stack).name, 'maxdiff', max, 'determ', m1.determinant(), m2.determinant());

               errcnt++;

               return false;
            }
         };


      tm1 = new Date().getTime();

      var cnt = this._clones.ScanVisible(arg);

      tm2 = new Date().getTime();

      console.log('Compare matrixes total',totalcnt,'errors',errcnt, 'takes', tm2-tm1, 'maxdiff', totalmax);
   }


   JSROOT.TGeoPainter.prototype.FillContextMenu = function(menu) {
      menu.add("header: Draw options");

      menu.addchk(this.options.update_browser, "Browser update", function() {
         this.options.update_browser = !this.options.update_browser;
         if (!this.options.update_browser) this.ActiavteInBrowser([]);
      });
      menu.addchk(this.options.show_controls, "Show Controls", function() {
         this.options.show_controls = !this.options.show_controls;
         this.showControlOptions(this.options.show_controls);
      });
      menu.addchk(this.TestAxisVisibility, "Show axes", function() {
         this.toggleAxisDraw();
      });
      menu.addchk(this.options.wireframe, "Wire frame", function() {
         this.options.wireframe = !this.options.wireframe;
         this.changeWireFrame(this._scene, this.options.wireframe);
      });
      menu.addchk(this.options.highlight, "Highlight volumes", function() {
         this.options.highlight = !this.options.highlight;
      });
      menu.addchk(this.options.wireframe, "Reset camera position", function() {
         this.focusCamera();
         this.Render3D();
      });
      menu.addchk(this.options.autoRotate, "Autorotate", function() {
         this.options.autoRotate = !this.options.autoRotate;
         this.autorotate(2.5);
      });
      menu.addchk(this.options.select_in_view, "Select in view", function() {
         this.options.select_in_view = !this.options.select_in_view;
         if (this.options.select_in_view) this.startDrawGeometry();
      });
   }

   JSROOT.TGeoPainter.prototype.changeGlobalTransparancy = function(value, skip_render) {
      this._toplevel.traverse( function (node) {
         if (node instanceof THREE.Mesh) {
            if (node.material.alwaysTransparent !== undefined) {
               if (!node.material.alwaysTransparent) {
                  node.material.transparent = value !== 1.0;
               }
               node.material.opacity = Math.min(value * value, node.material.inherentOpacity);
            }

         }
      });
      if (!skip_render) this.Render3D(0);
   }

   JSROOT.TGeoPainter.prototype.showControlOptions = function(on) {

      if (this._datgui) {
         if (on) return;
         this._datgui.destroy();
         delete this._datgui;
         return;
      }
      if (!on) return;

      var painter = this;

      this._datgui = new dat.GUI({ width: Math.min(650, painter._renderer.domElement.width / 2) });

      // Clipping Options

      var bound = new THREE.Box3().setFromObject(this._toplevel);
      bound.expandByVector(bound.getSize().multiplyScalar(0.01));

      var clipFolder = this._datgui.addFolder('Clipping');

      var toggleX = clipFolder.add(this, 'enableX').name('Enable X').listen();
      toggleX.onChange( function (value) {
         painter.enableX = value;
         painter._enableSSAO = value ? false : painter._enableSSAO;
         painter.updateClipping();
      });

      if (this.clipX === 0)
         this.clipX = (bound.min.x+bound.max.x)/2;
      var xclip = clipFolder.add(this, 'clipX', bound.min.x, bound.max.x).name('X Position');

      xclip.onChange( function (value) {
         painter.clipX = value;
         if (painter.enableX) painter.updateClipping();
      });

      var toggleY = clipFolder.add(this, 'enableY').name('Enable Y').listen();
      toggleY.onChange( function (value) {
         painter.enableY = value;
         painter._enableSSAO = value ? false : painter._enableSSAO;
         painter.updateClipping();
      });

      if (this.clipY === 0)
         this.clipY = (bound.min.y + bound.max.y)/2;
      var yclip = clipFolder.add(this, 'clipY', bound.min.y, bound.max.y).name('Y Position');

      yclip.onChange( function (value) {
         painter.clipY = value;
         if (painter.enableY) painter.updateClipping();
      });

      var toggleZ = clipFolder.add(this, 'enableZ').name('Enable Z').listen();
      toggleZ.onChange( function (value) {
         painter.enableZ = value;
         painter._enableSSAO = value ? false : painter._enableSSAO;
         painter.updateClipping();
      });

      if (this.clipZ === 0)
         this.clipZ = (bound.min.z + bound.max.z) / 2;
      var zclip = clipFolder.add(this, 'clipZ', bound.min.z, bound.max.z).name('Z Position');

      zclip.onChange( function (value) {
         painter.clipZ = value;
         if (painter.enableZ) painter.updateClipping();
      });

      // Appearance Options

      var appearance = this._datgui.addFolder('Appearance');

      if (this._webgl) {
         appearance.add(this, '_enableSSAO').name('Smooth Lighting (SSAO)').onChange( function (value) {
            painter._renderer.antialias = !painter._renderer.antialias;
            painter.enableX = value ? false : painter.enableX;
            painter.enableY = value ? false : painter.enableY;
            painter.enableZ = value ? false : painter.enableZ;
            painter.updateClipping();
         }).listen();
      }

      appearance.add(this.options, 'highlight').name('Highlight Selection').onChange( function (value) {
         if (value === false) {
            if (painter._selected.mesh !== null) {
               painter._selected.mesh.material.color = painter._selected.originalColor;
               painter.Render3D(0);
               painter._selected.mesh = null;
            }
         }
      });

      appearance.add(this.options, 'transparancy', 0.0, 1.0)
                     .listen().onChange(this.changeGlobalTransparancy.bind(this));

      appearance.add(this.options, 'wireframe').name('Wireframe').onChange( function (value) {
         painter.changeWireFrame(painter._scene, painter.options.wireframe);
      });

      appearance.add(this, 'focusCamera').name('Reset camera position');

      // Advanced Options

      if (this._webgl) {
         var advanced = this._datgui.addFolder('Advanced');

         advanced.add( this._advceOptions, 'aoClamp', 0.0, 1.0).listen().onChange( function (value) {
            painter._ssaoPass.uniforms[ 'aoClamp' ].value = value;
            painter._enableSSAO = true;
            painter.Render3D(0);
         });

         advanced.add( this._advceOptions, 'lumInfluence', 0.0, 1.0).listen().onChange( function (value) {
            painter._ssaoPass.uniforms[ 'lumInfluence' ].value = value;
            painter._enableSSAO = true;
            painter.Render3D(0);
         });

         advanced.add( this._advceOptions, 'clipIntersection').listen().onChange( function (value) {
            painter.clipIntersection = value;
            painter.updateClipping();
         });

         advanced.add(this._advceOptions, 'depthTest').onChange( function (value) {
            painter._toplevel.traverse( function (node) {
               if (node instanceof THREE.Mesh) {
                  node.material.depthTest = value;
               }
            });
            painter.Render3D(0);
         }).listen();

         advanced.add(this, 'resetAdvanced').name('Reset');
      }
   }


   JSROOT.TGeoPainter.prototype.OrbitContext = function(evnt, intersects) {

      JSROOT.Painter.createMenu(this, function(menu) {
         var numitems = 0, numnodes = 0, cnt = 0;
         if (intersects)
            for (var n=0;n<intersects.length;++n) {
               if (intersects[n].object.stack) numnodes++;
               if (intersects[n].object.geo_name) numitems++;
            }

         if (numnodes + numitems === 0) {
            menu.painter.FillContextMenu(menu);
         } else {
            var many = (numnodes + numitems) > 1;

            if (many) menu.add("header:" + ((numitems > 0) ? "Items" : "Nodes"));

            for (var n=0;n<intersects.length;++n) {
               var obj = intersects[n].object,
                   name, itemname, hdr;

               if (obj.geo_name) {
                  itemname = obj.geo_name;
                  name = itemname.substr(itemname.lastIndexOf("/")+1);
                  if (!name) name = itemname;
                  hdr = name;
               } else
               if (obj.stack) {
                  name = menu.painter._clones.ResolveStack(obj.stack).name;
                  itemname = menu.painter.GetStackFullName(obj.stack);
                  hdr = menu.painter.GetItemName();
                  if (name.indexOf("Nodes/") === 0) hdr = name.substr(6); else
                  if (name.length > 0) hdr = name; else
                  if (!hdr) hdr = "header";

               } else
                  continue;


               menu.add((many ? "sub:" : "header:") + hdr, itemname, function(arg) { this.ActiavteInBrowser([arg], true); });

               menu.add("Browse", itemname, function(arg) { this.ActiavteInBrowser([arg], true); });

               if (obj.geo_name) {
                  menu.add("Hide", n, function(indx) {
                     var mesh = intersects[indx].object;
                     mesh.visible = false; // just disable mesh
                     if (mesh.geo_object) mesh.geo_object._hidden_via_menu = true; // and hide object for further redraw
                     menu.painter.Render3D();
                  });

                  if (many) menu.add("endsub:");

                  continue;
               }

               var wireframe = menu.painter.accessObjectWireFrame(obj);

               if (wireframe!==undefined)
                  menu.addchk(wireframe, "Wireframe", n, function(indx) {
                     var m = intersects[indx].object.material;
                     m.wireframe = !m.wireframe;
                     this.Render3D();
                  });

               if (++cnt>1)
                  menu.add("Manifest", n, function(indx) {

                     if (this._last_manifest)
                        this._last_manifest.wireframe = !this._last_manifest.wireframe;

                     if (this._last_hidden)
                        this._last_hidden.forEach(function(obj) { obj.visible = true; });

                     this._last_hidden = [];

                     for (var i=0;i<indx;++i)
                        this._last_hidden.push(intersects[i].object);

                     this._last_hidden.forEach(function(obj) { obj.visible = false; });

                     this._last_manifest = intersects[indx].object.material;

                     this._last_manifest.wireframe = !this._last_manifest.wireframe;

                     this.Render3D();
                  });


               menu.add("Focus", n, function(indx) {
                  this.focusCamera(intersects[indx].object);
               });

               menu.add("Hide", n, function(indx) {
                  var resolve = menu.painter._clones.ResolveStack(intersects[indx].object.stack);

                  if (resolve.obj && (resolve.node.kind === 0) && resolve.obj.fVolume) {
                     JSROOT.GEO.SetBit(resolve.obj.fVolume, JSROOT.GEO.BITS.kVisThis, false);
                     JSROOT.GEO.updateBrowserIcons(resolve.obj.fVolume, JSROOT.hpainter);
                  } else
                  if (resolve.obj && (resolve.node.kind === 1)) {
                     resolve.obj.fRnrSelf = false;
                     JSROOT.GEO.updateBrowserIcons(resolve.obj, JSROOT.hpainter);
                  }
                  // intersects[arg].object.visible = false;
                  // this.Render3D();

                  this.testGeomChanges();// while many volumes may disapper, recheck all of them
               });

               if (many) menu.add("endsub:");
            }
         }
         menu.show(evnt);
      });
   }

   JSROOT.TGeoPainter.prototype.FilterIntersects = function(intersects) {

      // remove all elements without stack - indicator that this is geometry object
      for (var n=intersects.length-1; n>=0; --n) {

         var obj = intersects[n].object;

         var unique = (obj.stack !== undefined) || (obj.geo_name !== undefined);

         for (var k=0;(k<n) && unique;++k)
            if (intersects[k].object === obj) unique = false;

         if (!unique) intersects.splice(n,1);
      }

      if (this.enableX || this.enableY || this.enableZ ) {
         var clippedIntersects = [];

         for (var i = 0; i < intersects.length; ++i) {
            var clipped = false;
            var point = intersects[i].point;

            if (this.enableX && this._clipPlanes[0].normal.dot(point) > this._clipPlanes[0].constant ) {
               clipped = true;
            }
            if (this.enableY && this._clipPlanes[1].normal.dot(point) > this._clipPlanes[1].constant ) {
               clipped = true;
            }
            if (this.enableZ && this._clipPlanes[2].normal.dot(point) > this._clipPlanes[2].constant ) {
               clipped = true;
            }

            if (clipped)
               clippedIntersects.push(intersects[i]);
         }

         intersects = clippedIntersects;
      }

      return intersects;
   }

   JSROOT.TGeoPainter.prototype.testCameraPositionChange = function() {
      // function analyzes camera position and start redraw of geometry if
      // objects in view may be changed

      if (!this.options.select_in_view || this._draw_all_nodes) return;


      var matrix = JSROOT.GEO.CreateProjectionMatrix(this._camera);

      var frustum = JSROOT.GEO.CreateFrustum(matrix);

      // check if overall bounding box seen
      if (!frustum.CheckBox(new THREE.Box3().setFromObject(this._toplevel)))
         this.startDrawGeometry();
   }

   JSROOT.TGeoPainter.prototype.ResolveStack = function(stack) {
      return this._clones && stack ? this._clones.ResolveStack(stack) : null;
   }

   JSROOT.TGeoPainter.prototype.GetStackFullName = function(stack) {
      var mainitemname = this.GetItemName(),
          sub = this.ResolveStack(stack);

      if (!sub || !sub.name) return mainitemname;
      return mainitemname ? (mainitemname + "/" + sub.name) : sub.name;
   }

   JSROOT.TGeoPainter.prototype.addOrbitControls = function() {

      if (this._controls) return;

      var painter = this;

      this._controls = JSROOT.Painter.CreateOrbitControl(this, this._camera, this._scene, this._renderer, this._lookat);

      this._controls.ContextMenu = this.OrbitContext.bind(this);

      this._controls.ProcessMouseMove = function(intersects) {

         var tooltip = null, resolve = null;

         if (painter.options.highlight) {

            if (painter._selected.mesh !== null) {
               painter._selected.mesh.material.color = painter._selected.originalColor;
               if (painter._selected.mesh.hightlightLineWidth)
                  painter._selected.mesh.material.linewidth = painter._selected.mesh.hightlightLineWidth/3;
               if (painter._selected.mesh.highlightMarkerSize)
                  painter._selected.mesh.material.size = painter._selected.mesh.highlightMarkerSize/3;
            }

            if (intersects.length > 0) {
               painter._selected.mesh = intersects[0].object;
               painter._selected.originalColor = painter._selected.mesh.material.color;
               painter._selected.mesh.material.color = new THREE.Color( 0xffaa33 );
               if (painter._selected.mesh.hightlightLineWidth)
                  painter._selected.mesh.material.linewidth = painter._selected.mesh.hightlightLineWidth;
               if (painter._selected.mesh.highlightMarkerSize)
                  painter._selected.mesh.material.size = painter._selected.mesh.highlightMarkerSize;
               painter.Render3D(0);

               if (intersects[0].object.stack) {
                  tooltip = painter.GetStackFullName(intersects[0].object.stack);
                  if (tooltip) resolve = painter.ResolveStack(intersects[0].object.stack);
               }
               else if (intersects[0].object.geo_name)
                  tooltip = intersects[0].object.geo_name;
            }
         }

         if (intersects.length === 0 && painter._selected.mesh !== null) {
            painter._selected.mesh.material.color = painter._selected.originalColor;
            painter.Render3D(0);
            painter._selected.mesh = null;
         }

         var names = [];

         if (painter.options.update_browser) {
            if (painter.options.highlight) {
               if (tooltip !== null) names.push(tooltip);
            } else {
               for (var n=0;n<intersects.length;++n) {
                  var obj = intersects[n].object;
                  if (obj.geo_name) names.push(obj.geo_name); else
                  if (obj.stack) names.push(painter.GetStackFullName(obj.stack));
               }
            }

            painter.ActiavteInBrowser(names);
         }

         if (!resolve || !resolve.obj) return tooltip;

        return { name: resolve.obj.fName, title: resolve.obj.fTitle || resolve.obj._typename, line: tooltip };
      }

      this._controls.ProcessMouseLeave = function() {
         if (painter.options.update_browser)
            painter.ActiavteInBrowser([]);
      }

      this._controls.ProcessDblClick = function() {
         if (painter._last_manifest) {
            painter._last_manifest.wireframe = !painter._last_manifest.wireframe;
            if (painter._last_hidden)
               painter._last_hidden.forEach(function(obj) { obj.visible = true; });
            delete painter._last_hidden;
            delete painter._last_manifest;
            painter.Render3D();
         } else {
            painter.adjustCameraPosition();
         }
      }
   }

   JSROOT.TGeoPainter.prototype.addTransformControl = function() {
      if (this._tcontrols) return;

      if (! this.options._debug && !this.options._grid ) return;

      // FIXME: at the moment THREE.TransformControls is bogus in three.js, should be fixed and check again
      //return;

      this._tcontrols = new THREE.TransformControls( this._camera, this._renderer.domElement );
      this._scene.add( this._tcontrols );
      this._tcontrols.attach( this._toplevel );
      //this._tcontrols.setSize( 1.1 );
      var painter = this;

      window.addEventListener( 'keydown', function ( event ) {
         switch ( event.keyCode ) {
         case 81: // Q
            painter._tcontrols.setSpace( painter._tcontrols.space === "local" ? "world" : "local" );
            break;
         case 17: // Ctrl
            painter._tcontrols.setTranslationSnap( Math.ceil( painter._overall_size ) / 50 );
            painter._tcontrols.setRotationSnap( THREE.Math.degToRad( 15 ) );
            break;
         case 84: // T (Translate)
            painter._tcontrols.setMode( "translate" );
            break;
         case 82: // R (Rotate)
            painter._tcontrols.setMode( "rotate" );
            break;
         case 83: // S (Scale)
            painter._tcontrols.setMode( "scale" );
            break;
         case 187:
         case 107: // +, =, num+
            painter._tcontrols.setSize( painter._tcontrols.size + 0.1 );
            break;
         case 189:
         case 109: // -, _, num-
            painter._tcontrols.setSize( Math.max( painter._tcontrols.size - 0.1, 0.1 ) );
            break;
         }
      });
      window.addEventListener( 'keyup', function ( event ) {
         switch ( event.keyCode ) {
         case 17: // Ctrl
            painter._tcontrols.setTranslationSnap( null );
            painter._tcontrols.setRotationSnap( null );
            break;
         }
      });

      this._tcontrols.addEventListener( 'change', function() { painter.Render3D(0); });
   }


   JSROOT.TGeoPainter.prototype.createFlippedMesh = function(parent, shape, material) {
      // when transformation matrix includes one or several invertion of axis,
      // one should inverse geometry object, otherwise THREE.js cannot correctly draw it

      var flip =  new THREE.Vector3(1,1,-1);

      if (shape.geomZ === undefined) {

         if (shape.geom.type == 'BufferGeometry') {

            var pos = shape.geom.getAttribute('position').array,
                norm = shape.geom.getAttribute('normal').array,
                len = pos.length, n, shift = 0,
                newpos = new Float32Array(len),
                newnorm = new Float32Array(len);

            // we should swap second and third point in each face
            for (n=0; n<len; n+=3) {
               newpos[n]   = pos[n+shift];
               newpos[n+1] = pos[n+1+shift];
               newpos[n+2] = -pos[n+2+shift];

               newnorm[n]   = norm[n+shift];
               newnorm[n+1] = norm[n+1+shift];
               newnorm[n+2] = -norm[n+2+shift];

               shift+=3; if (shift===6) shift=-3; // values 0,3,-3
            }

            shape.geomZ = new THREE.BufferGeometry();
            shape.geomZ.addAttribute( 'position', new THREE.BufferAttribute( newpos, 3 ) );
            shape.geomZ.addAttribute( 'normal', new THREE.BufferAttribute( newnorm, 3 ) );
            // normals are calculated with normal geometry and correctly scaled
            // geom.computeVertexNormals();

         } else {

            shape.geomZ = shape.geom.clone();

            shape.geomZ.scale(flip.x, flip.y, flip.z);

            var face, d;
            for (var n=0;n<shape.geomZ.faces.length;++n) {
               face = geom.faces[n];
               d = face.b; face.b = face.c; face.c = d;
            }

            // normals are calculated with normal geometry and correctly scaled
            // geom.computeFaceNormals();
         }
      }

      var mesh = new THREE.Mesh( shape.geomZ, material );
      mesh.scale.copy(flip);
      mesh.updateMatrix();

      return mesh;
   }


   JSROOT.TGeoPainter.prototype.nextDrawAction = function() {
      // return false when nothing todo
      // return true if one could perform next action immediately
      // return 1 when call after short timeout required
      // return 2 when call must be done from processWorkerReply

      if (!this._clones || (this.drawing_stage == 0)) return false;

      if (this.drawing_stage == 1) {

         // wait until worker is really started
         if (this.options.use_worker>0) {
            if (!this._worker) { this.startWorker(); return 1; }
            if (!this._worker_ready) return 1;
         }

         // first copy visibility flags and check how many unique visible nodes exists
         var numvis = this._clones.MarkVisisble(),
             matrix = null, frustum = null;

         if (this.options.select_in_view && !this._first_drawing) {
            // extract camera projection matrix for selection

            matrix = JSROOT.GEO.CreateProjectionMatrix(this._camera);

            frustum = JSROOT.GEO.CreateFrustum(matrix);

            // check if overall bounding box seen
            if (frustum.CheckBox(new THREE.Box3().setFromObject(this._toplevel))) {
               matrix = null; // not use camera for the moment
               frustum = null;
            }
         }

         this._current_face_limit = this.options.maxlimit;
         this._current_nodes_limit = this.options.maxnodeslimit;
         if (matrix) {
            this._current_face_limit*=1.25;
            this._current_nodes_limit*=1.25;
         }

         // here we decide if we need worker for the drawings
         // main reason - too large geometry and large time to scan all camera positions
         var need_worker = (numvis > 10000) || (matrix && (this._clones.ScanVisible() > 1e5));

         // worker does not work when starting from file system
         if (need_worker && JSROOT.source_dir.indexOf("file://")==0) {
            console.log('disable worker for jsroot from file system');
            need_worker = false;
         }

         if (need_worker && !this._worker && (this.options.use_worker >= 0))
            this.startWorker(); // we starting worker, but it may not be ready so fast

         if (!need_worker || !this._worker_ready) {
            //var tm1 = new Date().getTime();
            var res = this._clones.CollectVisibles(this._current_face_limit, frustum, this._current_nodes_limit);
            this._new_draw_nodes = res.lst;
            this._draw_all_nodes = res.complete;
            //var tm2 = new Date().getTime();
            //console.log('Collect visibles', this._new_draw_nodes.length, 'takes', tm2-tm1);
            this.drawing_stage = 3;
            return true;
         }

         var job = {
               collect: this._current_face_limit,   // indicator for the command
               collect_nodes: this._current_nodes_limit,
               visible: this._clones.GetVisibleFlags(),
               matrix: matrix ? matrix.elements : null
         };

         this.submitToWorker(job);

         this.drawing_stage = 2;

         this.drawing_log = "Worker select visibles";

         return 2; // we now waiting for the worker reply
      }

      if (this.drawing_stage == 2) {
         // do nothing, we are waiting for worker reply

         this.drawing_log = "Worker select visibles";

         return 2;
      }

      if (this.drawing_stage == 3) {
         // here we merge new and old list of nodes for drawing,
         // normally operation is fast and can be implemented with one call

         this.drawing_log = "Analyse visibles";

         if (this._draw_nodes) {
            var del = this._clones.MergeVisibles(this._new_draw_nodes, this._draw_nodes);
            // remove should be fast, do it here
            for (var n=0;n<del.length;++n)
               this._clones.CreateObject3D(del[n].stack, this._toplevel, 'delete_mesh');

            if (del.length > 0)
               this.drawing_log = "Delete " + del.length + " nodes";
         }

         this._draw_nodes = this._new_draw_nodes;
         delete this._new_draw_nodes;
         this.drawing_stage = 4;
         return true;
      }

      if (this.drawing_stage === 4) {

         this.drawing_log = "Collect shapes";

         // collect shapes
         var shapes = this._clones.CollectShapes(this._draw_nodes);

         // merge old and new list with produced shapes
         this._build_shapes = this._clones.MergeShapesLists(this._build_shapes, shapes);

         this.drawing_stage = 5;
         return true;
      }


      if (this.drawing_stage === 5) {
         // this is building of geometries,
         // one can ask worker to build them or do it ourself

         if (this.canSubmitToWorker()) {
            var job = { limit: this._current_face_limit, shapes: [] }, cnt = 0;
            for (var n=0;n<this._build_shapes.length;++n) {
               var clone = null, item = this._build_shapes[n];
               // only submit not-done items
               if (item.ready || item.geom) {
                  // this is place holder for existing geometry
                  clone = { id: item.id, ready: true, nfaces: JSROOT.GEO.numGeometryFaces(item.geom), refcnt: item.refcnt };
               } else {
                  clone = JSROOT.clone(item, null, true);
                  cnt++;
               }

               job.shapes.push(clone);
            }

            if (cnt > 0) {
               /// only if some geom missing, submit job to the worker
               this.submitToWorker(job);
               this.drawing_log = "Worker build shapes";
               this.drawing_stage = 6;
               return 2;
            }
         }

         this.drawing_stage = 7;
      }

      if (this.drawing_stage === 6) {
         // waiting shapes from the worker, worker should activate our code
         return 2;
      }

      if ((this.drawing_stage === 7) || (this.drawing_stage === 8)) {

         if (this.drawing_stage === 7) {
            // building shapes
            var res = this._clones.BuildShapes(this._build_shapes, this._current_face_limit, 500);
            if (res.done) {
               this.drawing_stage = 8;
            } else {
               this.drawing_log = "Creating: " + res.shapes + " / " + this._build_shapes.length + " shapes,  "  + res.faces + " faces";
               if (res.notusedshapes < 30) return true;
            }
         }

         // final stage, create all meshes

         var tm0 = new Date().getTime(), ready = true;

         for (var n=0; n<this._draw_nodes.length;++n) {
            var entry = this._draw_nodes[n];
            if (entry.done) continue;

            var shape = this._build_shapes[entry.shapeid];
            if (!shape.ready) {
               if (this.drawing_stage === 8) console.warn('shape marked as not ready when should');
               ready = false;
               continue;
            }

            entry.done = true;
            shape.used = true; // indicate that shape was used in building

            if (!shape.geom || (shape.nfaces === 0)) {
               // node is visible, but shape does not created
               this._clones.CreateObject3D(entry.stack, this._toplevel, 'delete_mesh');
               continue;
            }

            var obj3d = this._clones.CreateObject3D(entry.stack, this._toplevel, this.options);

/*
            var info = this._clones.ResolveStack(entry.stack, true), ndiff = 0;
            for (var n=0;n<16;++n) {
               var v1 = info.matrix.elements[n], v2 = obj3d.matrixWorld.elements[n];
               mean = Math.abs(v1+v2)/2;
               if ((mean > 1e-5) && (Math.abs(v2-v1)/mean > 1e-6)) ndiff++;
            }
            if (ndiff>0) console.log('Mismatch for ' + info.name, info.matrix.elements, obj3d.matrixWorld.elements);
*/

            var nodeobj = this._clones.origin[entry.nodeid];
            var clone = this._clones.nodes[entry.nodeid];
            var prop = JSROOT.GEO.getNodeProperties(clone.kind, nodeobj, true);

            this._num_meshes++;
            this._num_faces += shape.nfaces;

            prop.material.wireframe = this.options.wireframe;

            prop.material.side = this.bothSides ? THREE.DoubleSide : THREE.FrontSide;

            var mesh;

            if (obj3d.matrixWorld.determinant() > -0.9) {
               mesh = new THREE.Mesh( shape.geom, prop.material );
            } else {
               mesh = this.createFlippedMesh(obj3d, shape, prop.material);
            }

            // keep full stack of nodes
            mesh.stack = entry.stack;

            obj3d.add(mesh);

            if (this.options._debug || this.options._full) {
               var helper = new THREE.WireframeHelper(mesh);
               helper.material.color.set(prop.fillcolor);
               helper.material.linewidth = ('fVolume' in nodeobj) ? nodeobj.fVolume.fLineWidth : 1;
               obj3d.add(helper);
            }

            if (this.options._bound || this.options._full) {
               var boxHelper = new THREE.BoxHelper( mesh );
               obj3d.add( boxHelper );
            }

            var tm1 = new Date().getTime();
            if (tm1 - tm0 > 500) { ready = false; break; }
         }

         if (ready) {
            this.drawing_log = "Building done";
            this.drawing_stage = 0;
            return false;
         }

         if (this.drawing_stage > 7)
            this.drawing_log = "Building meshes " + this._num_meshes + " / " + this._num_faces;
         return true;
      }

      console.log('never come here');

      return false;


   }

   JSROOT.TGeoPainter.prototype.SameMaterial = function(node1, node2) {

      if ((node1===null) || (node2===null)) return node1 === node2;

      if (node1.fVolume.fLineColor >= 0)
         return (node1.fVolume.fLineColor === node2.fVolume.fLineColor);

       var m1 = (node1.fVolume.fMedium !== null) ? node1.fVolume.fMedium.fMaterial : null;
       var m2 = (node2.fVolume.fMedium !== null) ? node2.fVolume.fMedium.fMaterial : null;

       if (m1 === m2) return true;

       if ((m1 === null) || (m2 === null)) return false;

       return (m1.fFillStyle === m2.fFillStyle) && (m1.fFillColor === m2.fFillColor);
    }

   JSROOT.TGeoPainter.prototype.createScene = function(webgl, w, h, pixel_ratio) {
      // three.js 3D drawing
      this._scene = new THREE.Scene();
      this._scene.fog = new THREE.Fog(0xffffff, 1, 10000);
      this._scene.overrideMaterial = new THREE.MeshLambertMaterial( { color: 0x7000ff, transparent: true, opacity: 0.2, depthTest: false } );

      this._scene_width = w;
      this._scene_height = h;

      this._camera = new THREE.PerspectiveCamera(25, w / h, 1, 10000);

      this._camera.up = this.options._yup ? new THREE.Vector3(0,1,0) : new THREE.Vector3(0,0,1);
      this._scene.add( this._camera );

      this._selected = {mesh:null, originalColor:null};

      this._overall_size = 10;

      this._toplevel = new THREE.Object3D();

      this._scene.add(this._toplevel);

      this._renderer = webgl ?
                        new THREE.WebGLRenderer({ antialias : true, logarithmicDepthBuffer: false,
                                                  preserveDrawingBuffer: true }) :
                        new THREE.CanvasRenderer({antialias : true });
      this._renderer.setPixelRatio(pixel_ratio);
      this._renderer.setClearColor(0xffffff, 1);
      this._renderer.setSize(w, h);
      this._renderer.localClippingEnabled = true;

      this._animating = false;

      // Clipping Planes

      this.clipIntersection = true;
      this.bothSides = false; // which material kind should be used
      this.enableX = this.enableY = this.enableZ = false;
      this.clipX = this.clipY = this.clipZ = 0.0;

      this._clipPlanes = [ new THREE.Plane(new THREE.Vector3( 1, 0, 0), this.clipX),
                           new THREE.Plane(new THREE.Vector3( 0, this.options._yup ? -1 : 1, 0), this.clipY),
                           new THREE.Plane(new THREE.Vector3( 0, 0, this.options._yup ? 1 : -1), this.clipZ) ];

       // Lights

      //var light = new THREE.HemisphereLight( 0xffffff, 0x999999, 0.5 );
      //this._scene.add(light);

      /*
      var directionalLight = new THREE.DirectionalLight( 0xffffff, 0.2 );
      directionalLight.position.set( 0, 1, 0 );
      this._scene.add( directionalLight );

      this._lights = new THREE.Object3D();
      var a = new THREE.PointLight(0xefefef, 0.2);
      var b = new THREE.PointLight(0xefefef, 0.2);
      var c = new THREE.PointLight(0xefefef, 0.2);
      var d = new THREE.PointLight(0xefefef, 0.2);
      this._lights.add(a);
      this._lights.add(b);
      this._lights.add(c);
      this._lights.add(d);
      this._camera.add( this._lights );
      a.position.set( 20000, 20000, 20000 );
      b.position.set( -20000, 20000, 20000 );
      c.position.set( 20000, -20000, 20000 );
      d.position.set( -20000, -20000, 20000 );
      */
      this._pointLight = new THREE.PointLight(0xefefef, 1);
      this._camera.add( this._pointLight );
      this._pointLight.position.set(10, 10, 10);
      //*/

      // Default Settings

      this._defaultAdvanced = { aoClamp: 0.70,
                                lumInfluence: 0.4,
                              //  shininess: 100,
                                clipIntersection: true,
                                depthTest: true
                              };

      // Smooth Lighting Shader (Screen Space Ambient Occulsion)
      // http://threejs.org/examples/webgl_postprocessing_ssao.html

      this._enableSSAO = this.options.ssao;

      if (webgl) {
         var renderPass = new THREE.RenderPass( this._scene, this._camera );
         // Setup depth pass
         this._depthMaterial = new THREE.MeshDepthMaterial( { side: THREE.DoubleSide });
         this._depthMaterial.depthPacking = THREE.RGBADepthPacking;
         this._depthMaterial.blending = THREE.NoBlending;
         var pars = { minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter };
         this._depthRenderTarget = new THREE.WebGLRenderTarget( w, h, pars );
         // Setup SSAO pass
         this._ssaoPass = new THREE.ShaderPass( THREE.SSAOShader );
         this._ssaoPass.renderToScreen = true;
         this._ssaoPass.uniforms[ "tDepth" ].value = this._depthRenderTarget.texture;
         this._ssaoPass.uniforms[ 'size' ].value.set( w, h );
         this._ssaoPass.uniforms[ 'cameraNear' ].value = this._camera.near;
         this._ssaoPass.uniforms[ 'cameraFar' ].value = this._camera.far;
         this._ssaoPass.uniforms[ 'onlyAO' ].value = false;//( postprocessing.renderMode == 1 );
         this._ssaoPass.uniforms[ 'aoClamp' ].value = this._defaultAdvanced.aoClamp;
         this._ssaoPass.uniforms[ 'lumInfluence' ].value = this._defaultAdvanced.lumInfluence;
         // Add pass to effect composer
         this._effectComposer = new THREE.EffectComposer( this._renderer );
         this._effectComposer.addPass( renderPass );
         this._effectComposer.addPass( this._ssaoPass );
      }

      this._advceOptions = {};
      this.resetAdvanced();
   }


   JSROOT.TGeoPainter.prototype.startDrawGeometry = function(force) {

      if (!force && (this.drawing_stage!==0)) {
         this._draw_nodes_again = true;
         return;
      }

      this._startm = new Date().getTime();
      this._last_render_tm = this._startm;
      this._last_render_meshes = 0;
      this.drawing_stage = 1;
      this.drawing_log = "collect visible";
      this._num_meshes = 0;
      this._num_faces = 0;

      delete this._last_manifest;
      delete this._last_hidden; // clear list of hidden objects

      delete this._draw_nodes_again; // forget about such flag

      this.continueDraw();
   }

   JSROOT.TGeoPainter.prototype.resetAdvanced = function() {
      if (this._webgl) {
         this._advceOptions.aoClamp = this._defaultAdvanced.aoClamp;
         this._advceOptions.lumInfluence = this._defaultAdvanced.lumInfluence;

         this._ssaoPass.uniforms[ 'aoClamp' ].value = this._defaultAdvanced.aoClamp;
         this._ssaoPass.uniforms[ 'lumInfluence' ].value = this._defaultAdvanced.lumInfluence;
      }

      this._advceOptions.depthTest = this._defaultAdvanced.depthTest;
      this._advceOptions.clipIntersection = this._defaultAdvanced.clipIntersection;
      this.clipIntersection = this._defaultAdvanced.clipIntersection;

      var painter = this;
      this._toplevel.traverse( function (node) {
         if (node instanceof THREE.Mesh) {
            node.material.depthTest = painter._defaultAdvanced.depthTest;
         }
      });

      this.Render3D(0);
   }

   JSROOT.TGeoPainter.prototype.updateMaterialSide = function(both_sides, force) {
      if ((this.bothSides === both_sides) && !force) return;

      this._scene.traverse( function(obj) {
         if (obj.hasOwnProperty("material") && ('emissive' in obj.material)) {
            obj.material.side = both_sides ? THREE.DoubleSide : THREE.FrontSide;
            obj.material.needsUpdate = true;
        }
      });
      this.bothSides = both_sides;
   }

   JSROOT.TGeoPainter.prototype.updateClipping = function(without_render) {
      this._clipPlanes[0].constant = this.clipX;
      this._clipPlanes[1].constant = -this.clipY;
      this._clipPlanes[2].constant = this.options._yup ? -this.clipZ : this.clipZ;

      var painter = this;
      this._scene.traverse( function (node) {
         if (node instanceof THREE.Mesh) {
            node.material.clipIntersection = painter.clipIntersection;
            node.material.clippingPlanes = [];
            if (painter.enableX) node.material.clippingPlanes.push(painter._clipPlanes[0]);
            if (painter.enableY) node.material.clippingPlanes.push(painter._clipPlanes[1]);
            if (painter.enableZ) node.material.clippingPlanes.push(painter._clipPlanes[2]);
         }
      });

      this.updateMaterialSide(this.enableX || this.enableY || this.enableZ);

      if (!without_render) this.Render3D(0);
   }

   JSROOT.TGeoPainter.prototype.adjustCameraPosition = function(first_time) {

      if (!this._toplevel) return;

      var extras = this.getExtrasContainer('get');
      if (extras) this._toplevel.remove(extras);

      var box = new THREE.Box3().setFromObject(this._toplevel);

      if (extras) this._toplevel.add(extras);

      var sizex = box.max.x - box.min.x,
          sizey = box.max.y - box.min.y,
          sizez = box.max.z - box.min.z,
          midx = (box.max.x + box.min.x)/2,
          midy = (box.max.y + box.min.y)/2,
          midz = (box.max.z + box.min.z)/2;

      this._overall_size = 2 * Math.max( sizex, sizey, sizez);

      this._scene.fog.near = this._overall_size * 2;
      this._camera.near = this._overall_size / 350;
      this._scene.fog.far = this._overall_size * 12;
      this._camera.far = this._overall_size * 12;

      if (this._webgl) {
         this._ssaoPass.uniforms[ 'cameraNear' ].value = this._camera.near;//*this._nFactor;
         this._ssaoPass.uniforms[ 'cameraFar' ].value = this._camera.far;///this._nFactor;
      }

      if (first_time) {
         this.clipX = midx;
         this.clipY = midy;
         this.clipZ = midz;
      }

      // this._camera.far = 100000000000;

      this._camera.updateProjectionMatrix();

      if (this.options._yup) {
         this._camera.position.set(midx-2*Math.max(sizex,sizez), midy+2*sizey, midz-2*Math.max(sizex,sizez));
      } else {
         this._camera.position.set(midx-2*Math.max(sizex,sizey), midy-2*Math.max(sizex,sizey), midz+2*sizez);
      }

      this._lookat = new THREE.Vector3(midx, midy, midz);
      this._camera.lookAt(this._lookat);

      this._pointLight.position.set(sizex/5, sizey/5, sizez/5);

      if (this._controls) {
         this._controls.target.copy(this._lookat);
         this._controls.update();
      }

      // recheck which elements to draw
      if (this.options.select_in_view)
         this.startDrawGeometry();
   }

   JSROOT.TGeoPainter.prototype.focusOnItem = function(itemname) {

      if (!itemname || !this._clones) return;

      var stack = this._clones.FindStackByName(itemname);

      if (!stack) return;

      var info = this._clones.ResolveStack(stack, true);

      this.focusCamera( info, false );
   }

   JSROOT.TGeoPainter.prototype.focusCamera = function( focus, clip ) {

      var autoClip = clip === undefined ? false : clip;

      var box = new THREE.Box3();
      if (focus === undefined) {
         box.setFromObject(this._toplevel);
      } else if (focus instanceof THREE.Mesh) {
         box.setFromObject(focus);
      } else {
         var center = new THREE.Vector3().setFromMatrixPosition(focus.matrix);
         var node = focus.node;
         var halfDelta = new THREE.Vector3( node.fDX, node.fDY, node.fDZ ).multiplyScalar(0.5);
         box.min = center.clone().sub(halfDelta) ;
         box.max = center.clone().add(halfDelta) ;
      }

      var sizex = box.max.x - box.min.x,
          sizey = box.max.y - box.min.y,
          sizez = box.max.z - box.min.z,
          midx = (box.max.x + box.min.x)/2,
          midy = (box.max.y + box.min.y)/2,
          midz = (box.max.z + box.min.z)/2;

      var position;
      if (this.options._yup)
         position = new THREE.Vector3(midx-2*Math.max(sizex,sizez), midy+2*sizey, midz-2*Math.max(sizex,sizez));
      else
         position = new THREE.Vector3(midx-2*Math.max(sizex,sizey), midy-2*Math.max(sizex,sizey), midz+2*sizez);

      var target = new THREE.Vector3(midx, midy, midz);
      //console.log("Zooming to x: " + target.x + " y: " + target.y + " z: " + target.z );


      // Find to points to animate "lookAt" between
      var dist = this._camera.position.distanceTo(target);
      var oldTarget = this._controls.target;

      var frames = 200;
      var step = 0;
      // Amount to change camera position at each step
      var posIncrement = position.sub(this._camera.position).divideScalar(frames);
      // Amount to change "lookAt" so it will end pointed at target
      var targetIncrement = target.sub(oldTarget).divideScalar(frames);
      // console.log( targetIncrement );

      // Automatic Clipping

      if (autoClip) {

         var topBox = new THREE.Box3().setFromObject(this._toplevel);

         this.clipX = this.enableX ? this.clipX : topBox.min.x;
         this.clipY = this.enableY ? this.clipY : topBox.min.y;
         this.clipZ = this.enableZ ? this.clipZ : topBox.min.z;

         this.enableX = this.enableY = this.enableZ = true;

         // These should be center of volume, box may not be doing this correctly
         var incrementX  = ((box.max.x + box.min.x) / 2 - this.clipX) / frames,
             incrementY  = ((box.max.y + box.min.y) / 2 - this.clipY) / frames,
             incrementZ  = ((box.max.z + box.min.z) / 2 - this.clipZ) / frames;

         this.updateClipping();
      }

      var painter = this;
      this._animating = true;

      // Interpolate //

      function animate() {
         if (painter._animating === undefined) return;

         if (painter._animating) {
            requestAnimationFrame( animate );
         } else {
            painter.startDrawGeometry();
         }
         var smoothFactor = -Math.cos( ( 2.0 * Math.PI * step ) / frames ) + 1.0;
         painter._camera.position.add( posIncrement.clone().multiplyScalar( smoothFactor ) );
         oldTarget.add( targetIncrement.clone().multiplyScalar( smoothFactor ) );
         painter._lookat = oldTarget;
         painter._camera.lookAt( painter._lookat );
         painter._camera.updateProjectionMatrix();
         if (autoClip) {
            painter.clipX += incrementX * smoothFactor;
            painter.clipY += incrementY * smoothFactor;
            painter.clipZ += incrementZ * smoothFactor;
            painter.updateClipping();
         } else {
            painter.Render3D();
         }
         step++;
         painter._animating = step < frames;
      }
      animate();

   //   this._controls.update();

   }

   JSROOT.TGeoPainter.prototype.autorotate = function(speed) {

      var rotSpeed = (speed === undefined) ? 2.0 : speed,
          painter = this, last = new Date();

      function animate() {
         if (!painter._renderer || !painter.options) return;

         var current = new Date();

         if ( painter.options.autoRotate ) requestAnimationFrame( animate );

         if (painter._controls) {
            painter._controls.autoRotate = painter.options.autoRotate;
            painter._controls.autoRotateSpeed = rotSpeed * ( current.getTime() - last.getTime() ) / 16.6666;
            painter._controls.update();
         }
         last = new Date();
         painter.Render3D(0);
      }
      animate();
   }

   JSROOT.TGeoPainter.prototype.completeScene = function() {

      if ( this.options._debug || this.options._grid ) {
         if ( this.options._full ) {
            var boxHelper = new THREE.BoxHelper(this._toplevel);
            this._scene.add( boxHelper );
         }
         this._scene.add( new THREE.AxisHelper( 2 * this._overall_size ) );
         this._scene.add( new THREE.GridHelper( Math.ceil( this._overall_size), Math.ceil( this._overall_size ) / 50 ) );
         this.helpText("<font face='verdana' size='1' color='red'><center>Transform Controls<br>" +
               "'T' translate | 'R' rotate | 'S' scale<br>" +
               "'+' increase size | '-' decrease size<br>" +
               "'W' toggle wireframe/solid display<br>"+
         "keep 'Ctrl' down to snap to grid</center></font>");
      }
   }


   JSROOT.TGeoPainter.prototype.drawCount = function(unqievis, clonetm) {

      var res = 'Unique nodes: ' + this._clones.nodes.length + '<br/>' +
                'Unique visible: ' + unqievis + '<br/>' +
                'Time to clone: ' + clonetm + 'ms <br/>';

      // need to fill cached value line numvischld
      this._clones.ScanVisible();

      var arg = {
         cnt: [],
         func: function(node) {
            if (this.cnt[this.last]===undefined)
               this.cnt[this.last] = 1;
            else
               this.cnt[this.last]++;
            return true;
         }
      };

      var tm1 = new Date().getTime();
      var numvis = this._clones.ScanVisible(arg);
      var tm2 = new Date().getTime();

      res += 'Total visible nodes: ' + numvis + '<br/>';

      for (var lvl=0;lvl<arg.cnt.length;++lvl) {
         if (arg.cnt[lvl] !== undefined)
            res += ('  lvl' + lvl + ': ' + arg.cnt[lvl] + '<br/>');
      }

      res += "Time to scan: " + (tm2-tm1) + "ms <br/>";

      res += "<br/><br/>Check timing for matrix calculations ...<br/>";

      var elem = this.select_main().style('overflow', 'auto').html(res);

      var painter = this;

      setTimeout(function() {
         arg.domatrix = true;
         tm1 = new Date().getTime();
         numvis = painter._clones.ScanVisible(arg);
         tm2 = new Date().getTime();
         elem.append("p").text("Time to scan with matrix: " + (tm2-tm1) + "ms");
      }, 100);

      return this.DrawingReady();
   }


   JSROOT.TGeoPainter.prototype.PerformDrop = function(obj, itemname, hitem) {

      if (this.drawExtras(obj, itemname, true)) {
         if (hitem) hitem._painter = this; // set for the browser item back pointer
         this.Render3D(100);
      }

      return null;
   }

   JSROOT.TGeoPainter.prototype.MouseOverHierarchy = function(on, itemname, hitem) {
      // function called when mouse is going over the item in the browser

      if (!this.options) return; // protection for cleaned-up painter

      var painter = this, obj = hitem._obj, mesh = null;
      if (this.options._debug)
         console.log('Mouse over', on, itemname, (hitem._obj ? hitem._obj._typename : "---"));

      // let's highlight tracks and hits only for the time being
      if (!hitem._obj || (hitem._obj._typename !== "TEveTrack" &&
          hitem._obj._typename !== "TEvePointSet")) return;

      // Be aware, that item name is real name in browser (with potentially cycle number in the name)
      // One can use object to identify which track should be highlighted
      painter.getExtrasContainer().children.some(function(node, index) {
         if (node.geo_object === obj) { mesh = node; return true; }
         return false;
      });
      if (mesh && on) {
         painter._selected.mesh = mesh;
         painter._selected.originalColor = mesh.material.color;
         painter._selected.originalSize = mesh.material.size;
         painter._selected.originalLineWidth = mesh.material.linewidth;
         painter._selected.mesh.material.color = new THREE.Color( 0x00ff00 );
         painter._selected.mesh.material.size *= 2;
         painter._selected.mesh.material.linewidth *= 2;
      }
      else if (painter._selected.mesh) {
         if (painter._selected.originalColor)
            painter._selected.mesh.material.color = painter._selected.originalColor;
         if (painter._selected.originalSize)
            painter._selected.mesh.material.size = painter._selected.originalSize;
         if (painter._selected.originalLineWidth)
            painter._selected.mesh.material.linewidth = painter._selected.originalLineWidth;
      }
      painter.Render3D(0);
   }

   JSROOT.TGeoPainter.prototype.addExtra = function(obj, itemname) {

      // register extra objects like tracks or hits
      // Check if object already exists to prevent duplication

      if (this._extraObjects === undefined)
         this._extraObjects = JSROOT.Create("TList");

      if (this._extraObjects.arr.indexOf(obj)>=0) return false;

      this._extraObjects.Add(obj, itemname);

      delete obj._hidden_via_menu; // remove previous hidden property

      return true;
   }

   JSROOT.TGeoPainter.prototype.ExtraObjectVisible = function(itemname, toggle) {
      if (!this._extraObjects) return;

      var indx = this._extraObjects.opt.indexOf(itemname);

      if (indx < 0) return;

      var obj = this._extraObjects.arr[indx];

      var res = obj._hidden_via_menu ? false : true;

      if (toggle) {
         obj._hidden_via_menu = res; res = !res;

         var mesh = null;
         // either found painted object or just draw once again
         this._toplevel.traverse(function(node) { if (node.geo_object === obj) mesh = node; });

         if (mesh) mesh.visible = res; else
         if (res) this.drawExtras(obj);

         if (mesh || res) this.Render3D();
      }

      return res;
   }


   JSROOT.TGeoPainter.prototype.drawExtras = function(obj, itemname, add_objects) {
      if (!obj || obj._typename===undefined) return false;

      // if object was hidden via menu, do not redraw it with next draw call
      if (!add_objects && obj._hidden_via_menu) return false;

      var isany = false;

      if (obj._typename === "TList") {
         if (!obj.arr) return false;
         for (var n=0;n<obj.arr.length;++n) {
            var sobj = obj.arr[n];
            var sname = (itemname === undefined) ? obj.opt[n] : (itemname + "/[" + n + "]");
            if (this.drawExtras(sobj, sname, add_objects)) isany = true;
         }
      } else
      if (obj._typename === 'TEveTrack') {
         if (add_objects && !this.addExtra(obj, itemname)) return false;
         isany = this.drawTrack(obj, itemname);
      } else
      if (obj._typename === 'TEvePointSet') {
         if (add_objects && !this.addExtra(obj, itemname)) return false;
         isany = this.drawHit(obj, itemname);
      }

      return isany;
   }

   JSROOT.TGeoPainter.prototype.getExtrasContainer = function(action) {
      if (!this._toplevel) return null;

      var extras = null;
      for (var n=0;n<this._toplevel.children.length;++n) {
         var chld = this._toplevel.children[n];
         if (chld._extras) { extras = chld; break; }
      }

      if (action==="delete") {
         if (extras) this._toplevel.remove(extras);
         JSROOT.Painter.DisposeThreejsObject(extras);
         return null;
      }

      if ((action!=="get") && !extras) {
         extras = new THREE.Object3D();
         extras._extras = true;
         this._toplevel.add(extras);
      }

      return extras;
   }


   JSROOT.TGeoPainter.prototype.drawTrack = function(track, itemname) {
      if (!track) return false;
      if (track.fN <= 0) return false;

      var track_width = track.fLineWidth;

      var track_color = JSROOT.Painter.root_colors[track.fLineColor];
      if (track_color == undefined) track_color = "rgb(255,0,255)";

      if (JSROOT.browser.isWin) track_width = 1; // not supported on windows

      var buf = new Float32Array((track.fN-1)*6), pos = 0;

      for (var k=0;k<track.fN-1;++k) {
         buf[pos]   = track.fP[k*3];
         buf[pos+1] = track.fP[k*3+1];
         buf[pos+2] = track.fP[k*3+2];
         buf[pos+3] = track.fP[k*3+3];
         buf[pos+4] = track.fP[k*3+4];
         buf[pos+5] = track.fP[k*3+5];
         pos+=6;
      }

      var geom = new THREE.BufferGeometry();
      geom.addAttribute( 'position', new THREE.BufferAttribute( buf, 3 ) );
      var lineMaterial = new THREE.LineBasicMaterial({ color: track_color, linewidth: track_width });
      var line = new THREE.LineSegments(geom, lineMaterial);

      line.geo_name = itemname;
      line.geo_object = track;
      if (!JSROOT.browser.isWin) line.hightlightLineWidth = track_width*3;

      this.getExtrasContainer().add(line);

      return true;
   }

   JSROOT.TGeoPainter.prototype.drawHit = function(hit, itemname) {
      if (!hit) return false;
      if (hit.fN <= 0) return false;

      var hit_size = 25.0 * hit.fMarkerSize;
      var hit_color = JSROOT.Painter.root_colors[hit.fMarkerColor];

      var use_points = this._webgl,
      size = hit.fN-1, step = 1, scale = hit_size*0.3,
      indicies = JSROOT.Painter.Box_Indexes,
      normals = JSROOT.Painter.Box_Normals,
      vertices = JSROOT.Painter.Box_Vertices,
      lll = 0, pos, norm;

      if (use_points) {
         pos = new Float32Array(size*3);
         norm = null;
      } else {
         // TODO: provide support of POINTS directly in the CanvasRenderer

         if (size > 1000) { step = Math.floor(size/500); if (step<2) step = 2; }

         pos = new Float32Array(indicies.length*3*Math.floor(size/step));
         norm = new Float32Array(indicies.length*3*Math.floor(size/step));
      }

      for (var i=0;i<size;i+=step) {

         var x = hit.fP[i*3],
         y = hit.fP[i*3+1],
         z = hit.fP[i*3+2];

         if (use_points) {
            pos[lll]   = x;
            pos[lll+1] = y;
            pos[lll+2] = z;
            lll+=3;
            continue;
         }

         for (var k=0,nn=-3;k<indicies.length;++k) {
            var vert = vertices[indicies[k]];
            pos[lll]   = x + (vert.x-0.5)*scale;
            pos[lll+1] = y + (vert.y-0.5)*scale;
            pos[lll+2] = z + (vert.z-0.5)*scale;

            if (k%6===0) nn+=3;
            norm[lll] = normals[nn];
            norm[lll+1] = normals[nn+1];
            norm[lll+2] = normals[nn+2];

            lll+=3;
         }
      }

      var geom = new THREE.BufferGeometry(), mesh;
      geom.addAttribute( 'position', new THREE.BufferAttribute( pos, 3 ) );
      if (norm) geom.addAttribute( 'normal', new THREE.BufferAttribute( norm, 3 ) );

      if (use_points) {
         var material = new THREE.PointsMaterial( { size: hit_size, color: hit_color } );
         mesh = new THREE.Points(geom, material);
         mesh.highlightMarkerSize = hit_size*3;
      } else {
         // var material = new THREE.MeshPhongMaterial({ color : fcolor, specular : 0x4f4f4f});
         var material = new THREE.MeshBasicMaterial( { color: hit_color, shading: THREE.SmoothShading  } );
         mesh = new THREE.Mesh(geom, material);

      }
      mesh.geo_name = itemname;
      mesh.geo_object = hit;

      this.getExtrasContainer().add(mesh);

      return true;
   }

   JSROOT.TGeoPainter.prototype.FindNodeWithVolume = function(name, action, prnt, itemname, volumes) {

      var first_level = false, res = null;

      if (!prnt) {
         prnt = this.GetObject();
         if (!prnt && (JSROOT.GEO.NodeKind(prnt)!==0)) return null;
         itemname = this.is_geo_manager ? prnt.fName : "";
         first_level = true;
         volumes = [];
      } else {
         if (itemname.length>0) itemname += "/";
         itemname += prnt.fName;
      }

      if (!prnt.fVolume || prnt.fVolume._searched) return null;

      if (name.test(prnt.fVolume.fName)) {
         res = action({ node: prnt, item: itemname });
         if (res) return res;
      }

      prnt.fVolume._searched = true;
      volumes.push(prnt.fVolume);

      if (prnt.fVolume.fNodes)
         for (var n=0;n<prnt.fVolume.fNodes.arr.length;++n) {
            res = this.FindNodeWithVolume(name, action, prnt.fVolume.fNodes.arr[n], itemname, volumes);
            if (res) break;
         }

      if (first_level)
         for (var n=0, len=volumes.length; n<len; ++n)
            delete volumes[n]._searched;

      return res;
   }

   JSROOT.TGeoPainter.prototype.SetRootDefaultColors = function() {
      // set default colors like TGeoManager::DefaultColors() does

      var dflt = { kWhite:0,   kBlack:1,   kGray:920,
                     kRed:632, kGreen:416, kBlue:600, kYellow:400, kMagenta:616, kCyan:432,
                     kOrange:800, kSpring:820, kTeal:840, kAzure:860, kViolet:880, kPink:900 };

      var nmax = 110, col = [];
      for (var i=0;i<nmax;i++) col.push(dflt.kGray);

      //here we should create a new TColor with the same rgb as in the default
      //ROOT colors used below
      col[ 3] = dflt.kYellow-10;
      col[ 4] = col[ 5] = dflt.kGreen-10;
      col[ 6] = col[ 7] = dflt.kBlue-7;
      col[ 8] = col[ 9] = dflt.kMagenta-3;
      col[10] = col[11] = dflt.kRed-10;
      col[12] = dflt.kGray+1;
      col[13] = dflt.kBlue-10;
      col[14] = dflt.kOrange+7;
      col[16] = dflt.kYellow+1;
      col[20] = dflt.kYellow-10;
      col[24] = col[25] = col[26] = dflt.kBlue-8;
      col[29] = dflt.kOrange+9;
      col[79] = dflt.kOrange-2;

      var name = { test:function() { return true; }} // select all volumes

      this.FindNodeWithVolume(name, function(arg) {
         var vol = arg.node.fVolume;
         var med = vol.fMedium;
         if (!med) return null;
         var mat = med.fMaterial;
         var matZ = Math.round(mat.fZ);
         vol.fLineColor = col[matZ];
         if (mat.fDensity<0.1) mat.fFillStyle = 3000+60; // vol.SetTransparency(60)
      });
   }


   JSROOT.TGeoPainter.prototype.checkScript = function(script_name, call_back) {

      var painter = this, draw_obj = this.GetObject(), name_prefix = "";

      if (this.is_geo_manager) name_prefix = draw_obj.fName;

      if (!script_name || (script_name.length<3) || (JSROOT.GEO.NodeKind(draw_obj)!==0))
         return JSROOT.CallBack(call_back, draw_obj, name_prefix);

      var mgr = {
            GetVolume: function (name) {
               var regexp = new RegExp("^"+name+"$");
               var currnode = painter.FindNodeWithVolume(regexp, function(arg) { return arg; } );

               if (!currnode) console.log('Did not found '+name + ' volume');

               // return proxy object with several methods, typically used in ROOT geom scripts
               return {
                   found: currnode,
                   fVolume: currnode ? currnode.node.fVolume : null,
                   InvisibleAll: function(flag) {
                      JSROOT.GEO.InvisibleAll.call(this.fVolume, flag);
                   },
                   Draw: function() {
                      if (!this.found || !this.fVolume) return;
                      draw_obj = this.found.node;
                      name_prefix = this.found.item;
                      console.log('Select volume for drawing', this.fVolume.fName, name_prefix);
                   },
                   SetTransparency: function(lvl) {
                     if (this.fVolume && this.fVolume.fMedium && this.fVolume.fMedium.fMaterial)
                        this.fVolume.fMedium.fMaterial.fFillStyle = 3000+lvl;
                   },
                   SetLineColor: function(col) {
                      if (this.fVolume) this.fVolume.fLineColor = col;
                   }
                };
            },

            DefaultColors: function() {
               painter.SetRootDefaultColors();
            },

            SetMaxVisNodes: function(limit) {
               console.log('Automatic visible depth for ' + limit + ' nodes');
               if (limit>0) painter.options.maxnodeslimit = limit;
            }
          };

      JSROOT.progress('Loading macro ' + script_name);

      var xhr = JSROOT.NewHttpRequest(script_name, "text", function(res) {
         if (!res || (res.length==0))
            return JSROOT.CallBack(call_back, draw_obj, name_prefix);

         var lines = res.split('\n');

         ProcessNextLine(0);

         function ProcessNextLine(indx) {

            var first_tm = new Date().getTime();
            while (indx < lines.length) {
               var line = lines[indx++].trim();

               if (line.indexOf('//')==0) continue;

               if (line.indexOf('gGeoManager')<0) continue;
               line = line.replace('->GetVolume','.GetVolume');
               line = line.replace('->InvisibleAll','.InvisibleAll');
               line = line.replace('->SetMaxVisNodes','.SetMaxVisNodes');
               line = line.replace('->DefaultColors','.DefaultColors');
               line = line.replace('->Draw','.Draw');
               line = line.replace('->SetTransparency','.SetTransparency');
               line = line.replace('->SetLineColor','.SetLineColor');
               if (line.indexOf('->')>=0) continue;

               // console.log(line);

               try {
                  var func = new Function('gGeoManager',line);
                  func(mgr);
               } catch(err) {
                  console.error('Problem by processing ' + line);
               }

               var now = new Date().getTime();
               if (now - first_tm > 300) {
                  JSROOT.progress('exec ' + line);
                  return setTimeout(ProcessNextLine.bind(this,indx),1);
               }
            }

            JSROOT.CallBack(call_back, draw_obj, name_prefix);
         }

      });

      xhr.send(null);
   }

   JSROOT.TGeoPainter.prototype.prepareObjectDraw = function(draw_obj, name_prefix) {
      var tm1 = new Date().getTime();

      this._clones = new JSROOT.GEO.ClonedNodes(draw_obj);

      this._clones.name_prefix = name_prefix;

      var uniquevis = this._clones.MarkVisisble(true);
      if (uniquevis <= 0)
         uniquevis = this._clones.MarkVisisble(false);
      else
         uniquevis = this._clones.MarkVisisble(true, true); // copy bits once and use normal visibility bits

      var tm2 = new Date().getTime();

      console.log('Creating clones', this._clones.nodes.length, 'takes', tm2-tm1, 'uniquevis', uniquevis);

      if (this.options._count)
         return this.drawCount(uniquevis, tm2-tm1);

      // this is limit for the visible faces, number of volumes does not matter
      this.options.maxlimit = (this._webgl ? 200000 : 100000) * this.options.more;

      this._first_drawing = true;

      // activate worker
      if (this.options.use_worker > 0) this.startWorker();

      var size = this.size_for_3d();

      this.createScene(this._webgl, size.width, size.height, window.devicePixelRatio);

      this.add_3d_canvas(size, this._renderer.domElement);

      // set top painter only when first child exists

      this.AccessTopPainter(true);

      this.CreateToolbar();

      this.startDrawGeometry(true);
   }

   JSROOT.TGeoPainter.prototype.DrawGeometry = function(opt, divid) {
      if (typeof opt !== 'string') opt = "";

      this._webgl = JSROOT.Painter.TestWebGL();

      this.options = this.decodeOptions(opt);

      if (!('_yup' in this.options))
         this.options._yup = this.svg_canvas().empty();

      // this.options.script_name = 'http://jsroot.gsi.de/files/geom/geomAlice.C'

      this.checkScript(this.options.script_name, this.prepareObjectDraw.bind(this));

      return this;
   }

   JSROOT.TGeoPainter.prototype.continueDraw = function() {

      // nothing to do - exit
      if (this.drawing_stage === 0) return;

      var tm0 = new Date().getTime(),
          interval = this._first_drawing ? 1000 : 200,
          now = tm0;

      while(true) {

         var res = this.nextDrawAction();

         if (!res) break;

         now = new Date().getTime();

         // stop creation after 100 sec, render as is
         if (now - this._startm > 1e5) {
            this.drawing_stage = 0;
            break;
         }

         // if we are that fast, do next action
         if ((res===true) && (now - tm0 < interval)) continue;

         if ((now - tm0 > interval) || (res === 1) || (res === 2)) {

            JSROOT.progress(this.drawing_log);

            if (this._first_drawing && this._webgl && (this._num_meshes - this._last_render_meshes > 100) && (now - this._last_render_tm > 2.5*interval)) {
               this.adjustCameraPosition();
               this.Render3D(-1);
               this._last_render_tm = new Date().getTime();
               this._last_render_meshes = this._num_meshes;
            }
            if (res !== 2) setTimeout(this.continueDraw.bind(this), (res === 1) ? 100 : 1);
            return;
         }
      }

      var take_time = now - this._startm;

      if (this._first_drawing)
         JSROOT.console('Create tm = ' + take_time + ' meshes ' + this._num_meshes + ' faces ' + this._num_faces);

      if (take_time > 300) {
         JSROOT.progress('Rendering geometry');
         return setTimeout(this.completeDraw.bind(this, true), 10);
      }

      this.completeDraw(true);
   }

   JSROOT.TGeoPainter.prototype.Render3D = function(tmout, measure) {
      if (!this._renderer) {
         console.warn('renderer object not exists - check code');
         return;
      }

      if (tmout === undefined) tmout = 5; // by default, rendering happens with timeout

      if (tmout <= 0) {
         if ('render_tmout' in this)
            clearTimeout(this.render_tmout);

         var tm1 = new Date();

         if (typeof this.TestAxisVisibility === 'function')
            this.TestAxisVisibility(this._camera, this._toplevel);

         // do rendering, most consuming time
         if (this._webgl && this._enableSSAO) {
            this._scene.overrideMaterial = this._depthMaterial;
        //    this._renderer.logarithmicDepthBuffer = false;
            this._renderer.render(this._scene, this._camera, this._depthRenderTarget, true);
            this._scene.overrideMaterial = null;
            this._effectComposer.render();
         } else {
       //     this._renderer.logarithmicDepthBuffer = true;
            this._renderer.render(this._scene, this._camera);
         }

         var tm2 = new Date();

         this.last_render_tm = tm2.getTime() - tm1.getTime();

         delete this.render_tmout;

         if ((this.first_render_tm === 0) && measure) {
            this.first_render_tm = this.last_render_tm;
            JSROOT.console('First render tm = ' + this.first_render_tm);
         }

         return;
      }

      // do not shoot timeout many times
      if (!this.render_tmout)
         this.render_tmout = setTimeout(this.Render3D.bind(this,0,measure), tmout);
   }


   JSROOT.TGeoPainter.prototype.startWorker = function() {

      if (this._worker) return;

      this._worker_ready = false;
      this._worker_jobs = 0; // counter how many requests send to worker

      var pthis = this;

      this._worker = new Worker(JSROOT.source_dir + "scripts/JSRootGeoWorker.js");

      this._worker.onmessage = function(e) {

         if (typeof e.data !== 'object') return;

         if ('log' in e.data)
            return JSROOT.console('geo: ' + e.data.log);

         if ('progress' in e.data)
            return JSROOT.progress(e.data.progress);

         e.data.tm3 = new Date().getTime();

         if ('init' in e.data) {
            pthis._worker_ready = true;
            return JSROOT.console('Worker ready: ' + (e.data.tm3 - e.data.tm0));
         }

         pthis.processWorkerReply(e.data);
      };

      // send initialization message with clones
      this._worker.postMessage( { init: true, browser: JSROOT.browser, tm0: new Date().getTime(), clones: this._clones.nodes, sortmap: this._clones.sortmap  } );
   }

   JSROOT.TGeoPainter.prototype.canSubmitToWorker = function(force) {
      if (!this._worker) return false;

      return this._worker_ready && ((this._worker_jobs == 0) || force);
   }

   JSROOT.TGeoPainter.prototype.submitToWorker = function(job) {
      if (!this._worker) return false;

      this._worker_jobs++;

      job.tm0 = new Date().getTime();

      this._worker.postMessage(job);
   }

   JSROOT.TGeoPainter.prototype.processWorkerReply = function(job) {
      this._worker_jobs--;

      if ('collect' in job) {
         this._new_draw_nodes = job.new_nodes;
         this._draw_all_nodes = job.complete;
         this.drawing_stage = 3;
         // invoke methods immediately
         return this.continueDraw();
      }

      if ('shapes' in job) {

         for (var n=0;n<job.shapes.length;++n) {
            var item = job.shapes[n],
                origin = this._build_shapes[n];

            // var shape = this._clones.GetNodeShape(item.nodeid);

            if (item.buf_pos && item.buf_norm) {
               if (item.buf_pos.length === 0) {
                  origin.geom = null;
               } else if (item.buf_pos.length !== item.buf_norm.length) {
                  console.error('item.buf_pos',item.buf_pos.length, 'item.buf_norm', item.buf_norm.length);
                  origin.geom = null;
               } else {
                  origin.geom = new THREE.BufferGeometry();

                  origin.geom.addAttribute( 'position', new THREE.BufferAttribute( item.buf_pos, 3 ) );
                  origin.geom.addAttribute( 'normal', new THREE.BufferAttribute( item.buf_norm, 3 ) );
               }

               origin.ready = true;
               origin.nfaces = item.nfaces;
            }
         }

         job.tm4 = new Date().getTime();

         // console.log('Get reply from worker', job.tm3-job.tm2, ' decode json in ', job.tm4-job.tm3);

         this.drawing_stage = 7; // first check which shapes are used, than build meshes

         // invoke methods immediately
         return this.continueDraw();
      }
   }

   JSROOT.TGeoPainter.prototype.testGeomChanges = function() {
      this.startDrawGeometry();
   }

   JSROOT.TGeoPainter.prototype.toggleAxisDraw = function(force_on) {
      if (this.TestAxisVisibility!==undefined) {
         if (force_on) return; // we want axis - we have axis
         this.TestAxisVisibility(null, this._toplevel);
      } else {
         var axis = JSROOT.Create("TNamed");
         axis._typename = "TAxis3D";
         axis._main = this;
         JSROOT.draw(this.divid, axis); // it will include drawing of
      }
   }

   JSROOT.TGeoPainter.prototype.completeDraw = function(close_progress) {

      var call_ready = false;

      if (!this.options) {
         console.warn('options object does not exist in completeDraw - something went wrong');
         return;
      }

      if (this._first_drawing) {
         this.adjustCameraPosition(true);
         this._first_drawing = false;
         call_ready = true;

         if (this._webgl) {
            this.enableX = this.options.clipx;
            this.enableY = this.options.clipy;
            this.enableZ = this.options.clipz;
            this.updateClipping(true); // only set clip panels, do not render
         }
      }

      if (this.options.transparancy!==1)
         this.changeGlobalTransparancy(this.options.transparancy, true);

      this.completeScene();

      if (this.options._axis) {
         this.options._axis = false;
         this.toggleAxisDraw();
      }

      this._scene.overrideMaterial = null;

      // if extra object where append, redraw them at the end
      this.getExtrasContainer("delete"); // delete old container
      this.drawExtras(this._extraObjects);

      this.Render3D(0, true);

      if (close_progress) JSROOT.progress();

      this.addOrbitControls();

      this.addTransformControl();

      this.showControlOptions(this.options.show_controls);

      if (call_ready) {

         // after first draw check if highlight can be enabled
         if (this.options.highlight === false)
            this.options.highlight = (this.first_render_tm < 1000);

         // if rotation was enabled, do it
         if (this.options.autoRotate) this.autorotate(2.5);

         this.DrawingReady();
      }

      if (this._draw_nodes_again)
         this.startDrawGeometry(); // relaunch drawing
   }

   JSROOT.TGeoPainter.prototype.Cleanup = function(first_time) {

      if (!first_time) {

         this.AccessTopPainter(false); // remove as pointer

         this.helpText();

         JSROOT.Painter.DisposeThreejsObject(this._scene);

         if (this._tcontrols)
            this._tcontrols.dispose();

         if (this._controls)
            this._controls.Cleanup();

         if (this._context_menu)
            this._renderer.domElement.removeEventListener( 'contextmenu', this._context_menu, false );

         if (this._datgui)
            this._datgui.destroy();

         var obj = this.GetObject();
         if (obj) delete obj._painter;

         if (this._worker) this._worker.terminate();

         JSROOT.TObjectPainter.prototype.Cleanup.call(this);

         delete this.options;

         delete this._animating;
      }

      if (this._renderer) {
         if (this._renderer.dispose) this._renderer.dispose();
         if (this._renderer.context) delete this._renderer.context;
      }

      delete this._scene;
      this._scene_width = 0;
      this._scene_height = 0;
      this._renderer = null;
      this._toplevel = null;
      this._camera = null;

      if (this._clones) this._clones.Cleanup(this._draw_nodes, this._build_shapes);
      delete this._clones;
      delete this._draw_nodes;
      delete this._build_shapes;
      delete this._new_draw_nodes;

      this.first_render_tm = 0;
      this.last_render_tm = 2000;

      this.drawing_stage = 0;

      delete this._datgui;
      delete this._controls;
      delete this._context_menu;
      delete this._tcontrols;
      delete this._toolbar;

      delete this._worker;
   }

   JSROOT.TGeoPainter.prototype.helpText = function(msg) {
      JSROOT.progress(msg);
   }

   JSROOT.TGeoPainter.prototype.CheckResize = function(size) {
      var pad_painter = this.pad_painter();

      // firefox is the only browser which correctly supports resize of embedded canvas,
      // for others we should force canvas redrawing at every step
      if (pad_painter)
         if (!pad_painter.CheckCanvasResize(size)) return false;

      var sz = this.size_for_3d();

      if ((this._scene_width === sz.width) && (this._scene_height === sz.height)) return false;
      if ((sz.width<10) || (sz.height<10)) return;

      this._scene_width = sz.width;
      this._scene_height = sz.height;

      this._camera.aspect = this._scene_width / this._scene_height;
      this._camera.updateProjectionMatrix();

      this._renderer.setSize( this._scene_width, this._scene_height );

      this.Render3D();

      return true;
   }

   JSROOT.TGeoPainter.prototype.ownedByTransformControls = function(child) {
      var obj = child.parent;
      while (obj && !(obj instanceof THREE.TransformControls) ) {
         obj = obj.parent;
      }
      return (obj && (obj instanceof THREE.TransformControls));
   }

   JSROOT.TGeoPainter.prototype.accessObjectWireFrame = function(obj, on) {
      // either change mesh wireframe or return current value
      // return undefined when wireframe cannot be accessed

      if (!obj.hasOwnProperty("material") || (obj instanceof THREE.GridHelper)) return;

      if (this.ownedByTransformControls(obj)) return;

      if ((on !== undefined) && obj.stack)
         obj.material.wireframe = on;

      return obj.material.wireframe;
   }


   JSROOT.TGeoPainter.prototype.changeWireFrame = function(obj, on) {
      var painter = this;

      obj.traverse(function(obj2) { painter.accessObjectWireFrame(obj2, on); });

      this.Render3D();
   }

   JSROOT.Painter.drawGeoObject = function(divid, obj, opt) {
      if (obj === null) return null;

      JSROOT.GEO.GradPerSegm = JSROOT.gStyle.GeoGradPerSegm;
      JSROOT.GEO.CompressComp = JSROOT.gStyle.GeoCompressComp;

      var shape = null, is_manager = false;

      if (('fShapeBits' in obj) && ('fShapeId' in obj)) {
         shape = obj; obj = null;
      } else
      if ((obj._typename === 'TGeoVolumeAssembly') || (obj._typename === 'TGeoVolume')) {
         shape = obj.fShape;
      } else
      if (obj._typename === "TEveGeoShapeExtract") {
         shape = obj.fShape;
      } else
      if (obj._typename === 'TGeoManager') {
         obj = obj.fMasterVolume;
         JSROOT.GEO.SetBit(obj, JSROOT.GEO.BITS.kVisThis, false);
         shape = obj.fShape;
         is_manager = true;
      } else
      if ('fVolume' in obj) {
         if (obj.fVolume) shape = obj.fVolume.fShape;
      } else {
         obj = null;
      }

      if (opt && opt.indexOf("comp")==0 && shape && (shape._typename == 'TGeoCompositeShape') && shape.fNode) {
         opt = opt.substr(4);
         obj = JSROOT.GEO.buildCompositeVolume(shape);
      }

      if (!obj && shape)
         obj = JSROOT.extend(JSROOT.Create("TEveGeoShapeExtract"),
                   { fTrans: null, fShape: shape, fRGBA: [ 0, 1, 0, 1], fElements: null, fRnrSelf: true });

      if (obj) {
         JSROOT.extend(this, new JSROOT.TGeoPainter(obj,is_manager));
         this.SetDivId(divid, 5);
         return this.DrawGeometry(opt, divid);
      }

      return this.DrawingReady();
   }

   /// keep for backwards compatibility
   JSROOT.Painter.drawGeometry = JSROOT.Painter.drawGeoObject;

   // ===================================================================================

   JSROOT.Painter.drawAxis3D = function(divid, axis, opt) {

      var painter = new JSROOT.TObjectPainter(axis);

      if (!('_main' in axis))
         painter.SetDivId(divid);

      painter.Draw3DAxis = function() {
         var main = this.main_painter();

         if ((main === null) && ('_main' in this.GetObject()))
            main = this.GetObject()._main; // simple workaround to get geo painter

         if ((main === null) || (main._toplevel === undefined))
            return console.warn('no geo object found for 3D axis drawing');

         var box = new THREE.Box3().setFromObject(main._toplevel);

         this.xmin = box.min.x; this.xmax = box.max.x;
         this.ymin = box.min.y; this.ymax = box.max.y;
         this.zmin = box.min.z; this.zmax = box.max.z;

         // use min/max values directly as graphical coordinates
         this.size_xy3d = this.size_z3d =  0;

         this.DrawXYZ = JSROOT.Painter.HPainter_DrawXYZ;

         this.DrawXYZ(main._toplevel);

         main.adjustCameraPosition();

         main.TestAxisVisibility = JSROOT.Painter.HPainter_TestAxisVisibility;

         main.Render3D();
      }

      painter.Draw3DAxis();

      return painter.DrawingReady();
   }

   // ===============================================================================

   JSROOT.GEO.buildCompositeVolume = function(comp, side) {
      // function used to build hierarchy of elements of composite shapes

      var vol = JSROOT.Create("TGeoVolume");
      if (side && (comp._typename!=='TGeoCompositeShape')) {
         vol.fName = side;
         JSROOT.GEO.SetBit(vol, JSROOT.GEO.BITS.kVisThis, true);
         vol.fLineColor = (side=="Left"? 2 : 3);
         vol.fShape = comp;
         return vol;
      }

      JSROOT.GEO.SetBit(vol, JSROOT.GEO.BITS.kVisDaughters, true);
      vol.$geoh = true; // workaround, let know browser that we are in volumes hierarchy
      vol.fName = "";

      var node1 = JSROOT.Create("TGeoNodeMatrix");
      node1.fName = "Left";
      node1.fMatrix = comp.fNode.fLeftMat;
      node1.fVolume = JSROOT.GEO.buildCompositeVolume(comp.fNode.fLeft, "Left");

      var node2 = JSROOT.Create("TGeoNodeMatrix");
      node2.fName = "Right";
      node2.fMatrix = comp.fNode.fRightMat;
      node2.fVolume = JSROOT.GEO.buildCompositeVolume(comp.fNode.fRight, "Right");

      vol.fNodes = JSROOT.Create("TList");
      vol.fNodes.Add(node1);
      vol.fNodes.Add(node2);

      return vol;
   }

   JSROOT.GEO.provideVisStyle = function(obj) {
      if (obj._typename === 'TEveGeoShapeExtract')
         return obj.fRnrSelf ? " geovis_this" : "";

      var vis = !JSROOT.GEO.TestBit(obj, JSROOT.GEO.BITS.kVisNone) &&
                JSROOT.GEO.TestBit(obj, JSROOT.GEO.BITS.kVisThis);

      var chld = JSROOT.GEO.TestBit(obj, JSROOT.GEO.BITS.kVisDaughters) ||
                 JSROOT.GEO.TestBit(obj, JSROOT.GEO.BITS.kVisOneLevel);

      if (chld && (!obj.fNodes || (obj.fNodes.arr.length === 0))) chld = false;

      if (vis && chld) return " geovis_all";
      if (vis) return " geovis_this";
      if (chld) return " geovis_daughters";
      return "";
   }


   JSROOT.GEO.getBrowserItem = function(item, itemname, callback) {
      // mark object as belong to the hierarchy, require to
      if (item._geoobj) item._geoobj.$geoh = true;

      JSROOT.CallBack(callback, item, item._geoobj);
   }


   JSROOT.GEO.createItem = function(node, obj, name) {
      var sub = {
         _kind: "ROOT." + obj._typename,
         _name: name ? name : obj.fName,
         _title: obj.fTitle,
         _parent: node,
         _geoobj: obj,
         _get: JSROOT.GEO.getBrowserItem
      };

      var volume, shape, subnodes, iseve = false;

      if (obj._typename == "TGeoMaterial") sub._icon = "img_geomaterial"; else
      if (obj._typename == "TGeoMedium") sub._icon = "img_geomedium"; else
      if (obj._typename == "TGeoMixture") sub._icon = "img_geomixture"; else
      if ((obj._typename.indexOf("TGeoNode")===0) && obj.fVolume) {
         sub._title = "node:"  + obj._typename;
         if (obj.fTitle.length > 0) sub._title += " " + obj.fTitle;
         volume = obj.fVolume;
      } else
      if (obj._typename.indexOf("TGeoVolume")===0) {
         volume = obj;
      } else
      if (obj._typename == "TEveGeoShapeExtract") {
         iseve = true;
         shape = obj.fShape;
         subnodes = obj.fElements ? obj.fElements.arr : null;
      } else
      if ((obj.fShapeBits !== undefined) && (obj.fShapeId !== undefined)) {
         shape = obj;
      }

      if (volume) {
         shape = volume.fShape;
         subnodes = volume.fNodes ? volume.fNodes.arr : null;
      }

      if (volume || shape || subnodes) {
         if (volume) sub._volume = volume;

         if (subnodes) {
            sub._more = true;
            sub._expand = JSROOT.GEO.expandObject;
         } else
         if (shape && (shape._typename === "TGeoCompositeShape") && shape.fNode) {
            sub._more = true;
            sub._shape = shape;
            sub._expand = function(node, obj) {
               JSROOT.GEO.createItem(node, node._shape.fNode.fLeft, 'Left');
               JSROOT.GEO.createItem(node, node._shape.fNode.fRight, 'Right');
               return true;
            }
         }

         if (!sub._title && (obj._typename != "TGeoVolume")) sub._title = obj._typename;

         if (shape) {
            if (sub._title == "")
               sub._title = shape._typename;

            sub._icon = JSROOT.GEO.getShapeIcon(shape);
         } else {
            sub._icon = sub._more ? "img_geocombi" : "img_geobbox";
         }

         if (volume)
            sub._icon += JSROOT.GEO.provideVisStyle(volume);
         else if (iseve)
            sub._icon += JSROOT.GEO.provideVisStyle(obj);

         sub._menu = JSROOT.GEO.provideMenu;
         sub._icon_click  = JSROOT.GEO.browserIconClick;
      }

      if (!node._childs) node._childs = [];
      node._childs.push(sub);

      return sub;
   }

   JSROOT.GEO.createList = function(parent, lst, name, title) {

      if ((lst==null) || !('arr' in lst) || (lst.arr.length==0)) return;

      var item = {
          _name: name,
          _kind: "ROOT.TList",
          _title: title,
          _more: true,
          _geoobj: lst,
          _parent: parent,
      }

      item._get = function(item, itemname, callback) {
         if ('_geoobj' in item)
            return JSROOT.CallBack(callback, item, item._geoobj);

         JSROOT.CallBack(callback, item, null);
      }

      item._expand = function(node, lst) {
         // only childs

         if ('fVolume' in lst)
            lst = lst.fVolume.fNodes;

         if (!('arr' in lst)) return false;

         node._childs = [];

         for (var n in lst.arr)
            JSROOT.GEO.createItem(node, lst.arr[n]);

         return true;
      }

      if (!parent._childs) parent._childs = [];
      parent._childs.push(item);

   };

   JSROOT.GEO.provideMenu = function(menu, item, hpainter) {

      if (!item._geoobj) return false;

      var obj = item._geoobj, vol = item._volume,
          iseve = (obj._typename === 'TEveGeoShapeExtract');

      if (!vol && !iseve) return false;

      menu.add("separator");

      function ScanEveVisible(obj, arg, skip_this) {
         if (!arg) arg = { visible: 0, hidden: 0 };

         if (!skip_this) {
            if (arg.assign!==undefined) obj.fRnrSelf = arg.assign; else
            if (obj.fRnrSelf) arg.vis++; else arg.hidden++;
         }

         if (obj.fElements)
            for (var n=0;n<obj.fElements.arr.length;++n)
               ScanEveVisible(obj.fElements.arr[n], arg, false);

         return arg;
      }

      function ToggleEveVisibility(arg) {
         if (arg === 'self') {
            obj.fRnrSelf = !obj.fRnrSelf;
            item._icon = item._icon.split(" ")[0] + JSROOT.GEO.provideVisStyle(obj);
            hpainter.UpdateTreeNode(item);
         } else {
            ScanEveVisible(obj, { assign: (arg === "true") }, true);
            hpainter.ForEach(function(m) {
               // update all child items
               if (m._geoobj && m._icon) {
                  m._icon = item._icon.split(" ")[0] + JSROOT.GEO.provideVisStyle(m._geoobj);
                  hpainter.UpdateTreeNode(m);
               }
            }, item);
         }

         JSROOT.GEO.findItemWithPainter(item, 'testGeomChanges');
      }

      function ToggleMenuBit(arg) {
         JSROOT.GEO.ToggleBit(vol, arg);
         var newname = item._icon.split(" ")[0] + JSROOT.GEO.provideVisStyle(vol);
         hpainter.ForEach(function(m) {
            // update all items with that volume
            if (item._volume === m._volume) {
               m._icon = newname;
               hpainter.UpdateTreeNode(m);
            }
         });

         hpainter.UpdateTreeNode(item);
         JSROOT.GEO.findItemWithPainter(item, 'testGeomChanges');
      }

      if ((item._geoobj._typename.indexOf("TGeoNode")===0) && JSROOT.GEO.findItemWithPainter(item))
         menu.add("Focus", function() {

           var drawitem = JSROOT.GEO.findItemWithPainter(item);

           if (!drawitem) return;

           var fullname = hpainter.itemFullName(item, drawitem);

           if (drawitem._painter && typeof drawitem._painter.focusOnItem == 'function')
              drawitem._painter.focusOnItem(fullname);
         });

      if (iseve) {
         menu.addchk(obj.fRnrSelf, "Visible", "self", ToggleEveVisibility);
         var res = ScanEveVisible(obj, undefined, true);

         if (res.hidden + res.visible > 0)
            menu.addchk((res.hidden==0), "Daughters", (res.hidden!=0) ? "true" : "false", ToggleEveVisibility);

      } else {
         menu.addchk(JSROOT.GEO.TestBit(vol, JSROOT.GEO.BITS.kVisNone), "Invisible",
               JSROOT.GEO.BITS.kVisNone, ToggleMenuBit);
         menu.addchk(JSROOT.GEO.TestBit(vol, JSROOT.GEO.BITS.kVisThis), "Visible",
               JSROOT.GEO.BITS.kVisThis, ToggleMenuBit);
         menu.addchk(JSROOT.GEO.TestBit(vol, JSROOT.GEO.BITS.kVisDaughters), "Daughters",
               JSROOT.GEO.BITS.kVisDaughters, ToggleMenuBit);
         menu.addchk(JSROOT.GEO.TestBit(vol, JSROOT.GEO.BITS.kVisOneLevel), "1lvl daughters",
               JSROOT.GEO.BITS.kVisOneLevel, ToggleMenuBit);
      }

      return true;
   }

   JSROOT.GEO.findItemWithPainter = function(hitem, funcname) {
      while (hitem) {
         if (hitem._painter && hitem._painter._camera) {
            if (funcname && typeof hitem._painter[funcname] == 'function')
               hitem._painter[funcname]();
            return hitem;
         }
         hitem = hitem._parent;
      }
      return null;
   }

   JSROOT.GEO.updateBrowserIcons = function(obj, hpainter) {
      if (!obj || !hpainter) return;

      hpainter.ForEach(function(m) {
         // update all items with that volume
         if ((obj === m._volume) || (obj === m._geoobj)) {
            m._icon = m._icon.split(" ")[0] + JSROOT.GEO.provideVisStyle(obj);
            hpainter.UpdateTreeNode(m);
         }
      });
   }

   JSROOT.GEO.browserIconClick = function(hitem, hpainter) {
      if (hitem._volume) {
         if (hitem._more && hitem._volume.fNodes && (hitem._volume.fNodes.arr.length>0))
            JSROOT.GEO.ToggleBit(hitem._volume, JSROOT.GEO.BITS.kVisDaughters);
         else
            JSROOT.GEO.ToggleBit(hitem._volume, JSROOT.GEO.BITS.kVisThis);

         JSROOT.GEO.updateBrowserIcons(hitem._volume, hpainter);

         JSROOT.GEO.findItemWithPainter(hitem, 'testGeomChanges');
         return false; // no need to update icon - we did it ourself
      }

      if (hitem._geoobj && hitem._geoobj._typename == "TEveGeoShapeExtract") {
         hitem._geoobj.fRnrSelf = !hitem._geoobj.fRnrSelf;

         JSROOT.GEO.updateBrowserIcons(hitem._geoobj, hpainter);
         JSROOT.GEO.findItemWithPainter(hitem, 'testGeomChanges');
         return false; // no need to update icon - we did it ourself
      }


      // first check that geo painter assigned with the item
      var drawitem = JSROOT.GEO.findItemWithPainter(hitem);
      if (!drawitem) return false;

      var newstate = drawitem._painter.ExtraObjectVisible(hpainter.itemFullName(hitem), true);

      // return true means browser should update icon for the item
      return (newstate!==undefined) ? true : false;
   }

   JSROOT.GEO.getShapeIcon = function(shape) {
      switch (shape._typename) {
         case "TGeoArb8" : return "img_geoarb8"; break;
         case "TGeoCone" : return "img_geocone"; break;
         case "TGeoConeSeg" : return "img_geoconeseg"; break;
         case "TGeoCompositeShape" : return "img_geocomposite"; break;
         case "TGeoTube" : return "img_geotube"; break;
         case "TGeoTubeSeg" : return "img_geotubeseg"; break;
         case "TGeoPara" : return "img_geopara"; break;
         case "TGeoParaboloid" : return "img_geoparab"; break;
         case "TGeoPcon" : return "img_geopcon"; break;
         case "TGeoPgon" : return "img_geopgon"; break;
         case "TGeoShapeAssembly" : return "img_geoassembly"; break;
         case "TGeoSphere" : return "img_geosphere"; break;
         case "TGeoTorus" : return "img_geotorus"; break;
         case "TGeoTrd1" : return "img_geotrd1"; break;
         case "TGeoTrd2" : return "img_geotrd2"; break;
         case "TGeoXtru" : return "img_geoxtru"; break;
         case "TGeoTrap" : return "img_geotrap"; break;
         case "TGeoGtra" : return "img_geogtra"; break;
         case "TGeoEltu" : return "img_geoeltu"; break;
         case "TGeoHype" : return "img_geohype"; break;
         case "TGeoCtub" : return "img_geoctub"; break;
      }
      return "img_geotube";
   }

   JSROOT.GEO.getBrowserIcon = function(hitem, hpainter) {
      var icon = "";
      if (hitem._kind == 'ROOT.TEveTrack') icon = 'img_evetrack'; else
      if (hitem._kind == 'ROOT.TEvePointSet') icon = 'img_evepoints';
      if (icon.length>0) {
         var drawitem = JSROOT.GEO.findItemWithPainter(hitem);
         if (drawitem)
            if (drawitem._painter.ExtraObjectVisible(hpainter.itemFullName(hitem)))
               icon += " geovis_this";
      }
      return icon;
   }

   JSROOT.GEO.expandObject = function(parent, obj) {
      if (!parent || !obj) return false;

      var isnode = (obj._typename.indexOf('TGeoNode') === 0),
          isvolume = (obj._typename.indexOf('TGeoVolume') === 0),
          ismanager = (obj._typename === 'TGeoManager'),
          iseve = (obj._typename === 'TEveGeoShapeExtract');

      if (!isnode && !isvolume && !ismanager && !iseve) return false;

      if (parent._childs) return true;

      if (ismanager) {
         JSROOT.GEO.createList(parent, obj.fMaterials, "Materials", "list of materials");
         JSROOT.GEO.createList(parent, obj.fMedia, "Media", "list of media");
         JSROOT.GEO.createList(parent, obj.fTracks, "fTracks", "list of tracks");

         JSROOT.GEO.SetBit(obj.fMasterVolume, JSROOT.GEO.BITS.kVisThis, false);
         JSROOT.GEO.createItem(parent, obj.fMasterVolume);
         return true;
      }

      var volume, subnodes, shape;

      if (iseve) {
         subnodes = obj.fElements ? obj.fElements.arr : null;
         shape = obj.fShape;
      } else {
         volume = (isnode ? obj.fVolume : obj);
         subnodes = volume && volume.fNodes ? volume.fNodes.arr : null;
         shape = volume ? volume.fShape : null;
      }

      if (!subnodes && shape && (shape._typename === "TGeoCompositeShape") && shape.fNode) {
         if (!parent._childs) {
            JSROOT.GEO.createItem(parent, shape.fNode.fLeft, 'Left');
            JSROOT.GEO.createItem(parent, shape.fNode.fRight, 'Right');
         }

         return true;
      }

      if (!subnodes) return false;

      for (var i=0;i<subnodes.length;++i)
         JSROOT.GEO.createItem(parent, subnodes[i]);

      return true;
   }

   JSROOT.addDrawFunc({ name: "TGeoVolumeAssembly", icon: 'img_geoassembly', func: JSROOT.Painter.drawGeoObject, expand: JSROOT.GEO.expandObject, opt: ";more;all;count" });
   JSROOT.addDrawFunc({ name: "TAxis3D", func: JSROOT.Painter.drawAxis3D });
   JSROOT.addDrawFunc({ name: "TEvePointSet", icon_get: JSROOT.GEO.getBrowserIcon, icon_click: JSROOT.GEO.browserIconClick });
   JSROOT.addDrawFunc({ name: "TEveTrack", icon_get: JSROOT.GEO.getBrowserIcon, icon_click: JSROOT.GEO.browserIconClick });

   return JSROOT.Painter;

}));
