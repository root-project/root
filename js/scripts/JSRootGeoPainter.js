/** JavaScript ROOT 3D geometry painter
 * @file JSRootGeoPainter.js */

(function( factory ) {
   if ( typeof define === "function" && define.amd ) {
      define( [ 'JSRootPainter', 'd3', 'threejs', 'dat.gui', 'JSRoot3DPainter', 'JSRootGeoBase' ], factory );
   } else
   if (typeof exports === 'object' && typeof module !== 'undefined') {
      var jsroot = require("./JSRootCore.js");
      if (!jsroot.nodejs && (typeof window != 'undefined')) require("./dat.gui.min.js");
      factory(jsroot, require("d3"), require("three"), require("./JSRoot3DPainter.js"), require("./JSRootGeoBase.js"),
              jsroot.nodejs || (typeof document=='undefined') ? jsroot.nodejs_document : document);
   } else {

      if (typeof JSROOT == 'undefined')
         throw new Error('JSROOT is not defined', 'JSRootGeoPainter.js');

      if (typeof JSROOT.Painter != 'object')
         throw new Error('JSROOT.Painter is not defined', 'JSRootGeoPainter.js');

      if (typeof d3 == 'undefined')
         throw new Error('d3 is not defined', 'JSRootGeoPainter.js');

      if (typeof THREE == 'undefined')
         throw new Error('THREE is not defined', 'JSRootGeoPainter.js');

      if (typeof dat == 'undefined')
         throw new Error('dat.gui is not defined', 'JSRootGeoPainter.js');

      factory( JSROOT, d3, THREE );
   }
} (function( JSROOT, d3, THREE, _3d, _geo, document ) {

   "use strict";

   JSROOT.sources.push("geom");

   if ((typeof document=='undefined') && (typeof window=='object')) document = window.document;

   if ( typeof define === "function" && define.amd )
      JSROOT.loadScript('$$$style/JSRootGeoPainter.css');

   if (typeof JSROOT.GEO !== 'object')
      console.error('JSROOT.GEO namespace is not defined')

   // ============================================================================================

   function Toolbar(container, buttons, bright) {
      this.bright = bright;
      if ((container !== undefined) && (typeof container.append == 'function'))  {
         this.element = container.append("div").attr('class','jsroot');
         this.addButtons(buttons);
      }
   }

   Toolbar.prototype.addButtons = function(buttons) {
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

   Toolbar.prototype.createButton = function(group, config) {

      var title = config.title;
      if (title === undefined) title = config.name;

      if (typeof config.click !== 'function')
         throw new Error('must provide button \'click\' function in button config');

      var button = group.append('a')
                        .attr('class', this.bright ? 'toolbar-btn-bright' : 'toolbar-btn')
                        .attr('rel', 'tooltip')
                        .attr('data-title', title)
                        .on('click', config.click);

      this.createIcon(button, config.icon || JSROOT.ToolbarIcons.question);
   }


   Toolbar.prototype.changeBrightness = function(bright) {
      this.bright = bright;
      if (!this.element) return;

      this.element.selectAll(bright ? '.toolbar-btn' : ".toolbar-btn-bright")
                  .attr("class", !bright ? 'toolbar-btn' : "toolbar-btn-bright");
   }


   Toolbar.prototype.createIcon = function(button, thisIcon) {
      var dimensions = thisIcon.size ? thisIcon.size.split(' ') : [512, 512],
          width = dimensions[0],
          height = dimensions[1] || dimensions[0],
          scale = thisIcon.scale || 1,
          svg = button.append("svg:svg")
                      .attr('height', '1em')
                      .attr('width', '1em')
                      .attr('viewBox', [0, 0, width, height].join(' '));

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

   Toolbar.prototype.removeAllButtons = function() {
      this.element.remove();
   };

   /**
    * @class TGeoPainter
    * @desc Holder of different functions and classes for drawing geometries
    * @memberof JSROOT
    */

   function TGeoPainter(obj) {

      if (obj && (obj._typename === "TGeoManager")) {
         this.geo_manager = obj;
         obj = obj.fMasterVolume;
      }

      if (obj && (obj._typename.indexOf('TGeoVolume') === 0))
         obj = { _typename:"TGeoNode", fVolume: obj, fName: obj.fName, $geoh: obj.$geoh, _proxy: true };

      JSROOT.TObjectPainter.call(this, obj);

      this.no_default_title = true; // do not set title to main DIV
      this.mode3d = true; // indication of 3D mode
      this.drawing_stage = 0; //

      this.Cleanup(true);
   }

   TGeoPainter.prototype = Object.create( JSROOT.TObjectPainter.prototype );

   TGeoPainter.prototype.CreateToolbar = function(args) {
      if (this._toolbar || this._usesvg || this._usesvgimg) return;
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

      buttonList.push({
         name: 'control',
         title: 'Toggle control UI',
         icon: JSROOT.ToolbarIcons.rect,
         click: function() { painter.showControlOptions('toggle'); }
      });

      buttonList.push({
         name: 'enlarge',
         title: 'Enlarge geometry drawing',
         icon: JSROOT.ToolbarIcons.circle,
         click: function() { painter.ToggleEnlarge(); }
      });

      // Only show VR icon if WebVR API available.
      if (navigator.getVRDisplays) {
         buttonList.push({
            name: 'entervr',
            title: 'Enter VR (It requires a VR Headset connected)',
            icon: JSROOT.ToolbarIcons.vrgoggles,
            click: function() { painter.ToggleVRMode(); }
         });
         this.InitVRMode();
      }

      if (JSROOT.gStyle.ContextMenu)
      buttonList.push({
         name: 'menu',
         title: 'Show context menu',
         icon: JSROOT.ToolbarIcons.question,
         click: function() {

            d3.event.preventDefault();
            d3.event.stopPropagation();

            var evnt = d3.event;

            if (!JSROOT.Painter.closeMenu())
               JSROOT.Painter.createMenu(painter, function(menu) {
                  menu.painter.FillContextMenu(menu);
                  menu.show(evnt);
               });
         }
      });

      var bkgr = new THREE.Color(this.options.background);

      this._toolbar = new Toolbar( this.select_main(), [buttonList], (bkgr.r + bkgr.g + bkgr.b) < 1);
   }

   TGeoPainter.prototype.InitVRMode = function() {
      var pthis = this;
      // Dolly contains camera and controllers in VR Mode
      // Allows moving the user in the scene
      this._dolly = new THREE.Group();
      this._scene.add(this._dolly);
      this._standingMatrix = new THREE.Matrix4();

      // Raycaster temp variables to avoid one per frame allocation.
      this._raycasterEnd = new THREE.Vector3();
      this._raycasterOrigin = new THREE.Vector3();

      navigator.getVRDisplays().then(function (displays) {
         var vrDisplay = displays[0];
         if (!vrDisplay) return;
         pthis._renderer.vr.setDevice(vrDisplay);
         pthis._vrDisplay = vrDisplay;
         pthis._standingMatrix.fromArray(vrDisplay.stageParameters.sittingToStandingTransform);
         pthis.InitVRControllersGeometry();
      });
   }

   TGeoPainter.prototype.InitVRControllersGeometry = function() {

      let geometry = new THREE.SphereGeometry(0.025, 18, 36);
      let material = new THREE.MeshBasicMaterial({color: 'grey'});
      let rayMaterial = new THREE.MeshBasicMaterial({color: 'fuchsia'});
      let rayGeometry = new THREE.BoxBufferGeometry(0.001, 0.001, 2);
      let ray1Mesh = new THREE.Mesh(rayGeometry, rayMaterial);
      let ray2Mesh = new THREE.Mesh(rayGeometry, rayMaterial);
      let sphere1 = new THREE.Mesh(geometry, material);
      let sphere2 = new THREE.Mesh(geometry, material);

      this._controllersMeshes = [];
      this._controllersMeshes.push(sphere1);
      this._controllersMeshes.push(sphere2);
      ray1Mesh.position.z -= 1;
      ray2Mesh.position.z -= 1;
      sphere1.add(ray1Mesh);
      sphere2.add(ray2Mesh);
      this._dolly.add(sphere1);
      this._dolly.add(sphere2);
      // Controller mesh hidden by default
      sphere1.visible = false;
      sphere2.visible = false;
   }

   TGeoPainter.prototype.UpdateVRControllersList = function() {
      var gamepads = navigator.getGamepads && navigator.getGamepads();
      // Has controller list changed?
      if (this.vrControllers && (gamepads.length === this.vrControllers.length)) { return; }
      // Hide meshes.
      this._controllersMeshes.forEach(function (mesh) { mesh.visible = false; });
      this._vrControllers = [];
      for (var i = 0; i < gamepads.length; ++i) {
         if (!gamepads[i].pose) { continue; }
         this._vrControllers.push({
            gamepad: gamepads[i],
            mesh: this._controllersMeshes[i]
         });
         this._controllersMeshes[i].visible = true;
      }
   }

   TGeoPainter.prototype.ProcessVRControllerIntersections = function() {
      var intersects = []
      for (var i = 0; i < this._vrControllers.length; ++i) {
         let controller = this._vrControllers[i].mesh;
         let end = controller.localToWorld(this._raycasterEnd.set(0, 0, -1));
         let origin = controller.localToWorld(this._raycasterOrigin.set(0, 0, 0));
         end.sub(origin).normalize();
         intersects = intersects.concat(this._controls.GetOriginDirectionIntersects(origin, end));
      }
      // Remove duplicates.
      intersects = intersects.filter(function (item, pos) {return intersects.indexOf(item) === pos});
      this._controls.ProcessMouseMove(intersects);
   }

   TGeoPainter.prototype.UpdateVRControllers = function() {
      this.UpdateVRControllersList();
      // Update pose.
      for (var i = 0; i < this._vrControllers.length; ++i) {
         let controller = this._vrControllers[i];
         let orientation = controller.gamepad.pose.orientation;
         let position = controller.gamepad.pose.position;
         let controllerMesh = controller.mesh;
         if (orientation) { controllerMesh.quaternion.fromArray(orientation); }
         if (position) { controllerMesh.position.fromArray(position); }
         controllerMesh.updateMatrix();
         controllerMesh.applyMatrix(this._standingMatrix);
         controllerMesh.matrixWorldNeedsUpdate = true;
      }
      this.ProcessVRControllerIntersections();
   }

   TGeoPainter.prototype.ToggleVRMode = function() {
      var pthis = this;
      if (!this._vrDisplay) return;
      // Toggle VR mode off
      if (this._vrDisplay.isPresenting) {
         this.ExitVRMode();
         return;
      }
      this._previousCameraPosition = this._camera.position.clone();
      this._previousCameraRotation = this._camera.rotation.clone();
      this._vrDisplay.requestPresent([{ source: this._renderer.domElement }]).then(function() {
         pthis._previousCameraNear = pthis._camera.near;
         pthis._dolly.add(pthis._camera);
         pthis._camera.near = 0.1;
         pthis._camera.updateProjectionMatrix();
         pthis._renderer.vr.enabled = true;
         pthis._renderer.setAnimationLoop(function () {
            pthis.UpdateVRControllers();
            pthis.Render3D(0);
         });
      });
      this._renderer.vr.enabled = true;

      window.addEventListener( 'keydown', function ( event ) {
         // Esc Key turns VR mode off
         if (event.keyCode === 27) pthis.ExitVRMode();
      });
   }

   TGeoPainter.prototype.ExitVRMode = function() {
      var pthis = this;
      if (!this._vrDisplay.isPresenting) return;
      this._renderer.vr.enabled = false;
      this._dolly.remove(this._camera);
      this._scene.add(this._camera);
      // Restore Camera pose
      this._camera.position.copy(this._previousCameraPosition);
      this._previousCameraPosition = undefined;
      this._camera.rotation.copy(this._previousCameraRotation);
      this._previousCameraRotation = undefined;
      this._camera.near = this._previousCameraNear;
      this._camera.updateProjectionMatrix();
      this._vrDisplay.exitPresent();
   }

   TGeoPainter.prototype.GetGeometry = function() {
      return this.GetObject();
   }

   TGeoPainter.prototype.ModifyVisisbility = function(name, sign) {
      if (JSROOT.GEO.NodeKind(this.GetGeometry()) !== 0) return;

      if (name == "")
         return JSROOT.GEO.SetBit(this.GetGeometry().fVolume, JSROOT.GEO.BITS.kVisThis, (sign === "+"));

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

   TGeoPainter.prototype.decodeOptions = function(opt) {
      if (typeof opt != "string") opt = "";

      var res = { _grid: false, _bound: false, _debug: false,
                  _full: false, _axis: false, _axis_center: false,
                  _count: false, wireframe: false,
                   scale: new THREE.Vector3(1,1,1), zoom: 1.0,
                   more: 1, maxlimit: 100000, maxnodeslimit: 3000,
                   use_worker: false, update_browser: true, show_controls: false,
                   highlight: false, highlight_scene: false, select_in_view: false,
                   project: '', is_main: false, tracks: false, ortho_camera: false,
                   clipx: false, clipy: false, clipz: false, ssao: false,
                   script_name: "", transparency: 0, autoRotate: false, background: '#FFFFFF',
                   depthMethod: "ray" };

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

      var d = new JSROOT.DrawOptions(opt);

      if (d.check("MAIN")) res.is_main = true;

      if (d.check("TRACKS")) res.tracks = true;
      if (d.check("ORTHO_CAMERA")) res.ortho_camera = true;

      if (d.check("DEPTHRAY") || d.check("DRAY")) res.depthMethod = "ray";
      if (d.check("DEPTHBOX") || d.check("DBOX")) res.depthMethod = "box";
      if (d.check("DEPTHPNT") || d.check("DPNT")) res.depthMethod = "pnt";
      if (d.check("DEPTHSIZE") || d.check("DSIZE")) res.depthMethod = "size";
      if (d.check("DEPTHDFLT") || d.check("DDFLT")) res.depthMethod = "dflt";

      if (d.check("ZOOM", true)) res.zoom = d.partAsInt(0, 100) / 100;

      if (d.check('BLACK')) res.background = "#000000";
      if (d.check('WHITE')) res.background = "#FFFFFF";

      if (d.check('BKGR_', true)) {
         var bckgr = null;
         if (d.partAsInt(1)>0) bckgr = JSROOT.Painter.root_colors[d.partAsInt()]; else
         for (var col=0;col<8;++col)
            if (JSROOT.Painter.root_colors[col].toUpperCase() === d.part) bckgr = JSROOT.Painter.root_colors[col];
         if (bckgr) res.background = "#" + new THREE.Color(bckgr).getHexString();
      }

      if (d.check("MORE3")) res.more = 3;
      if (d.check("MORE")) res.more = 2;
      if (d.check("ALL")) res.more = 100;

      if (d.check("CONTROLS") || d.check("CTRL")) res.show_controls = true;

      if (d.check("CLIPXYZ")) res.clipx = res.clipy = res.clipz = true;
      if (d.check("CLIPX")) res.clipx = true;
      if (d.check("CLIPY")) res.clipy = true;
      if (d.check("CLIPZ")) res.clipz = true;
      if (d.check("CLIP")) res.clipx = res.clipy = res.clipz = true;

      if (d.check("PROJX", true)) { res.project = 'x'; if (d.partAsInt(1)>0) res.projectPos = d.partAsInt(); }
      if (d.check("PROJY", true)) { res.project = 'y'; if (d.partAsInt(1)>0) res.projectPos = d.partAsInt(); }
      if (d.check("PROJZ", true)) { res.project = 'z'; if (d.partAsInt(1)>0) res.projectPos = d.partAsInt(); }

      if (d.check("DFLT_COLORS") || d.check("DFLT")) this.SetRootDefaultColors();
      if (d.check("SSAO")) res.ssao = true;

      if (d.check("NOWORKER")) res.use_worker = -1;
      if (d.check("WORKER")) res.use_worker = 1;

      if (d.check("NOHIGHLIGHT") || d.check("NOHIGH")) res.highlight = res.highlight_scene = 0;
      if (d.check("HIGHLIGHT")) res.highlight_scene = res.highlight = true;
      if (d.check("HSCENEONLY")) { res.highlight_scene = true; res.highlight = 0; }
      if (d.check("NOHSCENE")) res.highlight_scene = 0;
      if (d.check("HSCENE")) res.highlight_scene = true;

      if (d.check("WIRE")) res.wireframe = true;
      if (d.check("ROTATE")) res.autoRotate = true;

      if (d.check("INVX") || d.check("INVERTX")) res.scale.x = -1;
      if (d.check("INVY") || d.check("INVERTY")) res.scale.y = -1;
      if (d.check("INVZ") || d.check("INVERTZ")) res.scale.z = -1;

      if (d.check("COUNT")) res._count = true;

      if (d.check('TRANSP',true))
         res.transparency = d.partAsInt(0,100)/100;

      if (d.check('OPACITY',true))
         res.transparency = 1 - d.partAsInt(0,100)/100;

      if (d.check("AXISCENTER") || d.check("AC")) { res._axis = true; res._axis_center = true; }

      if (d.check("AXIS") || d.check("A")) res._axis = true;

      if (d.check("D")) res._debug = true;
      if (d.check("G")) res._grid = true;
      if (d.check("B")) res._bound = true;
      if (d.check("W")) res.wireframe = true;
      if (d.check("F")) res._full = true;
      if (d.check("Y")) res._yup = true;
      if (d.check("Z")) res._yup = false;

      // when drawing geometry without TCanvas, yup = true by default
      if (res._yup === undefined)
         res._yup = this.svg_canvas().empty();

      return res;
   }

   TGeoPainter.prototype.ActivateInBrowser = function(names, force) {
      // if (this.GetItemName() === null) return;

      if (typeof names == 'string') names = [ names ];

      if (this._hpainter) {
         // show browser if it not visible
         this._hpainter.activate(names, force);

         // if highlight in the browser disabled, suppress in few seconds
         if (!this.options.update_browser)
            setTimeout(this._hpainter.activate.bind(this._hpainter, []), 2000);
      }
   }

   TGeoPainter.prototype.TestMatrixes = function() {
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


   TGeoPainter.prototype.FillContextMenu = function(menu) {
      menu.add("header: Draw options");

      menu.addchk(this.options.update_browser, "Browser update", function() {
         this.options.update_browser = !this.options.update_browser;
         if (!this.options.update_browser) this.ActivateInBrowser([]);
      });
      menu.addchk(this.options.show_controls, "Show Controls", function() {
         this.showControlOptions('toggle');
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
      menu.addchk(this.options.highlight_scene, "Highlight scene", function() {
         this.options.highlight_scene = !this.options.highlight_scene;
      });
      menu.add("Reset camera position", function() {
         this.focusCamera();
         this.Render3D();
      });
      if (!this.options.project)
         menu.addchk(this.options.autoRotate, "Autorotate", function() {
            this.options.autoRotate = !this.options.autoRotate;
            this.autorotate(2.5);
         });
      menu.addchk(this.options.select_in_view, "Select in view", function() {
         this.options.select_in_view = !this.options.select_in_view;
         if (this.options.select_in_view) this.startDrawGeometry();
      });
   }

   /** Method used to set transparency for all geometrical shapes
    * As transperency value one could provide function */
   TGeoPainter.prototype.changeGlobalTransparency = function(transparency, skip_render) {
      var func = (typeof transparency == 'function') ? transparency : null;
      if (func) transparency = this.options.transparency;
      this._toplevel.traverse( function (node) {
         if (node && node.material && (node.material.inherentOpacity !== undefined)) {
            var t = func ? func(node) : undefined;
            if (t !== undefined)
               node.material.opacity = 1 - t;
            else
               node.material.opacity = Math.min(1 - (transparency || 0), node.material.inherentOpacity);
            node.material.transparent = node.material.opacity < 1;
         }
      });
      if (!skip_render) this.Render3D(-1);
   }

   TGeoPainter.prototype.showControlOptions = function(on) {
      if (on==='toggle') on = !this._datgui;

      this.options.show_controls = on;

      if (this._datgui) {
         if (on) return;
         d3.select(this._datgui.domElement).remove();
         this._datgui.destroy();
         delete this._datgui;
         return;
      }
      if (!on) return;

      var painter = this;

      this._datgui = new dat.GUI({ autoPlace: false, width: Math.min(650, painter._renderer.domElement.width / 2) });

      var main = this.select_main();
      if (main.style('position')=='static') main.style('position','relative');

      d3.select(this._datgui.domElement)
               .style('position','absolute')
               .style('top',0).style('right',0);

      main.node().appendChild(this._datgui.domElement);

      function createSSAOgui(is_on) {
         if (!is_on) {
            if (painter._datgui._ssao) {
               // there is no method to destroy folder - why?
               var dom = painter._datgui._ssao.domElement;
               dom.parentNode.removeChild(dom);
               painter._datgui._ssao.destroy();
               if (painter._datgui.__folders && painter._datgui.__folders['SSAO'])
                  painter._datgui.__folders['SSAO'] = undefined;
            }
            delete painter._datgui._ssao;
            return;
         }

         if (painter._datgui._ssao) return;

         painter._datgui._ssao = painter._datgui.addFolder('SSAO');

         painter._datgui._ssao.add( painter._ssaoPass, 'output', {
             'Default': THREE.SSAOPass.OUTPUT.Default,
             'SSAO Only': THREE.SSAOPass.OUTPUT.SSAO,
             'SSAO Only + Blur': THREE.SSAOPass.OUTPUT.Blur,
             'Beauty': THREE.SSAOPass.OUTPUT.Beauty,
             'Depth': THREE.SSAOPass.OUTPUT.Depth,
             'Normal': THREE.SSAOPass.OUTPUT.Normal
         } ).onChange( function ( value ) {
            painter._ssaoPass.output = parseInt( value );
            painter.Render3D();
         } );

         painter._datgui._ssao.add( painter._ssaoPass, 'kernelRadius', 0, 32).listen().onChange(function() {
            painter.Render3D();
         });

         painter._datgui._ssao.add( painter._ssaoPass, 'minDistance', 0.001, 0.02).listen().onChange(function() {
            painter.Render3D();
         });

         painter._datgui._ssao.add( painter._ssaoPass, 'maxDistance', 0.01, 0.3).listen().onChange(function() {
            painter.Render3D();
         });
      }

      if (this.options.project) {

         var bound = this.getGeomBoundingBox(this.getProjectionSource(), 0.01);

         var axis = this.options.project;

         if (this.options.projectPos === undefined)
            this.options.projectPos = (bound.min[axis] + bound.max[axis])/2;

         this._datgui.add(this.options, 'projectPos', bound.min[axis], bound.max[axis])
             .name(axis.toUpperCase() + ' projection')
             .onChange(function (value) {
               painter.startDrawGeometry();
           });

      } else {
         // Clipping Options

         var bound = this.getGeomBoundingBox(this._toplevel, 0.01);

         var clipFolder = this._datgui.addFolder('Clipping');

         for (var naxis=0;naxis<3;++naxis) {
            var axis = !naxis ? "x" : ((naxis===1) ? "y" : "z"),
                  axisC = axis.toUpperCase();

            clipFolder.add(this, 'enable' + axisC).name('Enable '+axisC)
            .listen() // react if option changed outside
            .onChange( function (value) {
               if (value) {
                  createSSAOgui(false);
                  painter._enableSSAO = false;
                  painter._enableClipping = true;
               }
               painter.updateClipping();
            });

            var clip = "clip" + axisC;
            if (this[clip] === 0) this[clip] = (bound.min[axis]+bound.max[axis])/2;

            var item = clipFolder.add(this, clip, bound.min[axis], bound.max[axis])
                   .name(axisC + ' Position')
                   .onChange(function (value) {
                     if (painter[this.enbale_flag]) painter.updateClipping();
                    });

            item.enbale_flag = "enable"+axisC;
         }
      }


      // Appearance Options

      var appearance = this._datgui.addFolder('Appearance');

      appearance.add(this.options, 'highlight').name('Highlight Selection').listen().onChange( function (value) {
         if (!value) painter.HighlightMesh(null);
      });

      appearance.add(this.options, 'transparency', 0.0, 1.0, 0.001)
                     .listen().onChange(this.changeGlobalTransparency.bind(this));

      appearance.add(this.options, 'wireframe').name('Wireframe').listen().onChange( function (value) {
         painter.changeWireFrame(painter._scene, painter.options.wireframe);
      });

      appearance.addColor(this.options, 'background').name('Background').onChange( function() {
          painter._renderer.setClearColor(painter.options.background, 1);
          painter.Render3D(0);
          var bkgr = new THREE.Color(painter.options.background);
          painter._toolbar.changeBrightness((bkgr.r + bkgr.g + bkgr.b) < 1);
      });

      appearance.add(this, 'focusCamera').name('Reset camera position');

      // Advanced Options

      if (this._webgl) {
         var advanced = this._datgui.addFolder('Advanced');

         advanced.add(this, '_enableSSAO').name('Smooth Lighting (SSAO)').onChange( function (value) {
            if (painter._enableSSAO)
               painter.createSSAO();
            createSSAOgui(painter._enableSSAO);

            painter._enableClipping = !painter._enableSSAO;
            painter.updateClipping();
         }).listen();

         advanced.add( this, '_clipIntersection').name("Clip intersection").listen().onChange( function (value) {
            painter.updateClipping();
         });

         advanced.add(this, '_depthTest').name("Depth test").onChange( function (value) {
            painter._toplevel.traverse( function (node) {
               if (node instanceof THREE.Mesh) {
                  node.material.depthTest = value;
               }
            });
            painter.Render3D(0);
         }).listen();

         advanced.add( this.options, 'depthMethod', {
            'Default': "dflt",
            'Raytraicing': "ray",
            'Boundary box': "box",
            'Mesh size': "size",
            'Central point': "pnt"
        } ).name("Rendering order").onChange( function ( value ) {
           delete painter._last_camera_position; // used for testing depth
           painter.Render3D();
        } );

        advanced.add(this, 'resetAdvanced').name('Reset');
      }

      createSSAOgui(this._enableSSAO && this._ssaoPass);
   }

   TGeoPainter.prototype.createSSAO = function() {
      if (!this._webgl || this._ssaoPass) return;

      // this._depthRenderTarget = new THREE.WebGLRenderTarget( this._scene_width, this._scene_height, { minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter } );
      // Setup SSAO pass
      this._ssaoPass = new THREE.SSAOPass( this._scene, this._camera, this._scene_width, this._scene_height );
      this._ssaoPass.kernelRadius = 16;
      this._ssaoPass.renderToScreen = true;

      // Add pass to effect composer
      this._effectComposer.addPass( this._ssaoPass );
   }

   TGeoPainter.prototype.OrbitContext = function(evnt, intersects) {

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
                  if (itemname.indexOf("<prnt>") == 0)
                     itemname = (this.GetItemName() || "top") + itemname.substr(6);
                  name = itemname.substr(itemname.lastIndexOf("/")+1);
                  if (!name) name = itemname;
                  hdr = name;
               } else if (obj.stack) {
                  name = menu.painter._clones.ResolveStack(obj.stack).name;
                  itemname = menu.painter.GetStackFullName(obj.stack);
                  hdr = menu.painter.GetItemName();
                  if (name.indexOf("Nodes/") === 0) hdr = name.substr(6); else
                  if (name.length > 0) hdr = name; else
                  if (!hdr) hdr = "header";

               } else
                  continue;

               menu.add((many ? "sub:" : "header:") + hdr, itemname, function(arg) { this.ActivateInBrowser([arg], true); });

               menu.add("Browse", itemname, function(arg) { this.ActivateInBrowser([arg], true); });

               if (menu.painter._hpainter)
                  menu.add("Inspect", itemname, function(arg) { this._hpainter.display(itemname, "inspect"); });

               if (obj.geo_name) {
                  menu.add("Hide", n, function(indx) {
                     var mesh = intersects[indx].object;
                     mesh.visible = false; // just disable mesh
                     if (mesh.geo_object) mesh.geo_object.$hidden_via_menu = true; // and hide object for further redraw
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

               if (!menu.painter._geom_viewer)
               menu.add("Hide", n, function(indx) {
                  var resolve = menu.painter._clones.ResolveStack(intersects[indx].object.stack);

                  if (resolve.obj && (resolve.node.kind === 0) && resolve.obj.fVolume) {
                     JSROOT.GEO.SetBit(resolve.obj.fVolume, JSROOT.GEO.BITS.kVisThis, false);
                     JSROOT.GEO.updateBrowserIcons(resolve.obj.fVolume, this._hpainter);
                  } else
                  if (resolve.obj && (resolve.node.kind === 1)) {
                     resolve.obj.fRnrSelf = false;
                     JSROOT.GEO.updateBrowserIcons(resolve.obj, this._hpainter);
                  }
                  // intersects[arg].object.visible = false;
                  // this.Render3D();

                  this.testGeomChanges();// while many volumes may disappear, recheck all of them
               });

               if (many) menu.add("endsub:");
            }
         }
         menu.show(evnt);
      });
   }

   TGeoPainter.prototype.FilterIntersects = function(intersects) {

      if (!intersects.length) return intersects;

      // check redirections
      for (var n=0;n<intersects.length;++n)
         if (intersects[n].object.geo_highlight)
            intersects[n].object = intersects[n].object.geo_highlight;

      // remove all elements without stack - indicator that this is geometry object
      // also remove all objects which are mostly transparent
      for (var n=intersects.length-1; n>=0; --n) {

         var obj = intersects[n].object;

         var unique = (obj.stack !== undefined) || (obj.geo_name !== undefined);

         if (unique && obj.material && (obj.material.opacity !== undefined))
            unique = obj.material.opacity >= 0.1;

         if (obj.jsroot_special) unique = false;

         for (var k=0;(k<n) && unique;++k)
            if (intersects[k].object === obj) unique = false;

         if (!unique) intersects.splice(n,1);
      }

      if (this.enableX || this.enableY || this.enableZ) {
         var clippedIntersects = [];

         function myXor(a,b) { return ( a && !b ) || (!a && b); }

         for (var i = 0; i < intersects.length; ++i) {
            var point = intersects[i].point, special = (intersects[i].object.type == "Points"), clipped = true;

            if (this.enableX && myXor(this._clipPlanes[0].normal.dot(point) > this._clipPlanes[0].constant, special)) clipped = false;
            if (this.enableY && myXor(this._clipPlanes[1].normal.dot(point) > this._clipPlanes[1].constant, special)) clipped = false;
            if (this.enableZ && (this._clipPlanes[2].normal.dot(point) > this._clipPlanes[2].constant)) clipped = false;

            if (!clipped) clippedIntersects.push(intersects[i]);
         }

         intersects = clippedIntersects;
      }

      return intersects;
   }

   TGeoPainter.prototype.testCameraPositionChange = function() {
      // function analyzes camera position and start redraw of geometry if
      // objects in view may be changed

      if (!this.options.select_in_view || this._draw_all_nodes) return;


      var matrix = JSROOT.GEO.CreateProjectionMatrix(this._camera);

      var frustum = JSROOT.GEO.CreateFrustum(matrix);

      // check if overall bounding box seen
      if (!frustum.CheckBox(this.getGeomBoundingBox(this._toplevel)))
         this.startDrawGeometry();
   }

   TGeoPainter.prototype.ResolveStack = function(stack) {
      return this._clones && stack ? this._clones.ResolveStack(stack) : null;
   }

   TGeoPainter.prototype.GetStackFullName = function(stack) {
      var mainitemname = this.GetItemName(),
          sub = this.ResolveStack(stack);
      if (!sub || !sub.name) return mainitemname;
      return mainitemname ? (mainitemname + "/" + sub.name) : sub.name;
   }

   /** Add handler which will be called when element is highlighted in geometry drawing
    * Handler should have HighlightMesh function with same arguments as TGeoPainter  */
   TGeoPainter.prototype.AddHighlightHandler = function(handler) {
      if (!handler || typeof handler.HighlightMesh != 'function') return;
      if (!this._highlight_handlers) this._highlight_handlers = [];
      this._highlight_handlers.push(handler);
   }

   //////////////////////

   function GeoDrawingControl(mesh) {
      JSROOT.Painter.InteractiveControl.call(this);
      this.mesh = (mesh && mesh.material) ? mesh : null;
   }

   GeoDrawingControl.prototype = Object.create(JSROOT.Painter.InteractiveControl.prototype);

   GeoDrawingControl.prototype.setHighlight = function(col, indx) {
      return this.drawSpecial(col, indx);
   }

   GeoDrawingControl.prototype.drawSpecial = function(col, indx) {
      var c = this.mesh;
      if (!c || !c.material) return;

      if (col) {
         if (!c.origin)
            c.origin = {
              color: c.material.color,
              opacity: c.material.opacity,
              width: c.material.linewidth,
              size: c.material.size
           };
         c.material.color = new THREE.Color( col );
         c.material.opacity = 1.;
         if (c.hightlightWidthScale && !JSROOT.browser.isWin)
            c.material.linewidth = c.origin.width * c.hightlightWidthScale;
         if (c.highlightScale)
            c.material.size = c.origin.size * c.highlightScale;
         return true;
      } else if (c.origin) {
         c.material.color = c.origin.color;
         c.material.opacity = c.origin.opacity;
         if (c.hightlightWidthScale)
            c.material.linewidth = c.origin.width;
         if (c.highlightScale)
            c.material.size = c.origin.size;
         return true;
      }
   }

   ////////////////////////

   TGeoPainter.prototype.HighlightMesh = function(active_mesh, color, geo_object, geo_index, geo_stack, no_recursive) {

      if (geo_object) {
         active_mesh = active_mesh ? [ active_mesh ] : [];
         var extras = this.getExtrasContainer();
         if (extras)
            extras.traverse(function(obj3d) {
               if ((obj3d.geo_object === geo_object) && (active_mesh.indexOf(obj3d)<0)) active_mesh.push(obj3d);
            });
      } else if (geo_stack && this._toplevel) {
         active_mesh = [];
         this._toplevel.traverse(function(mesh) {
            if ((mesh instanceof THREE.Mesh) && JSROOT.GEO.IsSameStack(mesh.stack, geo_stack)) active_mesh.push(mesh);
         });
      } else {
         active_mesh = active_mesh ? [ active_mesh ] : [];
      }

      if (!active_mesh.length) active_mesh = null;

      if (active_mesh) {
         // check if highlight is disabled for correspondent objects kinds
         if (active_mesh[0].geo_object) {
            if (!this.options.highlight_scene) active_mesh = null;
         } else {
            if (!this.options.highlight) active_mesh = null;
         }
      }

      if (!no_recursive) {
         // check all other painters

         if (active_mesh) {
            if (!geo_object) geo_object = active_mesh[0].geo_object;
            if (!geo_stack) geo_stack = active_mesh[0].stack;
         }

         var lst = this._highlight_handlers || (!this._main_painter ? this._slave_painters : this._main_painter._slave_painters.concat([this._main_painter]));

         for (var k=0;k<lst.length;++k)
            if (lst[k]!==this) lst[k].HighlightMesh(null, color, geo_object, geo_index, geo_stack, true);
      }

      var curr_mesh = this._selected_mesh;

      function get_ctrl(mesh) {
         return mesh.get_ctrl ? mesh.get_ctrl() : new GeoDrawingControl(mesh);
      }

      // check if selections are the same
      if (!curr_mesh && !active_mesh) return false;
      var same = false;
      if (curr_mesh && active_mesh && (curr_mesh.length == active_mesh.length)) {
         same = true;
         for (var k=0;(k<curr_mesh.length) && same;++k) {
            if ((curr_mesh[k] !== active_mesh[k]) || get_ctrl(curr_mesh[k]).checkHighlightIndex(geo_index)) same = false;
         }
      }
      if (same) return !!curr_mesh;

      if (curr_mesh)
         for (var k=0;k<curr_mesh.length;++k)
            get_ctrl(curr_mesh[k]).setHighlight();

      this._selected_mesh = active_mesh;

      if (active_mesh)
         for (var k=0;k<active_mesh.length;++k)
            get_ctrl(active_mesh[k]).setHighlight(color || 0xffaa33, geo_index);

      this.Render3D(0);

      return !!active_mesh;
   }

   TGeoPainter.prototype.ProcessMouseClick = function(pnt, intersects, evnt) {
      if (!intersects.length) return;

      var mesh = intersects[0].object;
      if (!mesh.get_ctrl) return;

      var ctrl = mesh.get_ctrl();

      var click_indx = ctrl.extractIndex(intersects[0]);

      ctrl.evnt = evnt;

      if (ctrl.setSelected("blue", click_indx))
         this.Render3D();

      ctrl.evnt = null;
   }

   TGeoPainter.prototype.addOrbitControls = function() {

      if (this._controls || this._usesvg || JSROOT.BatchMode) return;

      var painter = this;

      this.SetTooltipAllowed(JSROOT.gStyle.Tooltip > 0);

      this._controls = JSROOT.Painter.CreateOrbitControl(this, this._camera, this._scene, this._renderer, this._lookat);

      if (this.options.project || this.options.ortho_camera) this._controls.enableRotate = false;

      this._controls.ContextMenu = this.OrbitContext.bind(this);

      this._controls.ProcessMouseMove = function(intersects) {

         var active_mesh = null, tooltip = null, resolve = null, names = [], geo_object, geo_index;

         // try to find mesh from intersections
         for (var k=0;k<intersects.length;++k) {
            var obj = intersects[k].object, info = null;
            if (!obj) continue;
            if (obj.geo_object) info = obj.geo_name; else
            if (obj.stack) info = painter.GetStackFullName(obj.stack);
            if (!info) continue;

            if (info.indexOf("<prnt>")==0)
               info = painter.GetItemName() + info.substr(6);

            names.push(info);

            if (!active_mesh) {
               active_mesh = obj;
               tooltip = info;
               geo_object = obj.geo_object;
               if (obj.get_ctrl) {
                  geo_index = obj.get_ctrl().extractIndex(intersects[k]);
                  if ((geo_index !== undefined) && (typeof tooltip == "string")) tooltip += " indx:" + JSON.stringify(geo_index);
               }
               if (active_mesh.stack) resolve = painter.ResolveStack(active_mesh.stack);
            }
         }

         painter.HighlightMesh(active_mesh, undefined, geo_object, geo_index);

         if (painter.options.update_browser) {
            if (painter.options.highlight && tooltip) names = [ tooltip ];
            painter.ActivateInBrowser(names);
         }

         if (!resolve || !resolve.obj) return tooltip;

         var lines = JSROOT.GEO.provideInfo(resolve.obj);
         lines.unshift(tooltip);

         return { name: resolve.obj.fName, title: resolve.obj.fTitle || resolve.obj._typename, lines: lines };
      }

      this._controls.ProcessMouseLeave = function() {
         this.ProcessMouseMove([]); // to disable highlight and reset browser
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

   TGeoPainter.prototype.addTransformControl = function() {
      if (this._tcontrols) return;

      if ( !this.options._debug && !this.options._grid ) return;

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

   TGeoPainter.prototype.nextDrawAction = function() {
      // return false when nothing todo
      // return true if one could perform next action immediately
      // return 1 when call after short timeout required
      // return 2 when call must be done from processWorkerReply

      if (!this._clones || (this.drawing_stage == 0)) return false;

      if (this.drawing_stage == 1) {

         if (this._geom_viewer) {
            this._draw_all_nodes = false;
            this.drawing_stage = 3;
            return true;
         }

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
            if (frustum.CheckBox(this.getGeomBoundingBox(this._toplevel))) {
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
         var need_worker = !JSROOT.BatchMode && ((numvis > 10000) || (matrix && (this._clones.ScanVisible() > 1e5)));

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

         if (this._new_append_nodes) {

            this._new_draw_nodes = this._draw_nodes.concat(this._new_append_nodes);

            delete this._new_append_nodes;

         } else if (this._draw_nodes) {

            var del;
            if (this._geom_viewer)
               del = this._draw_nodes;
            else
               del = this._clones.MergeVisibles(this._new_draw_nodes, this._draw_nodes);

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

         var tm0 = new Date().getTime(), ready = true,
             toplevel = this.options.project ? this._full_geom : this._toplevel;

         for (var n = 0; n < this._draw_nodes.length; ++n) {
            var entry = this._draw_nodes[n];
            if (entry.done) continue;

            /// shape can be provided with entry itself
            var shape = entry.server_shape || this._build_shapes[entry.shapeid];
            if (!shape.ready) {
               if (this.drawing_stage === 8) console.warn('shape marked as not ready when should');
               ready = false;
               continue;
            }

            entry.done = true;
            shape.used = true; // indicate that shape was used in building

            if (this.createEntryMesh(entry, shape, toplevel)) {
               this._num_meshes++;
               this._num_faces += shape.nfaces;
            }

            var tm1 = new Date().getTime();
            if (tm1 - tm0 > 500) { ready = false; break; }
         }

         if (ready) {
            if (this.options.project) {
               this.drawing_log = "Build projection";
               this.drawing_stage = 10;
               return true;
            }

            this.drawing_log = "Building done";
            this.drawing_stage = 0;
            return false;
         }

         if (this.drawing_stage > 7)
            this.drawing_log = "Building meshes " + this._num_meshes + " / " + this._num_faces;
         return true;
      }

      if (this.drawing_stage === 9) {
         // wait for main painter to be ready

         if (!this._main_painter) {
            console.warn('MAIN PAINTER DISAPPER');
            this.drawing_stage = 0;
            return false;
         }
         if (!this._main_painter._drawing_ready) return 1;

         this.drawing_stage = 10; // just do projection
      }

      if (this.drawing_stage === 10) {
         this.doProjection();
         this.drawing_log = "Building done";
         this.drawing_stage = 0;
         return false;
      }

      console.error('never come here stage = ' + this.drawing_stage);

      return false;
   }

   /** Insert appropriate mesh for given entry
    * @private*/
   TGeoPainter.prototype.createEntryMesh = function(entry, shape, toplevel) {
      if (!shape.geom || (shape.nfaces === 0)) {
         // node is visible, but shape does not created
         this._clones.CreateObject3D(entry.stack, toplevel, 'delete_mesh');
         return false;
      }

      // workaround for the TGeoOverlap, where two branches should get predefined color
      if (this._splitColors && entry.stack) {
         if (entry.stack[0]===0) entry.custom_color = "green"; else
         if (entry.stack[0]===1) entry.custom_color = "blue";
      }

      var prop = this._clones.getDrawEntryProperties(entry);

      var obj3d = this._clones.CreateObject3D(entry.stack, toplevel, this.options);

      prop.material.wireframe = this.options.wireframe;

      prop.material.side = this.bothSides ? THREE.DoubleSide : THREE.FrontSide;

      var mesh;

      if (obj3d.matrixWorld.determinant() > -0.9) {
         mesh = new THREE.Mesh( shape.geom, prop.material );
      } else {
         mesh = JSROOT.GEO.createFlippedMesh(obj3d, shape, prop.material);
      }

      obj3d.add(mesh);

      // keep full stack of nodes
      mesh.stack = entry.stack;
      mesh.renderOrder = this._clones.maxdepth - entry.stack.length; // order of transparency handling

      // keep hierarchy level
      mesh.$jsroot_order = obj3d.$jsroot_depth;

      // set initial render order, when camera moves, one must refine it
      //mesh.$jsroot_order = mesh.renderOrder =
      //   this._clones.maxdepth - ((obj3d.$jsroot_depth !== undefined) ? obj3d.$jsroot_depth : entry.stack.length);

      if (this.options._debug || this.options._full) {
         var wfg = new THREE.WireframeGeometry( mesh.geometry ),
             wfm = new THREE.LineBasicMaterial( { color: prop.fillcolor, linewidth: prop.linewidth || 1 } ),
             helper = new THREE.LineSegments(wfg, wfm);
         obj3d.add(helper);
      }

      if (this.options._bound || this.options._full) {
         var boxHelper = new THREE.BoxHelper( mesh );
         obj3d.add( boxHelper );
      }

      return true;
   }

   /** function used by geometry viewer to show more nodes
    * These nodes excluded from selection logic and always inserted into the model
    * Shape already should be created and assigned to the node
    * @private */
   TGeoPainter.prototype.appendMoreNodes = function(nodes, from_drawing) {
      if (this.drawing_stage && !from_drawing) {
         this._provided_more_nodes = nodes;
         return;
      }

      // delete old nodes
      if (this._more_nodes)
         for (var n=0;n<this._more_nodes.length;++n) {
            var entry = this._more_nodes[n];
            var obj3d = this._clones.CreateObject3D(entry.stack, this._toplevel, 'delete_mesh');
            JSROOT.Painter.DisposeThreejsObject(obj3d);
            JSROOT.GEO.cleanupShape(entry.server_shape);
            delete entry.server_shape;
         }

      delete this._more_nodes;

      if (!nodes) return;

      var real_nodes = [];

      for (var k=0;k<nodes.length;++k) {
         var entry = nodes[k];
         var shape = entry.server_shape;
         if (!shape || !shape.ready) continue;

         entry.done = true;
         shape.used = true; // indicate that shape was used in building

         if (this.createEntryMesh(entry, shape, this._toplevel))
            real_nodes.push(entry);
      }

      // remember additional nodes only if they include shape - otherwise one can ignore them
      if (real_nodes) this._more_nodes = real_nodes;

      if (!from_drawing) this.Render3D();
   }

   /** Returns hierarchy of 3D objects used to produce projection.
    * Typically external master painter is used, but also internal data can be used
    * @private */

   TGeoPainter.prototype.getProjectionSource = function() {
      if (this._clones_owner)
         return this._full_geom;
      if (!this._main_painter) {
         console.warn('MAIN PAINTER DISAPPER');
         return null;
      }
      if (!this._main_painter._drawing_ready) {
         console.warn('MAIN PAINTER NOT READY WHEN DO PROJECTION');
         return null;
      }
      return this._main_painter._toplevel;
   }

   TGeoPainter.prototype.getGeomBoundingBox = function(topitem, scalar) {
      var box3 = new THREE.Box3(), check_any = !this._clones;

      if (!topitem) {
         box3.min.x = box3.min.y = box3.min.z = -1;
         box3.max.x = box3.max.y = box3.max.z = 1;
         return box3;
      }

      box3.makeEmpty();

      topitem.traverse(function(mesh) {
         if (check_any || ((mesh instanceof THREE.Mesh) && mesh.stack)) JSROOT.GEO.getBoundingBox(mesh, box3);
      });

      if (scalar !== undefined) box3.expandByVector(box3.getSize(new THREE.Vector3()).multiplyScalar(scalar));

      return box3;
   }


   TGeoPainter.prototype.doProjection = function() {
      var toplevel = this.getProjectionSource(), pthis = this;

      if (!toplevel) return false;

      JSROOT.Painter.DisposeThreejsObject(this._toplevel, true);

      var axis = this.options.project;

      if (this.options.projectPos === undefined) {

         var bound = this.getGeomBoundingBox(toplevel),
             min = bound.min[this.options.project], max = bound.max[this.options.project],
             mean = (min+max)/2;

         if ((min<0) && (max>0) && (Math.abs(mean) < 0.2*Math.max(-min,max))) mean = 0; // if middle is around 0, use 0

         this.options.projectPos = mean;
      }

      toplevel.traverse(function(mesh) {
         if (!(mesh instanceof THREE.Mesh) || !mesh.stack) return;

         var geom2 = JSROOT.GEO.projectGeometry(mesh.geometry, mesh.parent.matrixWorld, pthis.options.project, pthis.options.projectPos, mesh._flippedMesh);

         if (!geom2) return;

         var mesh2 = new THREE.Mesh( geom2, mesh.material.clone() );

         pthis._toplevel.add(mesh2);

         mesh2.stack = mesh.stack;
      });

      return true;
   }

   TGeoPainter.prototype.SameMaterial = function(node1, node2) {

      if ((node1===null) || (node2===null)) return node1 === node2;

      if (node1.fVolume.fLineColor >= 0)
         return (node1.fVolume.fLineColor === node2.fVolume.fLineColor);

       var m1 = (node1.fVolume.fMedium !== null) ? node1.fVolume.fMedium.fMaterial : null;
       var m2 = (node2.fVolume.fMedium !== null) ? node2.fVolume.fMedium.fMaterial : null;

       if (m1 === m2) return true;

       if ((m1 === null) || (m2 === null)) return false;

       return (m1.fFillStyle === m2.fFillStyle) && (m1.fFillColor === m2.fFillColor);
    }

   TGeoPainter.prototype.createScene = function(w, h) {
      // three.js 3D drawing
      this._scene = new THREE.Scene();
      this._scene.fog = new THREE.Fog(0xffffff, 1, 10000);
      this._scene.overrideMaterial = new THREE.MeshLambertMaterial( { color: 0x7000ff, transparent: true, opacity: 0.2, depthTest: false } );

      this._scene_width = w;
      this._scene_height = h;

      if (this.options.ortho_camera) {
         this._camera =  new THREE.OrthographicCamera(-600, 600, -600, 600, 1, 10000);
      } else {
         this._camera = new THREE.PerspectiveCamera(25, w / h, 1, 10000);
         this._camera.up = this.options._yup ? new THREE.Vector3(0,1,0) : new THREE.Vector3(0,0,1);
      }

      this._scene.add( this._camera );

      this._selected_mesh = null;

      this._overall_size = 10;

      this._toplevel = new THREE.Object3D();

      this._scene.add(this._toplevel);

      var rrr = JSROOT.Painter.Create3DRenderer(w, h, this._usesvg, this._usesvgimg, this._webgl,
            { antialias: true, logarithmicDepthBuffer: false, preserveDrawingBuffer: true });

      this._webgl = rrr.usewebgl;
      this._renderer = rrr.renderer;

      if (this._renderer.setPixelRatio && !JSROOT.nodejs)
         this._renderer.setPixelRatio(window.devicePixelRatio);
      this._renderer.setSize(w, h, !this._fit_main_area);
      this._renderer.localClippingEnabled = true;

      this._renderer.setClearColor(this.options.background, 1);

/*      if (usesvg) {
         // this._renderer = new THREE.SVGRenderer( { precision: 0, astext: true } );
         this._renderer = THREE.CreateSVGRenderer(false, 0, document);
         if (this._renderer.makeOuterHTML !== undefined) {
            // this is indication of new three.js functionality
            if (!JSROOT.svg_workaround) JSROOT.svg_workaround = [];
            this._renderer.workaround_id = JSROOT.svg_workaround.length;
            JSROOT.svg_workaround[this._renderer.workaround_id] = "<svg></svg>"; // dummy, need to be replaced

            this._renderer.domElement = document.createElementNS( 'http://www.w3.org/2000/svg', 'path');
            this._renderer.domElement.setAttribute('jsroot_svg_workaround', this._renderer.workaround_id);
         }
      } else {
         this._renderer = webgl ?
                           new THREE.WebGLRenderer({ antialias: true, logarithmicDepthBuffer: false,
                                                     preserveDrawingBuffer: true }) :
                           new THREE.SoftwareRenderer({ antialias: true });
         this._renderer.setPixelRatio(window.devicePixelRatio);
      }
      this._renderer.setSize(w, h, !this._fit_main_area);
      this._renderer.localClippingEnabled = true;

      */

      if (this._fit_main_area && !this._usesvg) {
         this._renderer.domElement.style.width = "100%";
         this._renderer.domElement.style.height = "100%";
         var main = this.select_main();
         if (main.style('position')=='static') main.style('position','relative');
      }

      this._animating = false;

      // Clipping Planes

      this._clipIntersection = true;
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

      // Default Settings

      this._depthTest = true;

      // Smooth Lighting Shader (Screen Space Ambient Occlusion)
      // http://threejs.org/examples/webgl_postprocessing_ssao.html

      // these two parameters are exclusive - either SSAO or clipping can work at same time
      this._enableSSAO = this.options.ssao;
      this._enableClipping = !this._enableSSAO;

      this._effectComposer = new THREE.EffectComposer( this._renderer );
      this._effectComposer.addPass( new THREE.RenderPass( this._scene, this._camera ) );

      if (this._enableSSAO)
         this.createSSAO();

      if (this._fit_main_area && (this._usesvg || this._usesvgimg)) {
         // create top-most SVG for geomtery drawings
         var svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
         svg.appendChild(rrr.dom);
         return svg;
      }

      return rrr.dom;
   }


   TGeoPainter.prototype.startDrawGeometry = function(force) {

      if (!force && (this.drawing_stage!==0)) {
         this._draw_nodes_again = true;
         return;
      }

      this._startm = new Date().getTime();
      this._last_render_tm = this._startm;
      this._last_render_meshes = 0;
      this.drawing_stage = 1;
      this._drawing_ready = false;
      this.drawing_log = "collect visible";
      this._num_meshes = 0;
      this._num_faces = 0;
      this._selected_mesh = null;

      if (this.options.project) {
         if (this._clones_owner) {
            if (this._full_geom) {
               this.drawing_stage = 10;
               this.drawing_log = "build projection";
            } else {
               this._full_geom = new THREE.Object3D();
            }

         } else {
            this.drawing_stage = 9;
            this.drawing_log = "wait for main painter";
         }
      }

      delete this._last_manifest;
      delete this._last_hidden; // clear list of hidden objects

      delete this._draw_nodes_again; // forget about such flag

      this.continueDraw();
   }

   TGeoPainter.prototype.resetAdvanced = function() {
      if (this._ssaoPass) {
         this._ssaoPass.kernelRadius = 16;
         this._ssaoPass.output = THREE.SSAOPass.OUTPUT.Default;
      }

      this._depthTest = true;
      this._clipIntersection = true;
      this.options.depthMethod = "ray";

      var painter = this;
      this._toplevel.traverse( function (node) {
         if (node instanceof THREE.Mesh) {
            node.material.depthTest = painter._depthTest;
         }
      });

      this.Render3D(0);
   }

   /** Assign clipping attributes to the meshes - supported only for webgl
    * @private */
   TGeoPainter.prototype.updateClipping = function(without_render) {
      if (!this._webgl) return;

      this._clipPlanes[0].constant = this.clipX;
      this._clipPlanes[1].constant = -this.clipY;
      this._clipPlanes[2].constant = this.options._yup ? -this.clipZ : this.clipZ;

      var panels = [];
      if (this._enableClipping) {
         if (this.enableX) panels.push(this._clipPlanes[0]);
         if (this.enableY) panels.push(this._clipPlanes[1]);
         if (this.enableZ) panels.push(this._clipPlanes[2]);
      }
      if (panels.length == 0) panels = null;

      var any_clipping = !!panels, ci = this._clipIntersection,
          material_side = any_clipping ? THREE.DoubleSide : THREE.FrontSide;

      this._scene.traverse( function (node) {
         if (node.hasOwnProperty("material") && node.material && (node.material.clippingPlanes !== undefined)) {

            if (node.material.clippingPlanes !== panels) {
               node.material.clipIntersection = ci;
               node.material.clippingPlanes = panels;
               node.material.needsUpdate = true;
            }

            if (node.material.emissive !== undefined) {
               if (node.material.side != material_side) {
                  node.material.side = material_side;
                  node.material.needsUpdate = true;
               }
            }
         }
      });

      this.bothSides = any_clipping;

      if (!without_render) this.Render3D(0);
   }

   TGeoPainter.prototype.getGeomBox = function() {
      var extras = this.getExtrasContainer('collect');

      var box = this.getGeomBoundingBox(this._toplevel);

      if (extras)
         for (var k=0;k<extras.length;++k) this._toplevel.add(extras[k]);

      return box;
   }

   TGeoPainter.prototype.getOverallSize = function(force) {
      if (!this._overall_size || force) {
         var box = this.getGeomBoundingBox(this._toplevel);

         // if detect of coordinates fails - ignore
         if (isNaN(box.min.x)) return 1000;

         var sizex = box.max.x - box.min.x,
             sizey = box.max.y - box.min.y,
             sizez = box.max.z - box.min.z;

         this._overall_size = 2 * Math.max(sizex, sizey, sizez);
      }

      return this._overall_size;
   }

   TGeoPainter.prototype.adjustCameraPosition = function(first_time) {

      if (!this._toplevel) return;

      var box = this.getGeomBoundingBox(this._toplevel);

      // if detect of coordinates fails - ignore
      if (isNaN(box.min.x)) return;

      var sizex = box.max.x - box.min.x,
          sizey = box.max.y - box.min.y,
          sizez = box.max.z - box.min.z,
          midx = (box.max.x + box.min.x)/2,
          midy = (box.max.y + box.min.y)/2,
          midz = (box.max.z + box.min.z)/2;

      this._overall_size = 2 * Math.max(sizex, sizey, sizez);

      this._scene.fog.near = this._overall_size * 2;
      this._camera.near = this._overall_size / 350;
      this._scene.fog.far = this._overall_size * 12;
      this._camera.far = this._overall_size * 12;

      if (first_time) {
         this.clipX = midx;
         this.clipY = midy;
         this.clipZ = midz;
      }

      if (this.options.ortho_camera) {
         this._camera.left = box.min.x;
         this._camera.right = box.max.x;
         this._camera.top = box.max.y;
         this._camera.bottom = box.min.y;
      }

      // this._camera.far = 100000000000;

      this._camera.updateProjectionMatrix();

      var k = 2*this.options.zoom;

      if (this.options.ortho_camera) {
         this._camera.position.set(midx, midy, Math.max(sizex,sizey));
      } else if (this.options.project) {
         switch (this.options.project) {
            case 'x': this._camera.position.set(k*1.5*Math.max(sizey,sizez), 0, 0); break;
            case 'y': this._camera.position.set(0, k*1.5*Math.max(sizex,sizez), 0); break;
            case 'z': this._camera.position.set(0, 0, k*1.5*Math.max(sizex,sizey)); break;
         }
      } else if (this.options._yup) {
         this._camera.position.set(midx-k*Math.max(sizex,sizez), midy+k*sizey, midz-k*Math.max(sizex,sizez));
      } else {
         this._camera.position.set(midx-k*Math.max(sizex,sizey), midy-k*Math.max(sizex,sizey), midz+k*sizez);
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

   TGeoPainter.prototype.focusOnItem = function(itemname) {

      if (!itemname || !this._clones) return;

      var stack = this._clones.FindStackByName(itemname);

      if (!stack) return;

      var info = this._clones.ResolveStack(stack, true);

      this.focusCamera( info, false );
   }

   TGeoPainter.prototype.focusCamera = function( focus, clip ) {

      if (this.options.project) return this.adjustCameraPosition();

      var autoClip = clip === undefined ? false : clip;

      var box = new THREE.Box3();
      if (focus === undefined) {
         box = this.getGeomBoundingBox(this._toplevel);
      } else if (focus instanceof THREE.Mesh) {
         box.setFromObject(focus);
      } else {
         var center = new THREE.Vector3().setFromMatrixPosition(focus.matrix),
             node = focus.node,
             halfDelta = new THREE.Vector3( node.fDX, node.fDY, node.fDZ ).multiplyScalar(0.5);
         box.min = center.clone().sub(halfDelta);
         box.max = center.clone().add(halfDelta);
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

      // probably, reduce number of frames
      var frames = 50, step = 0;

      // Amount to change camera position at each step
      var posIncrement = position.sub(this._camera.position).divideScalar(frames);
      // Amount to change "lookAt" so it will end pointed at target
      var targetIncrement = target.sub(oldTarget).divideScalar(frames);
      // console.log( targetIncrement );

      // Automatic Clipping

      if (autoClip) {

         var topBox = this.getGeomBoundingBox(this._toplevel);

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
            if (!painter._geom_viewer)
               painter.startDrawGeometry();
         }
         var smoothFactor = -Math.cos( ( 2.0 * Math.PI * step ) / frames ) + 1.0;
         painter._camera.position.add( posIncrement.clone().multiplyScalar( smoothFactor ) );
         oldTarget.add( targetIncrement.clone().multiplyScalar( smoothFactor ) );
         painter._lookat = oldTarget;
         painter._camera.lookAt( painter._lookat );
         painter._camera.updateProjectionMatrix();

         var tm1 = new Date().getTime();
         if (autoClip) {
            painter.clipX += incrementX * smoothFactor;
            painter.clipY += incrementY * smoothFactor;
            painter.clipZ += incrementZ * smoothFactor;
            painter.updateClipping();
         } else {
            painter.Render3D(0);
         }
         var tm2 = new Date().getTime();
         if ((step==0) && (tm2-tm1 > 200)) frames = 20;
         step++;
         painter._animating = step < frames;
      }

      animate();

   //   this._controls.update();
   }

   TGeoPainter.prototype.autorotate = function(speed) {

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

      if (this._webgl) animate();
   }

   TGeoPainter.prototype.completeScene = function() {

      if ( this.options._debug || this.options._grid ) {
         if ( this.options._full ) {
            var boxHelper = new THREE.BoxHelper(this._toplevel);
            this._scene.add( boxHelper );
         }
         this._scene.add( new THREE.AxesHelper( 2 * this._overall_size ) );
         this._scene.add( new THREE.GridHelper( Math.ceil( this._overall_size), Math.ceil( this._overall_size ) / 50 ) );
         this.helpText("<font face='verdana' size='1' color='red'><center>Transform Controls<br>" +
               "'T' translate | 'R' rotate | 'S' scale<br>" +
               "'+' increase size | '-' decrease size<br>" +
               "'W' toggle wireframe/solid display<br>"+
         "keep 'Ctrl' down to snap to grid</center></font>");
      }
   }


   TGeoPainter.prototype.drawCount = function(unqievis, clonetm) {

      var res = 'Unique nodes: ' + this._clones.nodes.length + '<br/>' +
                'Unique visible: ' + unqievis + '<br/>' +
                'Time to clone: ' + clonetm + 'ms <br/>';

      // need to fill cached value line numvischld
      this._clones.ScanVisible();

      var painter = this, nshapes = 0;

      var arg = {
         cnt: [],
         func: function(node) {
            if (this.cnt[this.last]===undefined)
               this.cnt[this.last] = 1;
            else
               this.cnt[this.last]++;

            nshapes += JSROOT.GEO.CountNumShapes(painter._clones.GetNodeShape(node.id));

            // for debugginf - search if there some TGeoHalfSpace
            //if (JSROOT.GEO.HalfSpace) {
            //    var entry = this.CopyStack();
            //    var res = painter._clones.ResolveStack(entry.stack);
            //    console.log('SAW HALF SPACE', res.name);
            //    JSROOT.GEO.HalfSpace = false;
            //}
            return true;
         }
      };

      var tm1 = new Date().getTime();
      var numvis = this._clones.ScanVisible(arg);
      var tm2 = new Date().getTime();

      res += 'Total visible nodes: ' + numvis + '<br/>';
      res += 'Total shapes: ' + nshapes + '<br/>';

      for (var lvl=0;lvl<arg.cnt.length;++lvl) {
         if (arg.cnt[lvl] !== undefined)
            res += ('  lvl' + lvl + ': ' + arg.cnt[lvl] + '<br/>');
      }

      res += "Time to scan: " + (tm2-tm1) + "ms <br/>";

      res += "<br/><br/>Check timing for matrix calculations ...<br/>";

      var elem = this.select_main().style('overflow', 'auto').html(res);

      setTimeout(function() {
         arg.domatrix = true;
         tm1 = new Date().getTime();
         numvis = painter._clones.ScanVisible(arg);
         tm2 = new Date().getTime();
         elem.append("p").text("Time to scan with matrix: " + (tm2-tm1) + "ms");
         painter.DrawingReady();
      }, 100);

      return this;
   }


   TGeoPainter.prototype.PerformDrop = function(obj, itemname, hitem, opt, call_back) {

      if (obj && (obj.$kind==='TTree')) {
         // drop tree means function call which must extract tracks from provided tree

         var funcname = "extract_geo_tracks";

         if (opt && opt.indexOf("$")>0) {
            funcname = opt.substr(0, opt.indexOf("$"));
            opt = opt.substr(opt.indexOf("$")+1);
         }

         var func = JSROOT.findFunction(funcname);

         if (!func) return JSROOT.CallBack(call_back);

         var geo_painter = this;

         return func(obj, opt, function(tracks) {
            if (tracks) {
               geo_painter.drawExtras(tracks, "", false); // FIXME: probably tracks should be remembered??
               this.updateClipping(true);
               geo_painter.Render3D(100);
            }
            JSROOT.CallBack(call_back); // finally callback
         });
      }

      if (this.drawExtras(obj, itemname)) {
         if (hitem) hitem._painter = this; // set for the browser item back pointer
      }

      JSROOT.CallBack(call_back);
   }

   TGeoPainter.prototype.MouseOverHierarchy = function(on, itemname, hitem) {
      // function called when mouse is going over the item in the browser

      if (!this.options) return; // protection for cleaned-up painter

      var obj = hitem._obj;
      if (this.options._debug)
         console.log('Mouse over', on, itemname, (obj ? obj._typename : "---"));

      // let's highlight tracks and hits only for the time being
      if (!obj || (obj._typename !== "TEveTrack" && obj._typename !== "TEvePointSet" && obj._typename !== "TPolyMarker3D")) return;

      this.HighlightMesh(null, 0x00ff00, on ? obj : null);
   }

   TGeoPainter.prototype.clearExtras = function() {
      this.getExtrasContainer("delete");
      delete this._extraObjects; // workaround, later will be normal function
      this.Render3D();
   }

   /** Register extra objects like tracks or hits
    * @desc Rendered after main geometry volumes are created
    * Check if object already exists to prevent duplication */
   TGeoPainter.prototype.addExtra = function(obj, itemname) {
      if (this._extraObjects === undefined)
         this._extraObjects = JSROOT.Create("TList");

      if (this._extraObjects.arr.indexOf(obj)>=0) return false;

      this._extraObjects.Add(obj, itemname);

      delete obj.$hidden_via_menu; // remove previous hidden property

      return true;
   }

   TGeoPainter.prototype.ExtraObjectVisible = function(hpainter, hitem, toggle) {
      if (!this._extraObjects) return;

      var itemname = hpainter.itemFullName(hitem),
          indx = this._extraObjects.opt.indexOf(itemname);

      if ((indx<0) && hitem._obj) {
         indx = this._extraObjects.arr.indexOf(hitem._obj);
         // workaround - if object found, replace its name
         if (indx>=0) this._extraObjects.opt[indx] = itemname;
      }

      if (indx < 0) return;

      var obj = this._extraObjects.arr[indx],
          res = obj.$hidden_via_menu ? false : true;

      if (toggle) {
         obj.$hidden_via_menu = res; res = !res;

         var mesh = null;
         // either found painted object or just draw once again
         this._toplevel.traverse(function(node) { if (node.geo_object === obj) mesh = node; });

         if (mesh) mesh.visible = res; else
         if (res) {
            this.drawExtras(obj, "", false);
            this.updateClipping(true);
         }

         if (mesh || res) this.Render3D();
      }

      return res;
   }

   TGeoPainter.prototype.drawExtras = function(obj, itemname, add_objects) {
      if (!obj || obj._typename===undefined) return false;

      // if object was hidden via menu, do not redraw it with next draw call
      if (!add_objects && obj.$hidden_via_menu) return false;

      var isany = false, do_render = false;
      if (add_objects === undefined) {
         add_objects = true;
         do_render = true;
      }

      if ((obj._typename === "TList") || (obj._typename === "TObjArray")) {
         if (!obj.arr) return false;
         for (var n=0;n<obj.arr.length;++n) {
            var sobj = obj.arr[n], sname = obj.opt[n];
            if (!sname) sname = (itemname || "<prnt>") + "/[" + n + "]";
            if (this.drawExtras(sobj, sname, add_objects)) isany = true;
         }
      } else if (obj._typename === 'THREE.Mesh') {
         // adding mesh as is
         this.getExtrasContainer().add(obj);
         isany = true;
      } else if (obj._typename === 'TGeoTrack') {
         if (add_objects && !this.addExtra(obj, itemname)) return false;
         isany = this.drawGeoTrack(obj, itemname);
      } else if ((obj._typename === 'TEveTrack') || (obj._typename === 'ROOT::Experimental::TEveTrack')) {
         if (add_objects && !this.addExtra(obj, itemname)) return false;
         isany = this.drawEveTrack(obj, itemname);
      } else if ((obj._typename === 'TEvePointSet') || (obj._typename === "ROOT::Experimental::TEvePointSet") || (obj._typename === "TPolyMarker3D")) {
         if (add_objects && !this.addExtra(obj, itemname)) return false;
         isany = this.drawHit(obj, itemname);
      } else if ((obj._typename === "TEveGeoShapeExtract") || (obj._typename === "ROOT::Experimental::TEveGeoShapeExtract")) {
         if (add_objects && !this.addExtra(obj, itemname)) return false;
         isany = this.drawExtraShape(obj, itemname);
      }

      if (isany && do_render) {
         this.updateClipping(true);
         this.Render3D(100);
      }

      return isany;
   }

   TGeoPainter.prototype.getExtrasContainer = function(action, name) {
      if (!this._toplevel) return null;

      if (!name) name = "tracks";

      var extras = null, lst = [];
      for (var n=0;n<this._toplevel.children.length;++n) {
         var chld = this._toplevel.children[n];
         if (!chld._extras) continue;
         if (action==='collect') { lst.push(chld); continue; }
         if (chld._extras === name) { extras = chld; break; }
      }

      if (action==='collect') {
         for (var k=0;k<lst.length;++k) this._toplevel.remove(lst[k]);
         return lst;
      }

      if (action==="delete") {
         if (extras) this._toplevel.remove(extras);
         JSROOT.Painter.DisposeThreejsObject(extras);
         return null;
      }

      if ((action!=="get") && !extras) {
         extras = new THREE.Object3D();
         extras._extras = name;
         this._toplevel.add(extras);
      }

      return extras;
   }

   TGeoPainter.prototype.drawGeoTrack = function(track, itemname) {
      if (!track || !track.fNpoints) return false;

      var track_width = track.fLineWidth || 1,
          track_color = JSROOT.Painter.root_colors[track.fLineColor] || "rgb(255,0,255)";

      if (JSROOT.browser.isWin) track_width = 1; // not supported on windows

      var npoints = Math.round(track.fNpoints/4),
          buf = new Float32Array((npoints-1)*6),
          pos = 0, projv = this.options.projectPos,
          projx = (this.options.project === "x"),
          projy = (this.options.project === "y"),
          projz = (this.options.project === "z");

      for (var k=0;k<npoints-1;++k) {
         buf[pos]   = projx ? projv : track.fPoints[k*4];
         buf[pos+1] = projy ? projv : track.fPoints[k*4+1];
         buf[pos+2] = projz ? projv : track.fPoints[k*4+2];
         buf[pos+3] = projx ? projv : track.fPoints[k*4+4];
         buf[pos+4] = projy ? projv : track.fPoints[k*4+5];
         buf[pos+5] = projz ? projv : track.fPoints[k*4+6];
         pos+=6;
      }

      var lineMaterial = new THREE.LineBasicMaterial({ color: track_color, linewidth: track_width }),
          line = JSROOT.Painter.createLineSegments(buf, lineMaterial);

      line.geo_name = itemname;
      line.geo_object = track;
      line.hightlightWidthScale = 2;

      this.getExtrasContainer().add(line);

      return true;
   }

   TGeoPainter.prototype.drawEveTrack = function(track, itemname) {
      if (!track || (track.fN <= 0)) return false;

      var track_width = track.fLineWidth || 1,
          track_color = JSROOT.Painter.root_colors[track.fLineColor] || "rgb(255,0,255)";

      if (JSROOT.browser.isWin) track_width = 1; // not supported on windows

      var buf = new Float32Array((track.fN-1)*6), pos = 0,
          projv = this.options.projectPos,
          projx = (this.options.project === "x"),
          projy = (this.options.project === "y"),
          projz = (this.options.project === "z");

      for (var k=0;k<track.fN-1;++k) {
         buf[pos]   = projx ? projv : track.fP[k*3];
         buf[pos+1] = projy ? projv : track.fP[k*3+1];
         buf[pos+2] = projz ? projv : track.fP[k*3+2];
         buf[pos+3] = projx ? projv : track.fP[k*3+3];
         buf[pos+4] = projy ? projv : track.fP[k*3+4];
         buf[pos+5] = projz ? projv : track.fP[k*3+5];
         pos+=6;
      }

      var lineMaterial = new THREE.LineBasicMaterial({ color: track_color, linewidth: track_width }),
          line = JSROOT.Painter.createLineSegments(buf, lineMaterial);

      line.geo_name = itemname;
      line.geo_object = track;
      line.hightlightWidthScale = 2;

      this.getExtrasContainer().add(line);

      return true;
   }

   /** Drawing different hits types like TPolyMarker3d
    * @private */
   TGeoPainter.prototype.drawHit = function(hit, itemname) {
      if (!hit || !hit.fN || (hit.fN < 0)) return false;

      // make hit size scaling factor of overall geometry size
      // otherwise it is not possible to correctly see hits at all
      var hit_size = hit.fMarkerSize * this.getOverallSize() * 0.005;
      if (hit_size <= 0) hit_size = 1;

      var size = hit.fN,
          projv = this.options.projectPos,
          projx = (this.options.project === "x"),
          projy = (this.options.project === "y"),
          projz = (this.options.project === "z"),
          pnts = new JSROOT.Painter.PointsCreator(size, this._webgl, hit_size);

      for (var i=0;i<size;i++)
         pnts.AddPoint(projx ? projv : hit.fP[i*3],
                       projy ? projv : hit.fP[i*3+1],
                       projz ? projv : hit.fP[i*3+2]);

      var mesh = pnts.CreatePoints(JSROOT.Painter.root_colors[hit.fMarkerColor] || "rgb(0,0,255)");

      mesh.highlightScale = 2;

      mesh.geo_name = itemname;
      mesh.geo_object = hit;

      this.getExtrasContainer().add(mesh);

      return true;
   }

   TGeoPainter.prototype.drawExtraShape = function(obj, itemname) {
      var toplevel = JSROOT.GEO.build(obj);
      if (!toplevel) return false;

      toplevel.geo_name = itemname;
      toplevel.geo_object = obj;

      this.getExtrasContainer().add(toplevel);
      return true;
   }

   TGeoPainter.prototype.FindNodeWithVolume = function(name, action, prnt, itemname, volumes) {

      var first_level = false, res = null;

      if (!prnt) {
         prnt = this.GetGeometry();
         if (!prnt && (JSROOT.GEO.NodeKind(prnt)!==0)) return null;
         itemname = this.geo_manager ? prnt.fName : "";
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

   TGeoPainter.prototype.SetRootDefaultColors = function() {
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


   TGeoPainter.prototype.checkScript = function(script_name, call_back) {

      var painter = this, draw_obj = this.GetGeometry(), name_prefix = "";

      if (this.geo_manager) name_prefix = draw_obj.fName;

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

      JSROOT.NewHttpRequest(script_name, "text", function(res) {
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

      }).send();
   }

   /** Assign clones, created outside.
    * Used by geometry painter, where clones are handled by the server */
   TGeoPainter.prototype.assignClones = function(clones) {
      this._clones_owner = true;
      this._clones = clones;
   }

   TGeoPainter.prototype.prepareObjectDraw = function(draw_obj, name_prefix) {

      if (name_prefix == "__geom_viewer_append__") {
         this._new_append_nodes = draw_obj;
         this.options.use_worker = 0;
         this._geom_viewer = true; // indicate that working with geom viewer
      } else if ((name_prefix == "__geom_viewer_selection__") && this._clones) {
         // these are selection done from geom viewer
         this._new_draw_nodes = draw_obj;
         this.options.use_worker = 0;
         this._geom_viewer = true; // indicate that working with geom viewer
      } else if (this._main_painter) {

         this._clones_owner = false;

         this._clones = this._main_painter._clones;

         console.log('Reuse clones', this._clones.nodes.length, 'from main painter');

      } else if (!draw_obj) {

         this._clones_owner = false;

         this._clones = null;

      } else {

         this._start_drawing_time = new Date().getTime();

         this._clones_owner = true;

         this._clones = new JSROOT.GEO.ClonedNodes(draw_obj);

         this._clones.name_prefix = name_prefix;

         var uniquevis = this._clones.MarkVisisble(true);
         if (uniquevis <= 0)
            uniquevis = this._clones.MarkVisisble(false);
         else
            uniquevis = this._clones.MarkVisisble(true, true); // copy bits once and use normal visibility bits

         var spent = new Date().getTime() - this._start_drawing_time;

         console.log('Creating clones', this._clones.nodes.length, 'takes', spent, 'uniquevis', uniquevis);

         if (this.options._count)
            return this.drawCount(uniquevis, spent);
      }

      if (!this._scene) {

         // this is limit for the visible faces, number of volumes does not matter
         this.options.maxlimit = (this._webgl ? 200000 : 100000) * this.options.more;

         this._first_drawing = true;

         // activate worker
         if (this.options.use_worker > 0) this.startWorker();

         var size = this.size_for_3d(this._usesvg ? 3 : undefined);

         this._fit_main_area = (size.can3d === -1);

         var dom = this.createScene(size.width, size.height);

         this.add_3d_canvas(size, dom);

         // set top painter only when first child exists
         this.AccessTopPainter(true);
      }

      this.CreateToolbar();

      if (this._clones) {
         this.showDrawInfo("Drawing geometry");
         this.startDrawGeometry(true);
      } else {
         this.completeDraw();
      }
   }

   TGeoPainter.prototype.showDrawInfo = function(msg) {
      // methods show info when first geometry drawing is performed

      if (!this._first_drawing || !this._start_drawing_time) return;

      var main = this._renderer.domElement.parentNode,
          info = d3.select(main).select(".geo_info");

      if (!msg) {
         info.remove();
      } else {
         var spent = (new Date().getTime() - this._start_drawing_time)*1e-3;
         if (info.empty()) info = d3.select(main).append("p").attr("class","geo_info");
         info.html(msg + ", " + spent.toFixed(1) + "s");
      }

   }

   TGeoPainter.prototype.continueDraw = function() {

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
         if ((res === true) && (now - tm0 < interval)) continue;

         if ((now - tm0 > interval) || (res === 1) || (res === 2)) {

            JSROOT.progress(this.drawing_log);

            this.showDrawInfo(this.drawing_log);

            if (this._first_drawing && this._webgl && (this._num_meshes - this._last_render_meshes > 100) && (now - this._last_render_tm > 2.5*interval)) {
               this.adjustCameraPosition();
               this.Render3D(-1);
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
         this.showDrawInfo("Rendering");
         return setTimeout(this.completeDraw.bind(this, true), 10);
      }

      this.completeDraw(true);
   }

   TGeoPainter.prototype.TestCameraPosition = function(force) {

      this._camera.updateMatrixWorld();
      var origin = this._camera.position.clone();

      if (!force && this._last_camera_position) {
         // if camera position does not changed a lot, ignore such change
         var dist = this._last_camera_position.distanceTo(origin);
         if (dist < (this._overall_size || 1000)/1e4) return;
      }

      this._last_camera_position = origin; // remember current camera position

      if (!this.options.project && this._webgl)
         JSROOT.GEO.produceRenderOrder(this._toplevel, origin, this.options.depthMethod, this._clones);
   }

   /** @brief Call 3D rendering of the geometry
     * @param tmout - specifies delay, after which actual rendering will be invoked
     * Timeout used to avoid multiple rendering of the picture when several 3D drawings
     * superimposed with each other. If tmeout<=0, rendering performed immediately
     * Several special values are used:
     *   -2222 - rendering performed only if there were previous calls, which causes timeout activation
     *   -1    - force recheck of rendering order based on camera position */

   TGeoPainter.prototype.Render3D = function(tmout, measure) {

      if (!this._renderer) {
         console.warn('renderer object not exists - check code');
         return;
      }

      if (tmout === undefined) tmout = 5; // by default, rendering happens with timeout

      if ((tmout <= 0) || this._usesvg) {
         if ('render_tmout' in this) {
            clearTimeout(this.render_tmout);
         } else {
            if (tmout === -2222) return; // special case to check if rendering timeout was active
         }

         var tm1 = new Date();

         if (typeof this.TestAxisVisibility === 'function')
            this.TestAxisVisibility(this._camera, this._toplevel);

         this.TestCameraPosition(tmout === -1);

         // its needed for outlinePass - do rendering, most consuming time
         if (this._effectComposer.passes.length > 1 || this._webgl && this._enableSSAO && this._ssaoPass) {
            this._effectComposer.render();
         } else {
       //     this._renderer.logarithmicDepthBuffer = true;
            this._renderer.render(this._scene, this._camera);
         }

         var tm2 = new Date();

         this.last_render_tm = tm2.getTime();

         delete this.render_tmout;

         if ((this.first_render_tm === 0) && measure) {
            this.first_render_tm = tm2.getTime() - tm1.getTime();
            JSROOT.console('First render tm = ' + this.first_render_tm);
         }

         return JSROOT.Painter.AfterRender3D(this._renderer);
      }

      // do not shoot timeout many times
      if (!this.render_tmout)
         this.render_tmout = setTimeout(this.Render3D.bind(this,0,measure), tmout);
   }


   TGeoPainter.prototype.startWorker = function() {

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

   TGeoPainter.prototype.canSubmitToWorker = function(force) {
      if (!this._worker) return false;

      return this._worker_ready && ((this._worker_jobs == 0) || force);
   }

   TGeoPainter.prototype.submitToWorker = function(job) {
      if (!this._worker) return false;

      this._worker_jobs++;

      job.tm0 = new Date().getTime();

      this._worker.postMessage(job);
   }

   TGeoPainter.prototype.processWorkerReply = function(job) {
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

   TGeoPainter.prototype.testGeomChanges = function() {
      if (this._main_painter) {
         console.warn('Get testGeomChanges call for slave painter');
         return this._main_painter.testGeomChanges();
      }
      this.startDrawGeometry();
      for (var k=0;k<this._slave_painters.length;++k)
         this._slave_painters[k].startDrawGeometry();
   }

   TGeoPainter.prototype.drawSimpleAxis = function() {

      var box = this.getGeomBoundingBox(this._toplevel);

      this.getExtrasContainer('delete', 'axis');
      var container = this.getExtrasContainer('create', 'axis');

      var text_size = 0.02 * Math.max( (box.max.x - box.min.x), (box.max.y - box.min.y), (box.max.z - box.min.z)),
          center = [0,0,0],
          names = ['x','y','z'],
          labels = ['X','Y','Z'],
          colors = ["red","green","blue"],
          ortho = this.options.ortho_camera,
          yup = [this.options._yup, this.options._yup, this.options._yup],
          numaxis = 3;

      if (this.options._axis_center)
         for (var naxis=0;naxis<3;++naxis) {
            var name = names[naxis];
            if ((box.min[name]<=0) && (box.max[name]>=0)) continue;
            center[naxis] = (box.min[name] + box.max[name])/2;
         }

      // only two dimensions are seen by ortho camera, X draws Z, can be configured better later
      if (this.options.ortho_camera) {
         numaxis = 2;
         labels[0] = labels[2];
         colors[0] = colors[2];
         yup[0] = yup[2];
         ortho = true;
      }

      for (var naxis=0;naxis<numaxis;++naxis) {

         var buf = new Float32Array(6), axiscol = colors[naxis], name = names[naxis];

         function Convert(value) {
            var range = box.max[name] - box.min[name];
            if (range<2) return value.toFixed(3);
            if (Math.abs(value)>1e5) return value.toExponential(3);
            return Math.round(value).toString();
         }

         var lbl = Convert(box.max[name]);

         buf[0] = box.min.x;
         buf[1] = box.min.y;
         buf[2] = box.min.z;

         buf[3] = box.min.x;
         buf[4] = box.min.y;
         buf[5] = box.min.z;

         switch (naxis) {
           case 0: buf[3] = box.max.x; if (yup[0] && !ortho) lbl = labels[0] + " " + lbl; else lbl += " " + labels[0]; break;
           case 1: buf[4] = box.max.y; if (yup[1]) lbl += " " + labels[1]; else lbl = labels[1] + " " + lbl; break;
           case 2: buf[5] = box.max.z; lbl += " " + labels[2]; break;
         }

         if (this.options._axis_center)
            for (var k=0;k<6;++k)
               if ((k % 3) !== naxis) buf[k] = center[k%3];

         var lineMaterial = new THREE.LineBasicMaterial({ color: axiscol }),
             mesh = JSROOT.Painter.createLineSegments(buf, lineMaterial);

         container.add(mesh);

         var textMaterial = new THREE.MeshBasicMaterial({ color: axiscol });

         if ((center[naxis]===0) && (center[naxis]>=box.min[name]) && (center[naxis]<=box.max[name]))
           if (!this.options._axis_center || (naxis===0)) {
               var geom = ortho ? new THREE.CircleBufferGeometry(text_size*0.25) :
                                  new THREE.SphereBufferGeometry(text_size*0.25);
               mesh = new THREE.Mesh(geom, textMaterial);
               mesh.translateX((naxis===0) ? center[0] : buf[0]);
               mesh.translateY((naxis===1) ? center[1] : buf[1]);
               mesh.translateZ((naxis===2) ? center[2] : buf[2]);
               container.add(mesh);
           }

         var text3d = new THREE.TextGeometry(lbl, { font: JSROOT.threejs_font_helvetiker_regular, size: text_size, height: 0, curveSegments: 5 });
         mesh = new THREE.Mesh(text3d, textMaterial);
         var textbox = new THREE.Box3().setFromObject(mesh);

         mesh.translateX(buf[3]);
         mesh.translateY(buf[4]);
         mesh.translateZ(buf[5]);

         if (yup[naxis]) {
            switch (naxis) {
               case 0:
                  if (!ortho) {
                     mesh.rotateY(Math.PI);
                     mesh.translateX(-textbox.max.x-text_size*0.5);
                  } else {
                     mesh.translateX(text_size*0.5);
                  }
                  mesh.translateY(-textbox.max.y/2);
                  break;
               case 1:
                  if (!ortho) {
                     mesh.rotateX(-Math.PI/2);
                     mesh.rotateY(-Math.PI/2);
                  } else {
                     mesh.rotateZ(Math.PI/2);
                  }
                  mesh.translateX(text_size*0.5);
                  mesh.translateY(-textbox.max.y/2);
                  break;
               case 2: mesh.rotateY(-Math.PI/2); mesh.translateX(text_size*0.5); mesh.translateY(-textbox.max.y/2); break;
           }
         } else {
            switch (naxis) {
               case 0: mesh.rotateX(Math.PI/2); mesh.translateY(-textbox.max.y/2); mesh.translateX(text_size*0.5); break;
               case 1: mesh.rotateX(Math.PI/2); mesh.rotateY(-Math.PI/2); mesh.translateX(-textbox.max.x-text_size*0.5); mesh.translateY(-textbox.max.y/2); break;
               case 2: mesh.rotateX(Math.PI/2); mesh.rotateZ(Math.PI/2); mesh.translateX(text_size*0.5); mesh.translateY(-textbox.max.y/2); break;
            }
         }

         container.add(mesh);

         text3d = new THREE.TextGeometry(Convert(box.min[name]), { font: JSROOT.threejs_font_helvetiker_regular, size: text_size, height: 0, curveSegments: 5 });

         mesh = new THREE.Mesh(text3d, textMaterial);
         textbox = new THREE.Box3().setFromObject(mesh);

         mesh.translateX(buf[0]);
         mesh.translateY(buf[1]);
         mesh.translateZ(buf[2]);

         if (yup[naxis]) {
            switch (naxis) {
               case 0:
                  if (!ortho) {
                     mesh.rotateY(Math.PI);
                     mesh.translateX(text_size*0.5);
                  } else {
                     mesh.translateX(-textbox.max.x-text_size*0.5);
                  }
                  mesh.translateY(-textbox.max.y/2);
                  break;
               case 1:
                  if (!ortho) {
                     mesh.rotateX(-Math.PI/2);
                     mesh.rotateY(-Math.PI/2);
                  } else {
                     mesh.rotateZ(Math.PI/2);
                  }
                  mesh.translateY(-textbox.max.y/2);
                  mesh.translateX(-textbox.max.x-text_size*0.5);
                  break;
               case 2: mesh.rotateY(-Math.PI/2);  mesh.translateX(-textbox.max.x-text_size*0.5); mesh.translateY(-textbox.max.y/2); break;
            }
         } else {
            switch (naxis) {
               case 0: mesh.rotateX(Math.PI/2); mesh.translateX(-textbox.max.x-text_size*0.5); mesh.translateY(-textbox.max.y/2); break;
               case 1: mesh.rotateX(Math.PI/2); mesh.rotateY(-Math.PI/2); mesh.translateY(-textbox.max.y/2); mesh.translateX(text_size*0.5); break;
               case 2: mesh.rotateX(Math.PI/2); mesh.rotateZ(Math.PI/2);  mesh.translateX(-textbox.max.x-text_size*0.5); mesh.translateY(-textbox.max.y/2); break;
            }
         }

         container.add(mesh);
      }

      this.TestAxisVisibility = function(camera, toplevel) {
         if (!camera) {
            this.getExtrasContainer('delete', 'axis');
            delete this.TestAxisVisibility;
            this.Render3D();
            return;
         }
      }
   }

   TGeoPainter.prototype.toggleAxisDraw = function(force_draw) {
      if (this.TestAxisVisibility) {
         if (!force_draw)
           this.TestAxisVisibility(null, this._toplevel);
      } else {
         this.drawSimpleAxis();
      }
   }

   TGeoPainter.prototype.completeDraw = function(close_progress) {

      var first_time = false, check_extras = true;

      if (!this.options) {
         console.warn('options object does not exist in completeDraw - something went wrong');
         return;
      }

      if (!this._clones) {
         check_extras = false;
         // if extra object where append, redraw them at the end
         this.getExtrasContainer("delete"); // delete old container
         var extras = (this._main_painter ? this._main_painter._extraObjects : null) || this._extraObjects;
         this.drawExtras(extras, "", false);
      }

      if (this._first_drawing) {
         this.adjustCameraPosition(true);
         this.showDrawInfo();
         this._first_drawing = false;
         first_time = true;

         if (this._webgl) {
            this.enableX = this.options.clipx;
            this.enableY = this.options.clipy;
            this.enableZ = this.options.clipz;
         }
         if (this.options.tracks && this.geo_manager && this.geo_manager.fTracks)
            this.addExtra(this.geo_manager.fTracks, "<prnt>/Tracks");
      }

      if (this.options.transparency!==0)
         this.changeGlobalTransparency(this.options.transparency, true);

      if (first_time) {
         this.completeScene();
         if (this.options._axis) this.toggleAxisDraw(true);
      }

      this._scene.overrideMaterial = null;

      if (this._provided_more_nodes !== undefined) {
         this.appendMoreNodes(this._provided_more_nodes, true);
         delete this._provided_more_nodes;
      }

      if (check_extras) {
         // if extra object where append, redraw them at the end
         this.getExtrasContainer("delete"); // delete old container
         var extras = (this._main_painter ? this._main_painter._extraObjects : null) || this._extraObjects;
         this.drawExtras(extras, "", false);
      }

      this.updateClipping(true); // do not render

      this.Render3D(0, true);

      if (close_progress) JSROOT.progress();

      this.addOrbitControls();

      this.addTransformControl();

      if (first_time) {

         // after first draw check if highlight can be enabled
         if (this.options.highlight === false)
            this.options.highlight = (this.first_render_tm < 1000);

         // also highlight of scene object can be assigned at the first draw
         if (this.options.highlight_scene === false)
            this.options.highlight_scene = this.options.highlight;

         // if rotation was enabled, do it
         if (this._webgl && this.options.autoRotate && !this.options.project) this.autorotate(2.5);
         if (!this._usesvg && this.options.show_controls && !JSROOT.BatchMode) this.showControlOptions(true);
      }

      this.DrawingReady();

      if (this._draw_nodes_again)
         return this.startDrawGeometry(); // relaunch drawing

      this._drawing_ready = true; // indicate that drawing is completed
   }

   /** Remove already drawn node. Used by geom viewer */
   TGeoPainter.prototype.RemoveDrawnNode = function(nodeid) {
      if (!this._draw_nodes) return;

      var new_nodes = [];

      for (var n = 0; n < this._draw_nodes.length; ++n) {
         var entry = this._draw_nodes[n];
         if ((entry.nodeid === nodeid) || (this._clones.IsNodeInStack(nodeid, entry.stack))) {
            this._clones.CreateObject3D(entry.stack, this._toplevel, 'delete_mesh');
         } else {
            new_nodes.push(entry);
         }
      }

      if (new_nodes.length < this._draw_nodes.length) {
         this._draw_nodes = new_nodes;
         this.Render3D();
      }
   }

   TGeoPainter.prototype.Cleanup = function(first_time) {

      if (!first_time) {

         this.AccessTopPainter(false); // remove as pointer

         this.helpText();

         JSROOT.Painter.DisposeThreejsObject(this._scene);

         JSROOT.Painter.DisposeThreejsObject(this._full_geom);

         if (this._tcontrols)
            this._tcontrols.dispose();

         if (this._controls)
            this._controls.Cleanup();

         if (this._context_menu)
            this._renderer.domElement.removeEventListener( 'contextmenu', this._context_menu, false );

         if (this._datgui)
            this._datgui.destroy();

         if (this._worker) this._worker.terminate();

         JSROOT.TObjectPainter.prototype.Cleanup.call(this);

         delete this.options;

         delete this._animating;

         var obj = this.GetGeometry();
         if (obj && this.options.is_main) {
            if (obj.$geo_painter===this) delete obj.$geo_painter; else
            if (obj.fVolume && obj.fVolume.$geo_painter===this) delete obj.fVolume.$geo_painter;
         }

         if (this._main_painter) {
            var pos = this._main_painter._slave_painters.indexOf(this);
            if (pos>=0) this._main_painter._slave_painters.splice(pos,1);
         }

         for (var k=0;k<this._slave_painters.length;++k) {
            var slave = this._slave_painters[k];
            if (slave && (slave._main_painter===this)) slave._main_painter = null;
         }
      }

      for (var k in this._slave_painters) {
         var slave = this._slave_painters[k];
         slave._main_painter = null;
         if (slave._clones === this._clones) slave._clones = null;
      }

      this._main_painter = null;
      this._slave_painters = [];

      if (this._renderer) {
         if (this._renderer.dispose) this._renderer.dispose();
         if (this._renderer.context) delete this._renderer.context;
      }

      delete this._scene;
      this._scene_width = 0;
      this._scene_height = 0;
      this._renderer = null;
      this._toplevel = null;
      this._full_geom = null;
      this._camera = null;
      this._selected_mesh = null;

      if (this._clones && this._clones_owner) this._clones.Cleanup(this._draw_nodes, this._build_shapes);
      delete this._clones;
      delete this._clones_owner;
      delete this._draw_nodes;
      delete this._drawing_ready;
      delete this._build_shapes;
      delete this._new_draw_nodes;
      delete this._new_append_nodes;
      delete this._last_camera_position;

      this.first_render_tm = 0; // time needed for first rendering
      this.last_render_tm = 0;

      this.drawing_stage = 0;
      delete this.drawing_log;

      delete this._datgui;
      delete this._controls;
      delete this._context_menu;
      delete this._tcontrols;
      delete this._toolbar;

      delete this._worker;
   }

   TGeoPainter.prototype.helpText = function(msg) {
      JSROOT.progress(msg);
   }

   TGeoPainter.prototype.CheckResize = function(arg) {
      var pad_painter = this.canv_painter();

      // firefox is the only browser which correctly supports resize of embedded canvas,
      // for others we should force canvas redrawing at every step
      if (pad_painter)
         if (!pad_painter.CheckCanvasResize(arg)) return false;

      var sz = this.size_for_3d();

      if ((this._scene_width === sz.width) && (this._scene_height === sz.height)) return false;
      if ((sz.width<10) || (sz.height<10)) return false;

      this._scene_width = sz.width;
      this._scene_height = sz.height;

      if (this._camera && this._renderer) {
         if (this._camera.type == "PerspectiveCamera")
            this._camera.aspect = this._scene_width / this._scene_height;
         this._camera.updateProjectionMatrix();
         this._renderer.setSize( this._scene_width, this._scene_height, !this._fit_main_area );
         this._effectComposer.setSize( this._scene_width, this._scene_height );

         if (!this.drawing_stage) this.Render3D();
      }

      return true;
   }

   TGeoPainter.prototype.ToggleEnlarge = function() {

      if (d3.event) {
         d3.event.preventDefault();
         d3.event.stopPropagation();
      }

      if (this.enlarge_main('toggle'))
        this.CheckResize();
   }


   TGeoPainter.prototype.ownedByTransformControls = function(child) {
      var obj = child.parent;
      while (obj && !(obj instanceof THREE.TransformControls) ) {
         obj = obj.parent;
      }
      return (obj && (obj instanceof THREE.TransformControls));
   }

   TGeoPainter.prototype.accessObjectWireFrame = function(obj, on) {
      // either change mesh wireframe or return current value
      // return undefined when wireframe cannot be accessed

      if (!obj.hasOwnProperty("material") || (obj instanceof THREE.GridHelper)) return;

      if (this.ownedByTransformControls(obj)) return;

      if ((on !== undefined) && obj.stack)
         obj.material.wireframe = on;

      return obj.material.wireframe;
   }


   TGeoPainter.prototype.changeWireFrame = function(obj, on) {
      var painter = this;

      obj.traverse(function(obj2) { painter.accessObjectWireFrame(obj2, on); });

      this.Render3D();
   }

   JSROOT.Painter.CreateGeoPainter = function(divid, obj, opt) {
      JSROOT.GEO.GradPerSegm = JSROOT.gStyle.GeoGradPerSegm;
      JSROOT.GEO.CompressComp = JSROOT.gStyle.GeoCompressComp;

      var painter = new TGeoPainter(obj);

      // one could use TGeoManager setting, but for some example JSROOT does not build composites
      // if (obj && obj._typename=='TGeoManager' && (obj.fNsegments > 3))
      //   JSROOT.GEO.GradPerSegm = 360/obj.fNsegments;

      painter.SetDivId(divid, 5);

      painter._usesvg = JSROOT.Painter.UseSVGFor3D();

      painter._usesvgimg = !painter._usesvg && JSROOT.BatchMode;

      painter._webgl = !painter._usesvg && JSROOT.Painter.TestWebGL();

      painter.options = painter.decodeOptions(opt);

      return painter;
   }

   JSROOT.Painter.drawGeoObject = function(divid, obj, opt) {
      if (!obj) return null;

      var shape = null, extras = null, extras_path = "";

      if (('fShapeBits' in obj) && ('fShapeId' in obj)) {
         shape = obj; obj = null;
      } else if ((obj._typename === 'TGeoVolumeAssembly') || (obj._typename === 'TGeoVolume')) {
         shape = obj.fShape;
      } else if ((obj._typename === "TEveGeoShapeExtract") || (obj._typename === "ROOT::Experimental::TEveGeoShapeExtract")) {
         shape = obj.fShape;
      } else if (obj._typename === 'TGeoManager') {
         JSROOT.GEO.SetBit(obj.fMasterVolume, JSROOT.GEO.BITS.kVisThis, false);
         shape = obj.fMasterVolume.fShape;
      } else if (obj._typename === 'TGeoOverlap') {
         extras = obj.fMarker; extras_path = "<prnt>/Marker";
         obj = JSROOT.GEO.buildOverlapVolume(obj);
         if (!opt) opt = "wire";
      } else if ('fVolume' in obj) {
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
                   { fTrans: null, fShape: shape, fRGBA: [0, 1, 0, 1], fElements: null, fRnrSelf: true });

      if (!obj) return null;

      var painter = JSROOT.Painter.CreateGeoPainter(divid, obj, opt);

      if (painter.options.is_main && !obj.$geo_painter)
         obj.$geo_painter = painter;

      if (!painter.options.is_main && painter.options.project && obj.$geo_painter) {
         painter._main_painter = obj.$geo_painter;
         painter._main_painter._slave_painters.push(painter);
      }

      if (extras) {
         painter._splitColors = true;
         painter.addExtra(extras, extras_path);
      }

      // this.options.script_name = 'https://root.cern/js/files/geom/geomAlice.C'

      painter.checkScript(painter.options.script_name, painter.prepareObjectDraw.bind(painter));

      return painter;
   }

   /// keep for backwards compatibility
   JSROOT.Painter.drawGeometry = JSROOT.Painter.drawGeoObject;

   // ===============================================================================

   /** Function used to build hierarchy of elements of composite shapes
    * @private */
   JSROOT.GEO.buildCompositeVolume = function(comp, side) {

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

   /** Function used to build hierarchy of elements of overlap object
    * @private */
   JSROOT.GEO.buildOverlapVolume = function(overlap) {

      var vol = JSROOT.Create("TGeoVolume");

      JSROOT.GEO.SetBit(vol, JSROOT.GEO.BITS.kVisDaughters, true);
      vol.$geoh = true; // workaround, let know browser that we are in volumes hierarchy
      vol.fName = "";

      var node1 = JSROOT.Create("TGeoNodeMatrix");
      node1.fName = overlap.fVolume1.fName || "Overlap1";
      node1.fMatrix = overlap.fMatrix1;
      node1.fVolume = overlap.fVolume1;
      // node1.fVolume.fLineColor = 2; // color assigned with _splitColors

      var node2 = JSROOT.Create("TGeoNodeMatrix");
      node2.fName = overlap.fVolume2.fName || "Overlap2";
      node2.fMatrix = overlap.fMatrix2;
      node2.fVolume = overlap.fVolume2;
      // node2.fVolume.fLineColor = 3;  // color assigned with _splitColors

      vol.fNodes = JSROOT.Create("TList");
      vol.fNodes.Add(node1);
      vol.fNodes.Add(node2);

      return vol;
   }

   JSROOT.GEO.provideVisStyle = function(obj) {
      if ((obj._typename === 'TEveGeoShapeExtract') || (obj._typename === 'ROOT::Experimental::TEveGeoShapeExtract'))
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
         _name: name ? name : JSROOT.GEO.ObjectName(obj),
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
      if ((obj._typename == "TEveGeoShapeExtract") || (obj._typename == "ROOT::Experimental::TEveGeoShapeExtract") ) {
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

      if (!sub._name)
         if (typeof node._name === 'string') {
            sub._name = node._name;
            if (sub._name.lastIndexOf("s")===sub._name.length-1)
               sub._name = sub._name.substr(0, sub._name.length-1);
            sub._name += "_" + node._childs.length;
         } else {
            sub._name = "item_" + node._childs.length;
         }

      node._childs.push(sub);

      return sub;
   }

   JSROOT.GEO.createList = function(parent, lst, name, title) {

      if (!lst || !('arr' in lst) || (lst.arr.length==0)) return;

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

         JSROOT.GEO.CheckDuplicates(null, lst.arr);

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
          iseve = ((obj._typename === 'TEveGeoShapeExtract') || (obj._typename === 'ROOT::Experimental::TEveGeoShapeExtract'));

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

      if (hitem._geoobj && (( hitem._geoobj._typename == "TEveGeoShapeExtract") || ( hitem._geoobj._typename == "ROOT::Experimental::TEveGeoShapeExtract"))) {
         hitem._geoobj.fRnrSelf = !hitem._geoobj.fRnrSelf;

         JSROOT.GEO.updateBrowserIcons(hitem._geoobj, hpainter);
         JSROOT.GEO.findItemWithPainter(hitem, 'testGeomChanges');
         return false; // no need to update icon - we did it ourself
      }


      // first check that geo painter assigned with the item
      var drawitem = JSROOT.GEO.findItemWithPainter(hitem);
      if (!drawitem) return false;

      var newstate = drawitem._painter.ExtraObjectVisible(hpainter, hitem, true);

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
      if (hitem._kind == 'ROOT.TEvePointSet') icon = 'img_evepoints'; else
      if (hitem._kind == 'ROOT.TPolyMarker3D') icon = 'img_evepoints';
      if (icon.length>0) {
         var drawitem = JSROOT.GEO.findItemWithPainter(hitem);
         if (drawitem)
            if (drawitem._painter.ExtraObjectVisible(hpainter, hitem))
               icon += " geovis_this";
      }
      return icon;
   }

   JSROOT.GEO.expandObject = function(parent, obj) {
      if (!parent || !obj) return false;

      var isnode = (obj._typename.indexOf('TGeoNode') === 0),
          isvolume = (obj._typename.indexOf('TGeoVolume') === 0),
          ismanager = (obj._typename === 'TGeoManager'),
          iseve = ((obj._typename === 'TEveGeoShapeExtract') || (obj._typename === 'ROOT::Experimental::TEveGeoShapeExtract')),
          isoverlap = (obj._typename === 'TGeoOverlap');

      if (!isnode && !isvolume && !ismanager && !iseve && !isoverlap) return false;

      if (parent._childs) return true;

      if (ismanager) {
         JSROOT.GEO.createList(parent, obj.fMaterials, "Materials", "list of materials");
         JSROOT.GEO.createList(parent, obj.fMedia, "Media", "list of media");
         JSROOT.GEO.createList(parent, obj.fTracks, "Tracks", "list of tracks");
         JSROOT.GEO.createList(parent, obj.fOverlaps, "Overlaps", "list of detected overlaps");

         JSROOT.GEO.SetBit(obj.fMasterVolume, JSROOT.GEO.BITS.kVisThis, false);
         JSROOT.GEO.createItem(parent, obj.fMasterVolume);
         return true;
      }

      if (isoverlap) {
         JSROOT.GEO.createItem(parent, obj.fVolume1);
         JSROOT.GEO.createItem(parent, obj.fVolume2);
         JSROOT.GEO.createItem(parent, obj.fMarker, 'Marker');
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

      JSROOT.GEO.CheckDuplicates(obj, subnodes);

      for (var i=0;i<subnodes.length;++i)
         JSROOT.GEO.createItem(parent, subnodes[i]);

      return true;
   }

   JSROOT.addDrawFunc({ name: "TGeoVolumeAssembly", icon: 'img_geoassembly', func: JSROOT.Painter.drawGeoObject, expand: JSROOT.GEO.expandObject, opt: ";more;all;count" });
   JSROOT.addDrawFunc({ name: "TEvePointSet", icon_get: JSROOT.GEO.getBrowserIcon, icon_click: JSROOT.GEO.browserIconClick });
   JSROOT.addDrawFunc({ name: "TEveTrack", icon_get: JSROOT.GEO.getBrowserIcon, icon_click: JSROOT.GEO.browserIconClick });

   JSROOT.TGeoPainter = TGeoPainter;

   JSROOT.Painter.GeoDrawingControl = GeoDrawingControl;

   return JSROOT.Painter;

}));
