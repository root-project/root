

find_package(PythonInterp)
if(NOT PYTHONINTERP_FOUND)
  message(STATUS "No python interpreter found. Can't setup ClangModules without!")
endif()

if(ClangModules_WithoutClang)
  set(ClangModules_ClanglessArg "--clangless")
endif()

set(ClangModules_IsClang NO)
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
  set(ClangModules_IsClang YES)
endif()

if(PYTHONINTERP_FOUND)
if(ClangModules_WithoutClang OR ClangModules_IsClang)

  cmake_minimum_required(VERSION 3.1)

  set(ClangModules_UNPACK_FOLDER "${CMAKE_BINARY_DIR}")

  function(ClangModules_UnpackFiles)
  #Write the python backend out:
file(WRITE "${ClangModules_UNPACK_FOLDER}/ClangModules.py" "#!/usr/bin/env python

import copy
import json
import subprocess
import re
import shutil
import sys
import os


def eprint(msg):
    sys.stderr.write(str(msg) + \"\\n\")
    sys.stderr.flush()


class InvokResult:
    def __init__(self, output, exit_code):
        self.output = str(output)
        self.exit_code = exit_code


class ParseDepsResult:
    def __init__(self, after, error=False):
        self.after = after
        self.error = error

    def get_values(self):
        assert not self.error
        return self.after


def make_match_pattern(key):
    return re.compile(\"^[\\s]*//[\\s]*\" + key + \":\")


provides_line_pattern = make_match_pattern(\"provides\")
after_line_pattern = make_match_pattern(\"after\")


def parse_value_line(pattern, line):
    depends_on = []
    if pattern.match(line):
        parts = line.split(\":\")
        assert len(parts) >= 2
        after = parts[1]
        deps_parts = after.split()
        for dep in deps_parts:
            stripped = dep.strip()
            if \" \" in stripped:
                return ParseDepsResult(None, True)
            if \"\\t\" in stripped:
                return ParseDepsResult(None, True)
            else:
                depends_on.append(stripped)
        return ParseDepsResult(depends_on)
    else:
        return ParseDepsResult(None, True)


assert parse_value_line(after_line_pattern, \"// after\").error
assert parse_value_line(after_line_pattern, \"/d/ after\").error
assert parse_value_line(after_line_pattern, \"d// after:\").error
assert parse_value_line(
    after_line_pattern, \"// after: bla\").get_values() == [\"bla\"]
assert parse_value_line(
    after_line_pattern, \" // after: bla\").get_values() == [\"bla\"]
assert parse_value_line(
    after_line_pattern, \"//  after: bla\").get_values() == [\"bla\"]
assert parse_value_line(
    after_line_pattern, \"//  after: c++\").get_values() == [\"c++\"]
assert parse_value_line(
    after_line_pattern, \"//  after: c++17\").get_values() == [\"c++17\"]
assert parse_value_line(
    after_line_pattern, \"// after: bla foo\").get_values() == [\"bla\", \"foo\"]
assert parse_value_line(
    after_line_pattern, \"// after: bla  foo\").get_values() == [\"bla\", \"foo\"]
assert parse_value_line(
    after_line_pattern, \"// after: bla \\t foo\").get_values() == [\"bla\", \"foo\"]


class MultipleProvidesError(Exception):
    pass


class NotOneProvideError(Exception):
    pass


class Modulemap:
    def __init__(self, mm_file):
        self.mm_file = mm_file
        self.depends_on = []
        self.provides = None
        self.headers = []
        with open(mm_file, \"r\") as f:
            for line in f:

                rematch = re.search(r\"header\\s+\\\"([^\\\"]+)\\\"\", line)
                if rematch:
                    self.headers.append(rematch.group(1))

                new_deps = parse_value_line(after_line_pattern, line)
                if not new_deps.error:
                    self.depends_on = self.depends_on + new_deps.get_values()
                new_provides = parse_value_line(provides_line_pattern, line)
                if not new_provides.error:
                    if self.provides is not None:
                        raise MultipleProvidesError()
                    if len(new_provides.get_values()) != 1:
                        raise NotOneProvideError()
                    self.provides = new_provides.get_values()[0]
        file_name = os.path.basename(mm_file)
        self.name = os.path.splitext(file_name)[0]

    def can_use_in_dir(self, path):
        for header in self.headers:
            if not os.path.isfile(os.path.sep.join([path, header])):
                eprint(\"  Missing \" + header)
                return False
        return True

    def matches(self, name):
        if self.name == name:
            return True
        if self.provides and self.provides == name:
            return True
        return False

    def __repr__(self):
        return self.name + \".modulemap\"


class ModulemapGraph:
    def __init__(self, modulemaps):
        self.modulemaps = modulemaps
        self.modulemaps.sort(key=lambda x: x.name)
        self.handled = {}
        self.success = {}
        self.providers = {}
        for mm in self.modulemaps:
            self.handled[mm.name] = False
            self.success[mm.name] = False
            if mm.provides:
                if mm.provides in self.providers:
                    self.providers[mm.provides].append(mm)
                else:
                    self.providers[mm.provides] = [mm]

        for mm in self.modulemaps:
            assert mm.name not in self.providers, \\
                \"modulemap name shared name with provided: %s\" % mm.name

    def is_provided(self, prov):
        assert prov in self.providers, \"Unknown prov %s \" % prov
        for p in self.providers[prov]:
            if self.success[p.name]:
                return True
        for p in self.providers[prov]:
            if not self.handled[p.name]:
                return False
        return True

    def requirement_success(self, req):
        if req in self.success:
            return self.success[req]
        assert req in self.providers, \"Unknown req %s \" % req
        for p in self.providers[req]:
            if self.success[p.name]:
                return True
        return False

    def requirement_done(self, req):
        if req in self.handled:
            return self.handled[req]
        return self.is_provided(req)

    def can_test_modulemap(self, mm):
        if self.handled[mm.name]:
            return False
        if mm.provides:
            if self.is_provided(mm.provides):
                return False
        for dep in mm.depends_on:
            if not self.requirement_done(dep):
                return False
        return True

    def mark_modulemap(self, mm, success):
        self.handled[mm.name] = True
        self.success[mm.name] = success

    def get_next_modulemap(self):
        for mm in self.modulemaps:
            if self.can_test_modulemap(mm):
                return mm
        return None


class FileBak:
    def __init__(self, path):
        self.path = path
        try:
            with open(self.path, 'r') as f:
                self.data = f.read()
        except EnvironmentError:
            self.data = None

    def revert(self):
        if self.data is None:
            os.remove(self.path)
        else:
            with open(self.path, 'w') as f:
                f.write(self.data)


class VirtualFileSystem:
    def __init__(self, yaml_file, cache_path):
        self.yaml_file = yaml_file
        self.yaml = {'version': 0, 'roots': []}
        self.roots = self.yaml[\"roots\"]
        self.file_bak = None
        self.cache_path = cache_path
        self.update_yaml()

    def has_target_path(self, target_path):
        for root in self.roots:
            if root[\"name\"] == target_path:
                return True
        return False

    def backup(self, path):
        self.yaml_bak = copy.deepcopy(self.yaml)
        self.file_bak = FileBak(path)

    def revert(self):
        self.yaml = self.yaml_bak
        self.roots = self.yaml[\"roots\"]
        self.file_bak.revert()

    def update_yaml(self):
        with open(self.yaml_file, 'w') as fp:
            json.dump(self.yaml, fp, sort_keys=False, indent=2)

    def make_cache_file(self, target_path):
        target_path = os.path.abspath(target_path)
        target_path = target_path.replace(\"/\", \"_\").replace(\".\", \"_\")
        return os.path.abspath(os.path.join(self.cache_path, target_path))

    def append_file(self, source, target, append):
        open_mode = \"w\"
        if append:
            open_mode = \"a\"
        with open(target, open_mode) as target_file:
            with open(source, \"r\") as source_file:
                target_file.write(source_file.read())

    def mount_file(self, source_file, target_dir,
                   file_name='module.modulemap'):
        cache_file = self.make_cache_file(target_dir)
        if not self.has_target_path(target_dir):
            try:
                os.remove(cache_file)
            except EnvironmentError:
                pass
        self.backup(cache_file)
        self.append_file(source_file, cache_file,
                         self.has_target_path(target_dir))
        if not self.has_target_path(target_dir):
            new_entry = {}
            new_entry[\"name\"] = str(target_dir)
            new_entry[\"type\"] = \"directory\"
            new_entry[\"contents\"] = [
                {'name': file_name, 'type': 'file',
                 'external-contents': str(cache_file)}]
            self.roots.append(new_entry)
            self.update_yaml()


class ClangModules:
    def __init__(self, clang_invok, clangless_mode, modulemap_dirs,
                 extra_inc_dirs, check_only):
        inc_args = \"\"
        for inc_dir in extra_inc_dirs:
            inc_args += \" -I \\\"\" + inc_dir + \"\\\" \"
        self.clang_invok = clang_invok + inc_args
        self.clangless_mode = clangless_mode
        self.check_only = check_only
        self.include_paths = self.calculate_include_paths()
        self.modulemap_dirs = modulemap_dirs
        self.mm_graph = None
        self.pcm_tmp_dir = pcm_tmp_dir
        self.parse_modulemaps()

    def invoke_clang(self, suffix, force_with_clangless=False):
        if self.clangless_mode and not force_with_clangless:
            return InvokResult(\"\", 0)
        out_encoding = sys.stdout.encoding
        if out_encoding is None:
            out_encoding = 'utf-8'
        try:
            output = subprocess.check_output(
                \"LANG=C \" + self.clang_invok + \" \" + suffix,
                stderr=subprocess.STDOUT, shell=True)
            output = output.decode(out_encoding)
        except subprocess.CalledProcessError as exc:
            return InvokResult(exc.output.decode(out_encoding), 1)
        else:
            return InvokResult(output, 0)

    def requirement_success(self, prov):
        return self.mm_graph.requirement_success(prov)

    def get_next_modulemap(self):
        while True:
            mm = self.mm_graph.get_next_modulemap()

            if mm is None:
                return None

            if self.check_only:
                for c in self.check_only:
                    if mm.matches(c):
                        return mm
                m.mm_graph.mark_modulemap(mm, False)
                continue

            return mm

    def parse_modulemaps(self):
        modulemaps = []
        for modulemap_dir in self.modulemap_dirs:
            for filename in os.listdir(modulemap_dir):
                if not filename.endswith(\".modulemap\"):
                    continue
                file_path = os.path.abspath(
                    os.path.sep.join([modulemap_dir, filename]))
                mm = Modulemap(file_path)
                modulemaps.append(mm)
        self.mm_graph = ModulemapGraph(modulemaps)

    def calculate_include_paths(self, make_abs=True):
        includes = []
        output = self.invoke_clang(\"-xc++ -v -E /dev/null\", True)
        # In clangless_mode we can fail when we have a non GCC compatible
        # compiler that doesn't like our invocation above.
        if output.exit_code != 0 and not self.clangless_mode:
            raise NameError(
                'Clang failed with non-zero exit code: ' + str(output.output))
        output = output.output
        in_includes = False
        for line in output.splitlines():
            if in_includes:
                if line.startswith(\" \"):
                    path = line.strip()
                    if make_abs:
                        path = os.path.abspath(path)

                    includes.append(path)
                else:
                    in_includes = False
            if '#include \"...\"' in line or '#include <...>' in line:
                in_includes = True
        return includes

    def create_test_file(self, mm, output):
        with open(output, \"w\") as f:
            for header in mm.headers:
                f.write(\"#include \\\"\" + header + \"\\\"\\n\")
            f.write(\"\\nint main() {}\\n\")


def arg_parse_error(message):
    sys.stderr.write(\"Error: \" + message + \"\\n\")
    exit(3)


# Argument parsing
clang_invok = None
parsing_invocation = False
modulemap_dirs = []
check_only = None
required_modules = None
output_dir = None
parsed_arg = True
extra_inc_dirs = []
clangless_mode = False
vfs_output = None

for i in range(0, len(sys.argv)):
    if parsed_arg:
        parsed_arg = False
        continue
    arg = sys.argv[i]
    if i + 1 < len(sys.argv):
        next_arg = sys.argv[i + 1]
    else:
        next_arg = None

    if parsing_invocation:
        clang_invok += \" \" + arg + \"\"
    else:
        if arg == \"--invocation\":
            parsing_invocation = True
            clang_invok = \"\"
        elif arg == \"--check-only\":
            if not next_arg:
                arg_parse_error(\"No arg supplied for --check-only\")
            if check_only is None:
                check_only = []
            check_only += filter(None, next_arg.split(\";\"))
            parsed_arg = True
        elif arg == \"--required-modules\":
            if not next_arg:
                arg_parse_error(\"No arg supplied for --required-modules\")
            if required_modules is None:
                required_modules = []
            required_modules += filter(None, next_arg.split(\";\"))
            parsed_arg = True
        elif arg == \"--clangless\":
            clangless_mode = True
        elif arg == \"--modulemap-dir\":
            if not next_arg:
                arg_parse_error(\"No arg supplied for --modulemap-dir\")
            for path in next_arg.split(\";\"):
                if len(path) > 0:
                    modulemap_dirs.append(path)
            parsed_arg = True
        elif arg == \"--vfs-output\":
            if not next_arg:
                arg_parse_error(\"No arg supplied for --vfs-output\")
            if next_arg != \"-\":
                vfs_output = next_arg
            parsed_arg = True
        elif arg == \"-I\":
            if not next_arg:
                arg_parse_error(\"No arg supplied for -I\")
            for path in next_arg.split(\":\"):
                if len(path) > 0:
                    extra_inc_dirs.append(path)
            parsed_arg = True
        elif arg == \"--output-dir\":
            if not next_arg:
                arg_parse_error(\"No arg supplied for --output-dir\")
            if output_dir:
                arg_parse_error(
                    \"specified multiple output dirs with --output-dir\")
            output_dir = next_arg
            parsed_arg = True
        else:
            arg_parse_error(\"Unknown arg: \" + arg)

if len(modulemap_dirs) == 0:
    arg_parse_error(\"Not modulemap directories specified with --modulemap-dir\")

if not output_dir:
    arg_parse_error(\"Not output_dir specified with --output-dir\")

if not clang_invok:
    arg_parse_error(\"No clang invocation specified with --invocation [...]\")

if not vfs_output:
    vfs_output = os.path.sep.join([output_dir, \"ClangModulesVFS.yaml\"])

vfs = VirtualFileSystem(vfs_output, output_dir)

pcm_tmp_dir = os.path.sep.join([output_dir, \"ClangModulesPCMs\"])

test_cpp_file = os.path.sep.join([output_dir, \"ClangModules.cpp\"])

m = ClangModules(clang_invok, clangless_mode,
                 modulemap_dirs, extra_inc_dirs, check_only)
# print(m.include_paths)

clang_flags = \" -fmodules -fcxx-modules -Xclang \" + \\
    \"-fmodules-local-submodule-visibility -ivfsoverlay \\\"\" + vfs_output + \"\\\" \"

while True:
    mm = m.get_next_modulemap()
    if mm is None:
        break
    success = False
    eprint(\"Module  \" + mm.name)
    for inc_path in m.include_paths:
        eprint(\" Selecting \" + inc_path)
        if not mm.can_use_in_dir(inc_path):
            continue
        eprint(\" Checking \" + inc_path)
        vfs.mount_file(mm.mm_file, inc_path)
        m.create_test_file(mm, test_cpp_file)
        shutil.rmtree(pcm_tmp_dir, True)
        invoke_result = m.invoke_clang(\"-fmodules-cache-path=\" + pcm_tmp_dir +
                                       \" -fsyntax-only -Rmodule-build \" +
                                       clang_flags + test_cpp_file)
        # print(m.mm_graph.providers)
        success = (invoke_result.exit_code == 0)
        if success:
            eprint(\"works \" + mm.name + \" \" + str(inc_path))
            break
        else:
            eprint(\"error:\" + invoke_result.output)
            vfs.revert()
    m.mm_graph.mark_modulemap(mm, success)

if required_modules:
    for mod in required_modules:
        if not m.requirement_success(mod):
            eprint(\"Missing module \" + mod)
            exit(2)

if not clangless_mode:
    print(clang_flags)
")

file(WRITE "${ClangModules_UNPACK_FOLDER}/tinyxml2.modulemap" "module tinyxml2 [system] { header \"tinyxml2.h\" export * }
")

file(WRITE "${ClangModules_UNPACK_FOLDER}/glog.modulemap" "// after: stl
module glog [system] {
  module logging { header \"glog/logging.h\" export * }
}
")

file(WRITE "${ClangModules_UNPACK_FOLDER}/eigen3_min.modulemap" "// provides: eigen3
// after: eigen3_big
module eigen3 [system] {
  module \"Cholesky\" { header \"Eigen/Cholesky\" export * }
  module \"Core\" { header \"Eigen/Core\" export * }
  module \"Dense\" { header \"Eigen/Dense\" export * }
  module \"Eigen\" { header \"Eigen/Eigen\" export * }
  module \"Eigenvalues\" { header \"Eigen/Eigenvalues\" export * }
  module \"Geometry\" { header \"Eigen/Geometry\" export * }
  module \"Householder\" { header \"Eigen/Householder\" export * }
  module \"IterativeLinearSolvers\" { header \"Eigen/IterativeLinearSolvers\" export * }
  module \"Jacobi\" { header \"Eigen/Jacobi\" export * }
  module \"LU\" { header \"Eigen/LU\" export * }
  module \"OrderingMethods\" { header \"Eigen/OrderingMethods\" export * }
  module \"QR\" { header \"Eigen/QR\" export * }
  module \"SVD\" { header \"Eigen/SVD\" export * }
  module \"Sparse\" { header \"Eigen/Sparse\" export * }
  module \"SparseCholesky\" { header \"Eigen/SparseCholesky\" export * }
  module \"SparseCore\" { header \"Eigen/SparseCore\" export * }
  module \"SparseLU\" { header \"Eigen/SparseLU\" export * }
  module \"SparseQR\" { header \"Eigen/SparseQR\" export * }
  module \"StdDeque\" { header \"Eigen/StdDeque\" export * }
  module \"StdList\" { header \"Eigen/StdList\" export * }
  module \"StdVector\" { header \"Eigen/StdVector\" export * }
}
")

file(WRITE "${ClangModules_UNPACK_FOLDER}/bullet_new.modulemap" "// provides: bullet
// after: stl
module bullet [system] {
  module \"BulletCollision/NarrowPhaseCollision/btVoronoiSimplexSolver.h\" { header \"BulletCollision/NarrowPhaseCollision/btVoronoiSimplexSolver.h\" export * }
  module \"Bullet2FileLoader/autogenerated/bullet2.h\" { header \"Bullet2FileLoader/autogenerated/bullet2.h\" export * }
  module \"Bullet3OpenCL/RigidBody/kernels/updateAabbsKernel.h\" { header \"Bullet3OpenCL/RigidBody/kernels/updateAabbsKernel.h\" export * }
  module \"BulletCollision/BroadphaseCollision/btAxisSweep3.h\" { header \"BulletCollision/BroadphaseCollision/btAxisSweep3.h\" export * }
  module \"BulletCollision/BroadphaseCollision/btBroadphaseInterface.h\" { header \"BulletCollision/BroadphaseCollision/btBroadphaseInterface.h\" export * }
  module \"BulletCollision/BroadphaseCollision/btBroadphaseProxy.h\" { header \"BulletCollision/BroadphaseCollision/btBroadphaseProxy.h\" export * }
  module \"BulletCollision/BroadphaseCollision/btCollisionAlgorithm.h\" { header \"BulletCollision/BroadphaseCollision/btCollisionAlgorithm.h\" export * }
  module \"BulletCollision/BroadphaseCollision/btDbvt.h\" { header \"BulletCollision/BroadphaseCollision/btDbvt.h\" export * }
  module \"BulletCollision/BroadphaseCollision/btDbvtBroadphase.h\" { header \"BulletCollision/BroadphaseCollision/btDbvtBroadphase.h\" export * }
  module \"BulletCollision/BroadphaseCollision/btDispatcher.h\" { header \"BulletCollision/BroadphaseCollision/btDispatcher.h\" export * }
  module \"BulletCollision/BroadphaseCollision/btOverlappingPairCache.h\" { header \"BulletCollision/BroadphaseCollision/btOverlappingPairCache.h\" export * }
  module \"BulletCollision/BroadphaseCollision/btQuantizedBvh.h\" { header \"BulletCollision/BroadphaseCollision/btQuantizedBvh.h\" export * }
  module \"BulletCollision/BroadphaseCollision/btSimpleBroadphase.h\" { header \"BulletCollision/BroadphaseCollision/btSimpleBroadphase.h\" export * }
  module \"BulletCollision/CollisionDispatch/SphereTriangleDetector.h\" { header \"BulletCollision/CollisionDispatch/SphereTriangleDetector.h\" export * }
  module \"BulletCollision/CollisionDispatch/btActivatingCollisionAlgorithm.h\" { header \"BulletCollision/CollisionDispatch/btActivatingCollisionAlgorithm.h\" export * }
  module \"BulletCollision/CollisionDispatch/btBox2dBox2dCollisionAlgorithm.h\" { header \"BulletCollision/CollisionDispatch/btBox2dBox2dCollisionAlgorithm.h\" export * }
  module \"BulletCollision/CollisionDispatch/btBoxBoxCollisionAlgorithm.h\" { header \"BulletCollision/CollisionDispatch/btBoxBoxCollisionAlgorithm.h\" export * }
  module \"BulletCollision/CollisionDispatch/btBoxBoxDetector.h\" { header \"BulletCollision/CollisionDispatch/btBoxBoxDetector.h\" export * }
  module \"BulletCollision/CollisionDispatch/btCollisionConfiguration.h\" { header \"BulletCollision/CollisionDispatch/btCollisionConfiguration.h\" export * }
  module \"BulletCollision/CollisionDispatch/btCollisionCreateFunc.h\" { header \"BulletCollision/CollisionDispatch/btCollisionCreateFunc.h\" export * }
  module \"BulletCollision/CollisionDispatch/btCollisionDispatcher.h\" { header \"BulletCollision/CollisionDispatch/btCollisionDispatcher.h\" export * }
  module \"BulletCollision/CollisionDispatch/btCollisionObject.h\" { header \"BulletCollision/CollisionDispatch/btCollisionObject.h\" export * }
  module \"BulletCollision/CollisionDispatch/btCollisionObjectWrapper.h\" { header \"BulletCollision/CollisionDispatch/btCollisionObjectWrapper.h\" export * }
  module \"BulletCollision/CollisionDispatch/btCollisionWorld.h\" { header \"BulletCollision/CollisionDispatch/btCollisionWorld.h\" export * }
  module \"BulletCollision/CollisionDispatch/btCollisionWorldImporter.h\" { header \"BulletCollision/CollisionDispatch/btCollisionWorldImporter.h\" export * }
  module \"BulletCollision/CollisionDispatch/btCompoundCollisionAlgorithm.h\" { header \"BulletCollision/CollisionDispatch/btCompoundCollisionAlgorithm.h\" export * }
  module \"BulletCollision/CollisionDispatch/btCompoundCompoundCollisionAlgorithm.h\" { header \"BulletCollision/CollisionDispatch/btCompoundCompoundCollisionAlgorithm.h\" export * }
  module \"BulletCollision/CollisionDispatch/btConvex2dConvex2dAlgorithm.h\" { header \"BulletCollision/CollisionDispatch/btConvex2dConvex2dAlgorithm.h\" export * }
  module \"BulletCollision/CollisionDispatch/btConvexConcaveCollisionAlgorithm.h\" { header \"BulletCollision/CollisionDispatch/btConvexConcaveCollisionAlgorithm.h\" export * }
  module \"BulletCollision/CollisionDispatch/btConvexConvexAlgorithm.h\" { header \"BulletCollision/CollisionDispatch/btConvexConvexAlgorithm.h\" export * }
  module \"BulletCollision/CollisionDispatch/btConvexPlaneCollisionAlgorithm.h\" { header \"BulletCollision/CollisionDispatch/btConvexPlaneCollisionAlgorithm.h\" export * }
  module \"BulletCollision/CollisionDispatch/btDefaultCollisionConfiguration.h\" { header \"BulletCollision/CollisionDispatch/btDefaultCollisionConfiguration.h\" export * }
  module \"BulletCollision/CollisionDispatch/btEmptyCollisionAlgorithm.h\" { header \"BulletCollision/CollisionDispatch/btEmptyCollisionAlgorithm.h\" export * }
  module \"BulletCollision/CollisionDispatch/btGhostObject.h\" { header \"BulletCollision/CollisionDispatch/btGhostObject.h\" export * }
  module \"BulletCollision/CollisionDispatch/btHashedSimplePairCache.h\" { header \"BulletCollision/CollisionDispatch/btHashedSimplePairCache.h\" export * }
  module \"BulletCollision/CollisionDispatch/btInternalEdgeUtility.h\" { header \"BulletCollision/CollisionDispatch/btInternalEdgeUtility.h\" export * }
  module \"BulletCollision/CollisionDispatch/btManifoldResult.h\" { header \"BulletCollision/CollisionDispatch/btManifoldResult.h\" export * }
  module \"BulletCollision/CollisionDispatch/btSimulationIslandManager.h\" { header \"BulletCollision/CollisionDispatch/btSimulationIslandManager.h\" export * }
  module \"BulletCollision/CollisionDispatch/btSphereBoxCollisionAlgorithm.h\" { header \"BulletCollision/CollisionDispatch/btSphereBoxCollisionAlgorithm.h\" export * }
  module \"BulletCollision/CollisionDispatch/btSphereSphereCollisionAlgorithm.h\" { header \"BulletCollision/CollisionDispatch/btSphereSphereCollisionAlgorithm.h\" export * }
  module \"BulletCollision/CollisionDispatch/btSphereTriangleCollisionAlgorithm.h\" { header \"BulletCollision/CollisionDispatch/btSphereTriangleCollisionAlgorithm.h\" export * }
  module \"BulletCollision/CollisionDispatch/btUnionFind.h\" { header \"BulletCollision/CollisionDispatch/btUnionFind.h\" export * }
  module \"BulletCollision/CollisionShapes/btBox2dShape.h\" { header \"BulletCollision/CollisionShapes/btBox2dShape.h\" export * }
  module \"BulletCollision/CollisionShapes/btBoxShape.h\" { header \"BulletCollision/CollisionShapes/btBoxShape.h\" export * }
  module \"BulletCollision/CollisionShapes/btBvhTriangleMeshShape.h\" { header \"BulletCollision/CollisionShapes/btBvhTriangleMeshShape.h\" export * }
  module \"BulletCollision/CollisionShapes/btCapsuleShape.h\" { header \"BulletCollision/CollisionShapes/btCapsuleShape.h\" export * }
  module \"BulletCollision/CollisionShapes/btCollisionMargin.h\" { header \"BulletCollision/CollisionShapes/btCollisionMargin.h\" export * }
  module \"BulletCollision/CollisionShapes/btCollisionShape.h\" { header \"BulletCollision/CollisionShapes/btCollisionShape.h\" export * }
  module \"BulletCollision/CollisionShapes/btCompoundShape.h\" { header \"BulletCollision/CollisionShapes/btCompoundShape.h\" export * }
  module \"BulletCollision/CollisionShapes/btConcaveShape.h\" { header \"BulletCollision/CollisionShapes/btConcaveShape.h\" export * }
  module \"BulletCollision/CollisionShapes/btConeShape.h\" { header \"BulletCollision/CollisionShapes/btConeShape.h\" export * }
  module \"BulletCollision/CollisionShapes/btConvex2dShape.h\" { header \"BulletCollision/CollisionShapes/btConvex2dShape.h\" export * }
  module \"BulletCollision/CollisionShapes/btConvexHullShape.h\" { header \"BulletCollision/CollisionShapes/btConvexHullShape.h\" export * }
  module \"BulletCollision/CollisionShapes/btConvexInternalShape.h\" { header \"BulletCollision/CollisionShapes/btConvexInternalShape.h\" export * }
  module \"BulletCollision/CollisionShapes/btConvexPointCloudShape.h\" { header \"BulletCollision/CollisionShapes/btConvexPointCloudShape.h\" export * }
  module \"BulletCollision/CollisionShapes/btConvexPolyhedron.h\" { header \"BulletCollision/CollisionShapes/btConvexPolyhedron.h\" export * }
  module \"BulletCollision/CollisionShapes/btConvexShape.h\" { header \"BulletCollision/CollisionShapes/btConvexShape.h\" export * }
  module \"BulletCollision/CollisionShapes/btConvexTriangleMeshShape.h\" { header \"BulletCollision/CollisionShapes/btConvexTriangleMeshShape.h\" export * }
  module \"BulletCollision/CollisionShapes/btCylinderShape.h\" { header \"BulletCollision/CollisionShapes/btCylinderShape.h\" export * }
  module \"BulletCollision/CollisionShapes/btEmptyShape.h\" { header \"BulletCollision/CollisionShapes/btEmptyShape.h\" export * }
  module \"BulletCollision/CollisionShapes/btHeightfieldTerrainShape.h\" { header \"BulletCollision/CollisionShapes/btHeightfieldTerrainShape.h\" export * }
  module \"BulletCollision/CollisionShapes/btMinkowskiSumShape.h\" { header \"BulletCollision/CollisionShapes/btMinkowskiSumShape.h\" export * }
  module \"BulletCollision/CollisionShapes/btMultiSphereShape.h\" { header \"BulletCollision/CollisionShapes/btMultiSphereShape.h\" export * }
  module \"BulletCollision/CollisionShapes/btMultimaterialTriangleMeshShape.h\" { header \"BulletCollision/CollisionShapes/btMultimaterialTriangleMeshShape.h\" export * }
  module \"BulletCollision/CollisionShapes/btOptimizedBvh.h\" { header \"BulletCollision/CollisionShapes/btOptimizedBvh.h\" export * }
  module \"BulletCollision/CollisionShapes/btPolyhedralConvexShape.h\" { header \"BulletCollision/CollisionShapes/btPolyhedralConvexShape.h\" export * }
  module \"BulletCollision/CollisionShapes/btScaledBvhTriangleMeshShape.h\" { header \"BulletCollision/CollisionShapes/btScaledBvhTriangleMeshShape.h\" export * }
  module \"BulletCollision/CollisionShapes/btShapeHull.h\" { header \"BulletCollision/CollisionShapes/btShapeHull.h\" export * }
  module \"BulletCollision/CollisionShapes/btSphereShape.h\" { header \"BulletCollision/CollisionShapes/btSphereShape.h\" export * }
  module \"BulletCollision/CollisionShapes/btStaticPlaneShape.h\" { header \"BulletCollision/CollisionShapes/btStaticPlaneShape.h\" export * }
  module \"BulletCollision/CollisionShapes/btStridingMeshInterface.h\" { header \"BulletCollision/CollisionShapes/btStridingMeshInterface.h\" export * }
  module \"BulletCollision/CollisionShapes/btTetrahedronShape.h\" { header \"BulletCollision/CollisionShapes/btTetrahedronShape.h\" export * }
  module \"BulletCollision/CollisionShapes/btTriangleBuffer.h\" { header \"BulletCollision/CollisionShapes/btTriangleBuffer.h\" export * }
  module \"BulletCollision/CollisionShapes/btTriangleCallback.h\" { header \"BulletCollision/CollisionShapes/btTriangleCallback.h\" export * }
  module \"BulletCollision/CollisionShapes/btTriangleIndexVertexArray.h\" { header \"BulletCollision/CollisionShapes/btTriangleIndexVertexArray.h\" export * }
  module \"BulletCollision/CollisionShapes/btTriangleIndexVertexMaterialArray.h\" { header \"BulletCollision/CollisionShapes/btTriangleIndexVertexMaterialArray.h\" export * }
  module \"BulletCollision/CollisionShapes/btTriangleInfoMap.h\" { header \"BulletCollision/CollisionShapes/btTriangleInfoMap.h\" export * }
  module \"BulletCollision/CollisionShapes/btTriangleMesh.h\" { header \"BulletCollision/CollisionShapes/btTriangleMesh.h\" export * }
  module \"BulletCollision/CollisionShapes/btTriangleMeshShape.h\" { header \"BulletCollision/CollisionShapes/btTriangleMeshShape.h\" export * }
  module \"BulletCollision/CollisionShapes/btTriangleShape.h\" { header \"BulletCollision/CollisionShapes/btTriangleShape.h\" export * }
  module \"BulletCollision/CollisionShapes/btUniformScalingShape.h\" { header \"BulletCollision/CollisionShapes/btUniformScalingShape.h\" export * }
  module \"BulletCollision/Gimpact/btBoxCollision.h\" { header \"BulletCollision/Gimpact/btBoxCollision.h\" export * }
  module \"BulletCollision/Gimpact/btClipPolygon.h\" { header \"BulletCollision/Gimpact/btClipPolygon.h\" export * }
  module \"BulletCollision/Gimpact/btCompoundFromGimpact.h\" { header \"BulletCollision/Gimpact/btCompoundFromGimpact.h\" export * }
  module \"BulletCollision/Gimpact/btContactProcessing.h\" { header \"BulletCollision/Gimpact/btContactProcessing.h\" export * }
  module \"BulletCollision/Gimpact/btGImpactBvh.h\" { header \"BulletCollision/Gimpact/btGImpactBvh.h\" export * }
  module \"BulletCollision/Gimpact/btGImpactCollisionAlgorithm.h\" { header \"BulletCollision/Gimpact/btGImpactCollisionAlgorithm.h\" export * }
  module \"BulletCollision/Gimpact/btGImpactMassUtil.h\" { header \"BulletCollision/Gimpact/btGImpactMassUtil.h\" export * }
  module \"BulletCollision/Gimpact/btGImpactQuantizedBvh.h\" { header \"BulletCollision/Gimpact/btGImpactQuantizedBvh.h\" export * }
  module \"BulletCollision/Gimpact/btGImpactShape.h\" { header \"BulletCollision/Gimpact/btGImpactShape.h\" export * }
  module \"BulletCollision/Gimpact/btGenericPoolAllocator.h\" { header \"BulletCollision/Gimpact/btGenericPoolAllocator.h\" export * }
  module \"BulletCollision/Gimpact/btQuantization.h\" { header \"BulletCollision/Gimpact/btQuantization.h\" export * }
  module \"BulletCollision/Gimpact/btTriangleShapeEx.h\" { header \"BulletCollision/Gimpact/btTriangleShapeEx.h\" export * }
  module \"BulletCollision/Gimpact/gim_array.h\" { header \"BulletCollision/Gimpact/gim_array.h\" export * }
  module \"BulletCollision/Gimpact/gim_basic_geometry_operations.h\" { header \"BulletCollision/Gimpact/gim_basic_geometry_operations.h\" export * }
  module \"BulletCollision/Gimpact/gim_bitset.h\" { header \"BulletCollision/Gimpact/gim_bitset.h\" export * }
  module \"BulletCollision/Gimpact/gim_box_collision.h\" { header \"BulletCollision/Gimpact/gim_box_collision.h\" export * }
  module \"BulletCollision/Gimpact/gim_box_set.h\" { header \"BulletCollision/Gimpact/gim_box_set.h\" export * }
  module \"BulletCollision/Gimpact/gim_contact.h\" { header \"BulletCollision/Gimpact/gim_contact.h\" export * }
  module \"BulletCollision/Gimpact/gim_geom_types.h\" { header \"BulletCollision/Gimpact/gim_geom_types.h\" export * }
  module \"BulletCollision/Gimpact/gim_geometry.h\" { header \"BulletCollision/Gimpact/gim_geometry.h\" export * }
  module \"BulletCollision/Gimpact/gim_linear_math.h\" { header \"BulletCollision/Gimpact/gim_linear_math.h\" export * }
  module \"BulletCollision/Gimpact/gim_math.h\" { header \"BulletCollision/Gimpact/gim_math.h\" export * }
  module \"BulletCollision/Gimpact/gim_memory.h\" { header \"BulletCollision/Gimpact/gim_memory.h\" export * }
  module \"BulletCollision/Gimpact/gim_radixsort.h\" { header \"BulletCollision/Gimpact/gim_radixsort.h\" export * }
  module \"BulletCollision/Gimpact/gim_tri_collision.h\" { header \"BulletCollision/Gimpact/gim_tri_collision.h\" export * }
  module \"BulletCollision/NarrowPhaseCollision/btComputeGjkEpaPenetration.h\" { header \"BulletCollision/NarrowPhaseCollision/btComputeGjkEpaPenetration.h\" export * }
  module \"BulletCollision/NarrowPhaseCollision/btContinuousConvexCollision.h\" { header \"BulletCollision/NarrowPhaseCollision/btContinuousConvexCollision.h\" export * }
  module \"BulletCollision/NarrowPhaseCollision/btConvexCast.h\" { header \"BulletCollision/NarrowPhaseCollision/btConvexCast.h\" export * }
  module \"BulletCollision/NarrowPhaseCollision/btConvexPenetrationDepthSolver.h\" { header \"BulletCollision/NarrowPhaseCollision/btConvexPenetrationDepthSolver.h\" export * }
  module \"BulletCollision/NarrowPhaseCollision/btDiscreteCollisionDetectorInterface.h\" { header \"BulletCollision/NarrowPhaseCollision/btDiscreteCollisionDetectorInterface.h\" export * }
  module \"BulletCollision/NarrowPhaseCollision/btGjkCollisionDescription.h\" { header \"BulletCollision/NarrowPhaseCollision/btGjkCollisionDescription.h\" export * }
  module \"BulletCollision/NarrowPhaseCollision/btGjkConvexCast.h\" { header \"BulletCollision/NarrowPhaseCollision/btGjkConvexCast.h\" export * }
  module \"BulletCollision/NarrowPhaseCollision/btGjkEpa2.h\" { header \"BulletCollision/NarrowPhaseCollision/btGjkEpa2.h\" export * }
  module \"BulletCollision/NarrowPhaseCollision/btGjkEpa3.h\" { header \"BulletCollision/NarrowPhaseCollision/btGjkEpa3.h\" export * }
  module \"BulletCollision/NarrowPhaseCollision/btGjkEpaPenetrationDepthSolver.h\" { header \"BulletCollision/NarrowPhaseCollision/btGjkEpaPenetrationDepthSolver.h\" export * }
  module \"BulletCollision/NarrowPhaseCollision/btGjkPairDetector.h\" { header \"BulletCollision/NarrowPhaseCollision/btGjkPairDetector.h\" export * }
  module \"BulletCollision/NarrowPhaseCollision/btManifoldPoint.h\" { header \"BulletCollision/NarrowPhaseCollision/btManifoldPoint.h\" export * }
  module \"BulletCollision/NarrowPhaseCollision/btMinkowskiPenetrationDepthSolver.h\" { header \"BulletCollision/NarrowPhaseCollision/btMinkowskiPenetrationDepthSolver.h\" export * }
  module \"BulletCollision/NarrowPhaseCollision/btMprPenetration.h\" { header \"BulletCollision/NarrowPhaseCollision/btMprPenetration.h\" export * }
  module \"BulletCollision/NarrowPhaseCollision/btPersistentManifold.h\" { header \"BulletCollision/NarrowPhaseCollision/btPersistentManifold.h\" export * }
  module \"BulletCollision/NarrowPhaseCollision/btPointCollector.h\" { header \"BulletCollision/NarrowPhaseCollision/btPointCollector.h\" export * }
  module \"BulletCollision/NarrowPhaseCollision/btPolyhedralContactClipping.h\" { header \"BulletCollision/NarrowPhaseCollision/btPolyhedralContactClipping.h\" export * }
  module \"BulletCollision/NarrowPhaseCollision/btRaycastCallback.h\" { header \"BulletCollision/NarrowPhaseCollision/btRaycastCallback.h\" export * }
  module \"BulletCollision/NarrowPhaseCollision/btSimplexSolverInterface.h\" { header \"BulletCollision/NarrowPhaseCollision/btSimplexSolverInterface.h\" export * }
  module \"BulletCollision/NarrowPhaseCollision/btSubSimplexConvexCast.h\" { header \"BulletCollision/NarrowPhaseCollision/btSubSimplexConvexCast.h\" export * }
  module \"BulletCollision/btBulletCollisionCommon.h\" { header \"BulletCollision/btBulletCollisionCommon.h\" export * }
  module \"BulletDynamics/Character/btCharacterControllerInterface.h\" { header \"BulletDynamics/Character/btCharacterControllerInterface.h\" export * }
  module \"BulletDynamics/Character/btKinematicCharacterController.h\" { header \"BulletDynamics/Character/btKinematicCharacterController.h\" export * }
  module \"BulletDynamics/ConstraintSolver/btConeTwistConstraint.h\" { header \"BulletDynamics/ConstraintSolver/btConeTwistConstraint.h\" export * }
  module \"BulletDynamics/ConstraintSolver/btConstraintSolver.h\" { header \"BulletDynamics/ConstraintSolver/btConstraintSolver.h\" export * }
  module \"BulletDynamics/ConstraintSolver/btContactConstraint.h\" { header \"BulletDynamics/ConstraintSolver/btContactConstraint.h\" export * }
  module \"BulletDynamics/ConstraintSolver/btContactSolverInfo.h\" { header \"BulletDynamics/ConstraintSolver/btContactSolverInfo.h\" export * }
  module \"BulletDynamics/ConstraintSolver/btFixedConstraint.h\" { header \"BulletDynamics/ConstraintSolver/btFixedConstraint.h\" export * }
  module \"BulletDynamics/ConstraintSolver/btGearConstraint.h\" { header \"BulletDynamics/ConstraintSolver/btGearConstraint.h\" export * }
  module \"BulletDynamics/ConstraintSolver/btGeneric6DofConstraint.h\" { header \"BulletDynamics/ConstraintSolver/btGeneric6DofConstraint.h\" export * }
  module \"BulletDynamics/ConstraintSolver/btGeneric6DofSpring2Constraint.h\" { header \"BulletDynamics/ConstraintSolver/btGeneric6DofSpring2Constraint.h\" export * }
  module \"BulletDynamics/ConstraintSolver/btGeneric6DofSpringConstraint.h\" { header \"BulletDynamics/ConstraintSolver/btGeneric6DofSpringConstraint.h\" export * }
  module \"BulletDynamics/ConstraintSolver/btHinge2Constraint.h\" { header \"BulletDynamics/ConstraintSolver/btHinge2Constraint.h\" export * }
  module \"BulletDynamics/ConstraintSolver/btHingeConstraint.h\" { header \"BulletDynamics/ConstraintSolver/btHingeConstraint.h\" export * }
  module \"BulletDynamics/ConstraintSolver/btJacobianEntry.h\" { header \"BulletDynamics/ConstraintSolver/btJacobianEntry.h\" export * }
  module \"BulletDynamics/ConstraintSolver/btNNCGConstraintSolver.h\" { header \"BulletDynamics/ConstraintSolver/btNNCGConstraintSolver.h\" export * }
  module \"BulletDynamics/ConstraintSolver/btPoint2PointConstraint.h\" { header \"BulletDynamics/ConstraintSolver/btPoint2PointConstraint.h\" export * }
  module \"BulletDynamics/ConstraintSolver/btSequentialImpulseConstraintSolver.h\" { header \"BulletDynamics/ConstraintSolver/btSequentialImpulseConstraintSolver.h\" export * }
  module \"BulletDynamics/ConstraintSolver/btSliderConstraint.h\" { header \"BulletDynamics/ConstraintSolver/btSliderConstraint.h\" export * }
  module \"BulletDynamics/ConstraintSolver/btSolve2LinearConstraint.h\" { header \"BulletDynamics/ConstraintSolver/btSolve2LinearConstraint.h\" export * }
  module \"BulletDynamics/ConstraintSolver/btSolverBody.h\" { header \"BulletDynamics/ConstraintSolver/btSolverBody.h\" export * }
  module \"BulletDynamics/ConstraintSolver/btSolverConstraint.h\" { header \"BulletDynamics/ConstraintSolver/btSolverConstraint.h\" export * }
  module \"BulletDynamics/ConstraintSolver/btTypedConstraint.h\" { header \"BulletDynamics/ConstraintSolver/btTypedConstraint.h\" export * }
  module \"BulletDynamics/ConstraintSolver/btUniversalConstraint.h\" { header \"BulletDynamics/ConstraintSolver/btUniversalConstraint.h\" export * }
  module \"BulletDynamics/Dynamics/btActionInterface.h\" { header \"BulletDynamics/Dynamics/btActionInterface.h\" export * }
  module \"BulletDynamics/Dynamics/btDiscreteDynamicsWorld.h\" { header \"BulletDynamics/Dynamics/btDiscreteDynamicsWorld.h\" export * }
  module \"BulletDynamics/Dynamics/btDiscreteDynamicsWorldMt.h\" { header \"BulletDynamics/Dynamics/btDiscreteDynamicsWorldMt.h\" export * }
  module \"BulletDynamics/Dynamics/btDynamicsWorld.h\" { header \"BulletDynamics/Dynamics/btDynamicsWorld.h\" export * }
  module \"BulletDynamics/Dynamics/btRigidBody.h\" { header \"BulletDynamics/Dynamics/btRigidBody.h\" export * }
  module \"BulletDynamics/Dynamics/btSimpleDynamicsWorld.h\" { header \"BulletDynamics/Dynamics/btSimpleDynamicsWorld.h\" export * }
  module \"BulletDynamics/Dynamics/btSimulationIslandManagerMt.h\" { header \"BulletDynamics/Dynamics/btSimulationIslandManagerMt.h\" export * }
  module \"BulletDynamics/Featherstone/btMultiBody.h\" { header \"BulletDynamics/Featherstone/btMultiBody.h\" export * }
  module \"BulletDynamics/Featherstone/btMultiBodyConstraint.h\" { header \"BulletDynamics/Featherstone/btMultiBodyConstraint.h\" export * }
  module \"BulletDynamics/Featherstone/btMultiBodyConstraintSolver.h\" { header \"BulletDynamics/Featherstone/btMultiBodyConstraintSolver.h\" export * }
  module \"BulletDynamics/Featherstone/btMultiBodyDynamicsWorld.h\" { header \"BulletDynamics/Featherstone/btMultiBodyDynamicsWorld.h\" export * }
  module \"BulletDynamics/Featherstone/btMultiBodyFixedConstraint.h\" { header \"BulletDynamics/Featherstone/btMultiBodyFixedConstraint.h\" export * }
  module \"BulletDynamics/Featherstone/btMultiBodyJointFeedback.h\" { header \"BulletDynamics/Featherstone/btMultiBodyJointFeedback.h\" export * }
  module \"BulletDynamics/Featherstone/btMultiBodyJointLimitConstraint.h\" { header \"BulletDynamics/Featherstone/btMultiBodyJointLimitConstraint.h\" export * }
  module \"BulletDynamics/Featherstone/btMultiBodyJointMotor.h\" { header \"BulletDynamics/Featherstone/btMultiBodyJointMotor.h\" export * }
  module \"BulletDynamics/Featherstone/btMultiBodyLink.h\" { header \"BulletDynamics/Featherstone/btMultiBodyLink.h\" export * }
  module \"BulletDynamics/Featherstone/btMultiBodyLinkCollider.h\" { header \"BulletDynamics/Featherstone/btMultiBodyLinkCollider.h\" export * }
  module \"BulletDynamics/Featherstone/btMultiBodyPoint2Point.h\" { header \"BulletDynamics/Featherstone/btMultiBodyPoint2Point.h\" export * }
  module \"BulletDynamics/Featherstone/btMultiBodySliderConstraint.h\" { header \"BulletDynamics/Featherstone/btMultiBodySliderConstraint.h\" export * }
  module \"BulletDynamics/Featherstone/btMultiBodySolverConstraint.h\" { header \"BulletDynamics/Featherstone/btMultiBodySolverConstraint.h\" export * }
  module \"BulletDynamics/MLCPSolvers/btDantzigLCP.h\" { header \"BulletDynamics/MLCPSolvers/btDantzigLCP.h\" export * }
  module \"BulletDynamics/MLCPSolvers/btDantzigSolver.h\" { header \"BulletDynamics/MLCPSolvers/btDantzigSolver.h\" export * }
  module \"BulletDynamics/MLCPSolvers/btLemkeAlgorithm.h\" { header \"BulletDynamics/MLCPSolvers/btLemkeAlgorithm.h\" export * }
  module \"BulletDynamics/MLCPSolvers/btLemkeSolver.h\" { header \"BulletDynamics/MLCPSolvers/btLemkeSolver.h\" export * }
  module \"BulletDynamics/MLCPSolvers/btMLCPSolver.h\" { header \"BulletDynamics/MLCPSolvers/btMLCPSolver.h\" export * }
  module \"BulletDynamics/MLCPSolvers/btMLCPSolverInterface.h\" { header \"BulletDynamics/MLCPSolvers/btMLCPSolverInterface.h\" export * }
  module \"BulletDynamics/MLCPSolvers/btPATHSolver.h\" { header \"BulletDynamics/MLCPSolvers/btPATHSolver.h\" export * }
  module \"BulletDynamics/MLCPSolvers/btSolveProjectedGaussSeidel.h\" { header \"BulletDynamics/MLCPSolvers/btSolveProjectedGaussSeidel.h\" export * }
  module \"BulletDynamics/Vehicle/btRaycastVehicle.h\" { header \"BulletDynamics/Vehicle/btRaycastVehicle.h\" export * }
  module \"BulletDynamics/Vehicle/btVehicleRaycaster.h\" { header \"BulletDynamics/Vehicle/btVehicleRaycaster.h\" export * }
  module \"BulletDynamics/Vehicle/btWheelInfo.h\" { header \"BulletDynamics/Vehicle/btWheelInfo.h\" export * }
  module \"BulletDynamics/btBulletDynamicsCommon.h\" { header \"BulletDynamics/btBulletDynamicsCommon.h\" export * }
  module \"BulletFileLoader/autogenerated/bullet.h\" { header \"BulletFileLoader/autogenerated/bullet.h\" export * }
  module \"BulletFileLoader/bChunk.h\" { header \"BulletFileLoader/bChunk.h\" export * }
  module \"BulletFileLoader/bCommon.h\" { header \"BulletFileLoader/bCommon.h\" export * }
  module \"BulletFileLoader/bDNA.h\" { header \"BulletFileLoader/bDNA.h\" export * }
  module \"BulletFileLoader/bDefines.h\" { header \"BulletFileLoader/bDefines.h\" export * }
  module \"BulletFileLoader/bFile.h\" { header \"BulletFileLoader/bFile.h\" export * }
  module \"BulletFileLoader/btBulletFile.h\" { header \"BulletFileLoader/btBulletFile.h\" export * }
  module \"BulletInverseDynamics/btBulletCollisionCommon.h\" { header \"BulletInverseDynamics/btBulletCollisionCommon.h\" export * }
  module \"BulletSoftBody/btDefaultSoftBodySolver.h\" { header \"BulletSoftBody/btDefaultSoftBodySolver.h\" export * }
  module \"BulletSoftBody/btSoftBody.h\" { header \"BulletSoftBody/btSoftBody.h\" export * }
  module \"BulletSoftBody/btSoftBodyConcaveCollisionAlgorithm.h\" { header \"BulletSoftBody/btSoftBodyConcaveCollisionAlgorithm.h\" export * }
  module \"BulletSoftBody/btSoftBodyData.h\" { header \"BulletSoftBody/btSoftBodyData.h\" export * }
  module \"BulletSoftBody/btSoftBodyHelpers.h\" { header \"BulletSoftBody/btSoftBodyHelpers.h\" export * }
  module \"BulletSoftBody/btSoftBodyInternals.h\" { header \"BulletSoftBody/btSoftBodyInternals.h\" export * }
  module \"BulletSoftBody/btSoftBodyRigidBodyCollisionConfiguration.h\" { header \"BulletSoftBody/btSoftBodyRigidBodyCollisionConfiguration.h\" export * }
  module \"BulletSoftBody/btSoftBodySolverVertexBuffer.h\" { header \"BulletSoftBody/btSoftBodySolverVertexBuffer.h\" export * }
  module \"BulletSoftBody/btSoftBodySolvers.h\" { header \"BulletSoftBody/btSoftBodySolvers.h\" export * }
  module \"BulletSoftBody/btSoftMultiBodyDynamicsWorld.h\" { header \"BulletSoftBody/btSoftMultiBodyDynamicsWorld.h\" export * }
  module \"BulletSoftBody/btSoftRigidCollisionAlgorithm.h\" { header \"BulletSoftBody/btSoftRigidCollisionAlgorithm.h\" export * }
  module \"BulletSoftBody/btSoftRigidDynamicsWorld.h\" { header \"BulletSoftBody/btSoftRigidDynamicsWorld.h\" export * }
  module \"BulletSoftBody/btSoftSoftCollisionAlgorithm.h\" { header \"BulletSoftBody/btSoftSoftCollisionAlgorithm.h\" export * }
  module \"BulletSoftBody/btSparseSDF.h\" { header \"BulletSoftBody/btSparseSDF.h\" export * }
  module \"BulletWorldImporter/btBulletWorldImporter.h\" { header \"BulletWorldImporter/btBulletWorldImporter.h\" export * }
  module \"BulletWorldImporter/btWorldImporter.h\" { header \"BulletWorldImporter/btWorldImporter.h\" export * }
  module \"BulletXmlWorldImporter/string_split.h\" { header \"BulletXmlWorldImporter/string_split.h\" export * }
  module \"BulletXmlWorldImporter/tinystr.h\" { header \"BulletXmlWorldImporter/tinystr.h\" export * }
  module \"BulletXmlWorldImporter/tinyxml.h\" { header \"BulletXmlWorldImporter/tinyxml.h\" export * }
  module \"ConvexDecomposition/ConvexBuilder.h\" { header \"ConvexDecomposition/ConvexBuilder.h\" export * }
  module \"ConvexDecomposition/ConvexDecomposition.h\" { header \"ConvexDecomposition/ConvexDecomposition.h\" export * }
  module \"ConvexDecomposition/bestfit.h\" { header \"ConvexDecomposition/bestfit.h\" export * }
  module \"ConvexDecomposition/bestfitobb.h\" { header \"ConvexDecomposition/bestfitobb.h\" export * }
  module \"ConvexDecomposition/cd_hull.h\" { header \"ConvexDecomposition/cd_hull.h\" export * }
  module \"ConvexDecomposition/cd_vector.h\" { header \"ConvexDecomposition/cd_vector.h\" export * }
  module \"ConvexDecomposition/cd_wavefront.h\" { header \"ConvexDecomposition/cd_wavefront.h\" export * }
  module \"ConvexDecomposition/concavity.h\" { header \"ConvexDecomposition/concavity.h\" export * }
  module \"ConvexDecomposition/fitsphere.h\" { header \"ConvexDecomposition/fitsphere.h\" export * }
  module \"ConvexDecomposition/float_math.h\" { header \"ConvexDecomposition/float_math.h\" export * }
  module \"ConvexDecomposition/meshvolume.h\" { header \"ConvexDecomposition/meshvolume.h\" export * }
  module \"ConvexDecomposition/planetri.h\" { header \"ConvexDecomposition/planetri.h\" export * }
  module \"ConvexDecomposition/raytri.h\" { header \"ConvexDecomposition/raytri.h\" export * }
  module \"ConvexDecomposition/splitplane.h\" { header \"ConvexDecomposition/splitplane.h\" export * }
  module \"ConvexDecomposition/vlookup.h\" { header \"ConvexDecomposition/vlookup.h\" export * }
  module \"GIMPACTUtils/btGImpactConvexDecompositionShape.h\" { header \"GIMPACTUtils/btGImpactConvexDecompositionShape.h\" export * }
  module \"HACD/hacdCircularList.h\" { header \"HACD/hacdCircularList.h\" export * }
  module \"HACD/hacdGraph.h\" { header \"HACD/hacdGraph.h\" export * }
  module \"HACD/hacdHACD.h\" { header \"HACD/hacdHACD.h\" export * }
  module \"HACD/hacdICHull.h\" { header \"HACD/hacdICHull.h\" export * }
  module \"HACD/hacdManifoldMesh.h\" { header \"HACD/hacdManifoldMesh.h\" export * }
  module \"HACD/hacdVector.h\" { header \"HACD/hacdVector.h\" export * }
  module \"HACD/hacdVersion.h\" { header \"HACD/hacdVersion.h\" export * }
  module \"LinearMath/btAabbUtil2.h\" { header \"LinearMath/btAabbUtil2.h\" export * }
  module \"LinearMath/btAlignedAllocator.h\" { header \"LinearMath/btAlignedAllocator.h\" export * }
  module \"LinearMath/btAlignedObjectArray.h\" { header \"LinearMath/btAlignedObjectArray.h\" export * }
  module \"LinearMath/btConvexHull.h\" { header \"LinearMath/btConvexHull.h\" export * }
  module \"LinearMath/btConvexHullComputer.h\" { header \"LinearMath/btConvexHullComputer.h\" export * }
  module \"LinearMath/btCpuFeatureUtility.h\" { header \"LinearMath/btCpuFeatureUtility.h\" export * }
  module \"LinearMath/btDefaultMotionState.h\" { header \"LinearMath/btDefaultMotionState.h\" export * }
  module \"LinearMath/btGeometryUtil.h\" { header \"LinearMath/btGeometryUtil.h\" export * }
  module \"LinearMath/btGrahamScan2dConvexHull.h\" { header \"LinearMath/btGrahamScan2dConvexHull.h\" export * }
  module \"LinearMath/btHashMap.h\" { header \"LinearMath/btHashMap.h\" export * }
  module \"LinearMath/btIDebugDraw.h\" { header \"LinearMath/btIDebugDraw.h\" export * }
  module \"LinearMath/btList.h\" { header \"LinearMath/btList.h\" export * }
  module \"LinearMath/btMatrix3x3.h\" { header \"LinearMath/btMatrix3x3.h\" export * }
  module \"LinearMath/btMatrixX.h\" { header \"LinearMath/btMatrixX.h\" export * }
  module \"LinearMath/btMinMax.h\" { header \"LinearMath/btMinMax.h\" export * }
  module \"LinearMath/btMotionState.h\" { header \"LinearMath/btMotionState.h\" export * }
  module \"LinearMath/btPolarDecomposition.h\" { header \"LinearMath/btPolarDecomposition.h\" export * }
  module \"LinearMath/btPoolAllocator.h\" { header \"LinearMath/btPoolAllocator.h\" export * }
  module \"LinearMath/btQuadWord.h\" { header \"LinearMath/btQuadWord.h\" export * }
  module \"LinearMath/btQuaternion.h\" { header \"LinearMath/btQuaternion.h\" export * }
  module \"LinearMath/btQuickprof.h\" { header \"LinearMath/btQuickprof.h\" export * }
  module \"LinearMath/btScalar.h\" { header \"LinearMath/btScalar.h\" export * }
  module \"LinearMath/btSerializer.h\" { header \"LinearMath/btSerializer.h\" export * }
  module \"LinearMath/btSpatialAlgebra.h\" { header \"LinearMath/btSpatialAlgebra.h\" export * }
  module \"LinearMath/btStackAlloc.h\" { header \"LinearMath/btStackAlloc.h\" export * }
  module \"LinearMath/btThreads.h\" { header \"LinearMath/btThreads.h\" export * }
  module \"LinearMath/btTransform.h\" { header \"LinearMath/btTransform.h\" export * }
  module \"LinearMath/btTransformUtil.h\" { header \"LinearMath/btTransformUtil.h\" export * }
  module \"LinearMath/btVector3.h\" { header \"LinearMath/btVector3.h\" export * }
  module \"btBulletCollisionCommon.h\" { header \"btBulletCollisionCommon.h\" export * }
  module \"btBulletDynamicsCommon.h\" { header \"btBulletDynamicsCommon.h\" export * }
}
")

file(WRITE "${ClangModules_UNPACK_FOLDER}/eigen3_big.modulemap" "// provides: eigen3
// after: stl
module eigen3 [system] {
  module \"Array\" { header \"Eigen/Array\" export * }
  module \"Cholesky\" { header \"Eigen/Cholesky\" export * }
  module \"CholmodSupport\" { header \"Eigen/CholmodSupport\" export * }
  module \"Core\" { header \"Eigen/Core\" export * }
  module \"Dense\" { header \"Eigen/Dense\" export * }
  module \"Eigen\" { header \"Eigen/Eigen\" export * }
  module \"Eigen2Support\" { header \"Eigen/Eigen2Support\" export * }
  module \"Eigenvalues\" { header \"Eigen/Eigenvalues\" export * }
  module \"Geometry\" { header \"Eigen/Geometry\" export * }
  module \"Householder\" { header \"Eigen/Householder\" export * }
  module \"IterativeLinearSolvers\" { header \"Eigen/IterativeLinearSolvers\" export * }
  module \"Jacobi\" { header \"Eigen/Jacobi\" export * }
  module \"LU\" { header \"Eigen/LU\" export * }
  module \"LeastSquares\" { header \"Eigen/LeastSquares\" export * }
  module \"MetisSupport\" { header \"Eigen/MetisSupport\" export * }
  module \"OrderingMethods\" { header \"Eigen/OrderingMethods\" export * }
  module \"PaStiXSupport\" { header \"Eigen/PaStiXSupport\" export * }
  module \"PardisoSupport\" { header \"Eigen/PardisoSupport\" export * }
  module \"QR\" { header \"Eigen/QR\" export * }
  module \"SPQRSupport\" { header \"Eigen/SPQRSupport\" export * }
  module \"SVD\" { header \"Eigen/SVD\" export * }
  module \"Sparse\" { header \"Eigen/Sparse\" export * }
  module \"SparseCholesky\" { header \"Eigen/SparseCholesky\" export * }
  module \"SparseCore\" { header \"Eigen/SparseCore\" export * }
  module \"SparseLU\" { header \"Eigen/SparseLU\" export * }
  module \"SparseQR\" { header \"Eigen/SparseQR\" export * }
  module \"StdDeque\" { header \"Eigen/StdDeque\" export * }
  module \"StdList\" { header \"Eigen/StdList\" export * }
  module \"StdVector\" { header \"Eigen/StdVector\" export * }
  module \"SuperLUSupport\" { header \"Eigen/SuperLUSupport\" export * }
  module \"UmfPackSupport\" { header \"Eigen/UmfPackSupport\" export * }
}
")

file(WRITE "${ClangModules_UNPACK_FOLDER}/gtest.modulemap" "// after: stl
module gtest [system] { header \"gtest/gtest.h\" export * }
")

file(WRITE "${ClangModules_UNPACK_FOLDER}/stl14.modulemap" "// after: stl17
// provides: stl
module stl14 [system] {
  module \"algorithm\" { header \"algorithm\" export * } // Algorithms that operate on containers
  module \"array\" { header \"array\" export * } // (since C++11) std::array container
  module \"atomic\" { header \"atomic\" export * } // (since C++11)  Atomic operations library
  module \"bitset\" { header \"bitset\" export * } //  std::bitset class template
  module \"cassert\" { header \"cassert\" export * } // Conditionally compiled macro that compares its argument to zero
  module \"cctype\" { header \"cctype\" export * } //  functions to determine the type contained in character data
  module \"cerrno\" { header \"cerrno\" export * } //  Macro containing the last error number
  module \"cfenv\" { header \"cfenv\" export * } // (since C++11) Floating-point environment access functions
  module \"cfloat\" { header \"cfloat\" export * } //  limits of float types
  module \"chrono\" { header \"chrono\" export * } // (since C++11)  C++ time utilites
  module \"cinttypes\" { header \"cinttypes\" export * } // (since C++11) formatting macros , intmax_t and uintmax_t math and conversions
  module \"climits\" { header \"climits\" export * } // limits of integral types
  module \"clocale\" { header \"clocale\" export * } // C localization utilities
  module \"cmath\" { header \"cmath\" export * } // Common mathematics functions
  module \"codecvt\" { header \"codecvt\" export * } // (since C++11) Unicode conversion facilities
  module \"complex\" { header \"complex\" export * } // Complex number type
  module \"condition_variable\" { header \"condition_variable\" export * } // (since C++11)  thread waiting conditions
  module \"csetjmp\" { header \"csetjmp\" export * } // Macro (and function) that saves (and jumps) to an execution context
  module \"csignal\" { header \"csignal\" export * } // Functions and macro constants for signal management
  module \"cstdarg\" { header \"cstdarg\" export * } // Handling of variable length argument lists
  module \"cstddef\" { header \"cstddef\" export * } // typedefs for types such as size_t, NULL and others
  module \"cstdint\" { header \"cstdint\" export * } // (since C++11) fixed-size types and limits of other types
  module \"cstdio\" { header \"cstdio\" export * } //  C-style input-output functions
  module \"cstdlib\" { header \"cstdlib\" export * } // General purpose utilities: program control, dynamic memory allocation, random numbers, sort and search
  module \"cstring\" { header \"cstring\" export * } // various narrow character string handling functions
  module \"ctime\" { header \"ctime\" export * } // C-style time/date utilites
  module \"cuchar\" { header \"cuchar\" export * } // (since C++11)  C-style Unicode character conversion functions
  module \"cwchar\" { header \"cwchar\" export * } //  various wide and multibyte string handling functions
  module \"cwctype\" { header \"cwctype\" export * } // functions for determining the type of wide character data
  module \"deque\" { header \"deque\" export * } // std::deque container
  module \"exception\" { header \"exception\" export * } // Exception handling utilities
  module \"forward_list\" { header \"forward_list\" export * } // (since C++11)  std::forward_list container
  module \"fstream\" { header \"fstream\" export * } // std::basic_fstream, std::basic_ifstream, std::basic_ofstream class templates and several typedefs
  module \"functional\" { header \"functional\" export * } //  Function objects, designed for use with the standard algorithms
  module \"future\" { header \"future\" export * } // (since C++11)  primitives for asynchronous computations
  module \"initializer_list\" { header \"initializer_list\" export * } // (since C++11)  std::initializer_list class template
  module \"iomanip\" { header \"iomanip\" export * } // Helper functions to control the format or input and output
  module \"iosfwd\" { header \"iosfwd\" export * } //  forward declarations of all classes in the input/output library
  module \"ios\" { header \"ios\" export * } // std::ios_base class, std::basic_ios class template and several typedefs
  module \"iostream\" { header \"iostream\" export * } //  several standard stream objects
  module \"istream\" { header \"istream\" export * } // std::basic_istream class template and several typedefs
  module \"iterator\" { header \"iterator\" export * } //  Container iterators
  module \"limits\" { header \"limits\" export * } //  standardized way to query properties of arithmetic types
  module \"list\" { header \"list\" export * } //  std::list container
  module \"locale\" { header \"locale\" export * } //  Localization utilities
  module \"map\" { header \"map\" export * } // std::map and std::multimap associative containers
  module \"memory\" { header \"memory\" export * } //  Higher level memory management utilities
  module \"mutex\" { header \"mutex\" export * } // (since C++11) mutual exclusion primitives
  module \"new\" { header \"new\" export * } // Low-level memory management utilities
  module \"numeric\" { header \"numeric\" export * } // Numeric operations on values in containers
  module \"ostream\" { header \"ostream\" export * } // std::basic_ostream, std::basic_iostream class templates and several typedefs
  module \"queue\" { header \"queue\" export * } // std::queue and std::priority_queue container adaptors
  module \"random\" { header \"random\" export * } // (since C++11)  Random number generators and distributions
  module \"ratio\" { header \"ratio\" export * } // (since C++11) Compile-time rational arithmetic
  module \"regex\" { header \"regex\" export * } // (since C++11) Classes, algorithms and iterators to support regular expression processing
  module \"scoped_allocator\" { header \"scoped_allocator\" export * } // (since C++11)  Nested allocator class
  module \"set\" { header \"set\" export * } // std::set and std::multiset associative containers
  module \"shared_mutex\" { header \"shared_mutex\" export * } // (since C++14)  shared mutual exclusion primitives
  module \"sstream\" { header \"sstream\" export * } // std::basic_stringstream, std::basic_istringstream, std::basic_ostringstream class templates and several typedefs
  module \"stack\" { header \"stack\" export * } // std::stack container adaptor
  module \"stdexcept\" { header \"stdexcept\" export * } // Standard exception objects
  module \"streambuf\" { header \"streambuf\" export * } // std::basic_streambuf class template
  module \"string\" { header \"string\" export * } //  std::basic_string class template
  module \"system_error\" { header \"system_error\" export * } // (since C++11)  defines std::error_code, a platform-dependent error code
  module \"thread\" { header \"thread\" export * } // (since C++11)  std::thread class and supporting functions
  module \"tuple\" { header \"tuple\" export * } // (since C++11) std::tuple class template
  module \"typeindex\" { header \"typeindex\" export * } // (since C++11) std::type_index
  module \"typeinfo\" { header \"typeinfo\" export * } //  Runtime type information utilities
  module \"type_traits\" { header \"type_traits\" export * } // (since C++11) Compile-time type information
  module \"unordered_map\" { header \"unordered_map\" export * } // (since C++11) std::unordered_map and std::unordered_multimap unordered associative containers
  module \"unordered_set\" { header \"unordered_set\" export * } // (since C++11) std::unordered_set and std::unordered_multiset unordered associative containers
  module \"utility\" { header \"utility\" export * } // Various utility components
  module \"valarray\" { header \"valarray\" export * } //  Class for representing and manipulating arrays of values
  module \"vector\" { header \"vector\" export * } //  std::vector container
}
")

file(WRITE "${ClangModules_UNPACK_FOLDER}/sfml_old.modulemap" "// provides: sfml
// after: sfml_newer
module sfml [system] {
  module \"SFML/Audio.hpp\" { header \"SFML/Audio.hpp\" export * }
  module \"SFML/Audio/AlResource.hpp\" { header \"SFML/Audio/AlResource.hpp\" export * }
  module \"SFML/Audio/Export.hpp\" { header \"SFML/Audio/Export.hpp\" export * }
  module \"SFML/Audio/InputSoundFile.hpp\" { header \"SFML/Audio/InputSoundFile.hpp\" export * }
  module \"SFML/Audio/Listener.hpp\" { header \"SFML/Audio/Listener.hpp\" export * }
  module \"SFML/Audio/Music.hpp\" { header \"SFML/Audio/Music.hpp\" export * }
  module \"SFML/Audio/OutputSoundFile.hpp\" { header \"SFML/Audio/OutputSoundFile.hpp\" export * }
  module \"SFML/Audio/Sound.hpp\" { header \"SFML/Audio/Sound.hpp\" export * }
  module \"SFML/Audio/SoundBuffer.hpp\" { header \"SFML/Audio/SoundBuffer.hpp\" export * }
  module \"SFML/Audio/SoundBufferRecorder.hpp\" { header \"SFML/Audio/SoundBufferRecorder.hpp\" export * }
  module \"SFML/Audio/SoundFileFactory.hpp\" { header \"SFML/Audio/SoundFileFactory.hpp\" export * }
  module \"SFML/Audio/SoundFileReader.hpp\" { header \"SFML/Audio/SoundFileReader.hpp\" export * }
  module \"SFML/Audio/SoundFileWriter.hpp\" { header \"SFML/Audio/SoundFileWriter.hpp\" export * }
  module \"SFML/Audio/SoundRecorder.hpp\" { header \"SFML/Audio/SoundRecorder.hpp\" export * }
  module \"SFML/Audio/SoundSource.hpp\" { header \"SFML/Audio/SoundSource.hpp\" export * }
  module \"SFML/Audio/SoundStream.hpp\" { header \"SFML/Audio/SoundStream.hpp\" export * }
  module \"SFML/Config.hpp\" { header \"SFML/Config.hpp\" export * }
  module \"SFML/Graphics.hpp\" { header \"SFML/Graphics.hpp\" export * }
  module \"SFML/Graphics/BlendMode.hpp\" { header \"SFML/Graphics/BlendMode.hpp\" export * }
  module \"SFML/Graphics/CircleShape.hpp\" { header \"SFML/Graphics/CircleShape.hpp\" export * }
  module \"SFML/Graphics/Color.hpp\" { header \"SFML/Graphics/Color.hpp\" export * }
  module \"SFML/Graphics/ConvexShape.hpp\" { header \"SFML/Graphics/ConvexShape.hpp\" export * }
  module \"SFML/Graphics/Drawable.hpp\" { header \"SFML/Graphics/Drawable.hpp\" export * }
  module \"SFML/Graphics/Export.hpp\" { header \"SFML/Graphics/Export.hpp\" export * }
  module \"SFML/Graphics/Font.hpp\" { header \"SFML/Graphics/Font.hpp\" export * }
  module \"SFML/Graphics/Glsl.hpp\" { header \"SFML/Graphics/Glsl.hpp\" export * }
  module \"SFML/Graphics/Glyph.hpp\" { header \"SFML/Graphics/Glyph.hpp\" export * }
  module \"SFML/Graphics/Image.hpp\" { header \"SFML/Graphics/Image.hpp\" export * }
  module \"SFML/Graphics/PrimitiveType.hpp\" { header \"SFML/Graphics/PrimitiveType.hpp\" export * }
  module \"SFML/Graphics/Rect.hpp\" { header \"SFML/Graphics/Rect.hpp\" export * }
  module \"SFML/Graphics/RectangleShape.hpp\" { header \"SFML/Graphics/RectangleShape.hpp\" export * }
  module \"SFML/Graphics/RenderStates.hpp\" { header \"SFML/Graphics/RenderStates.hpp\" export * }
  module \"SFML/Graphics/RenderTarget.hpp\" { header \"SFML/Graphics/RenderTarget.hpp\" export * }
  module \"SFML/Graphics/RenderTexture.hpp\" { header \"SFML/Graphics/RenderTexture.hpp\" export * }
  module \"SFML/Graphics/RenderWindow.hpp\" { header \"SFML/Graphics/RenderWindow.hpp\" export * }
  module \"SFML/Graphics/Shader.hpp\" { header \"SFML/Graphics/Shader.hpp\" export * }
  module \"SFML/Graphics/Shape.hpp\" { header \"SFML/Graphics/Shape.hpp\" export * }
  module \"SFML/Graphics/Sprite.hpp\" { header \"SFML/Graphics/Sprite.hpp\" export * }
  module \"SFML/Graphics/Text.hpp\" { header \"SFML/Graphics/Text.hpp\" export * }
  module \"SFML/Graphics/Texture.hpp\" { header \"SFML/Graphics/Texture.hpp\" export * }
  module \"SFML/Graphics/Transform.hpp\" { header \"SFML/Graphics/Transform.hpp\" export * }
  module \"SFML/Graphics/Transformable.hpp\" { header \"SFML/Graphics/Transformable.hpp\" export * }
  module \"SFML/Graphics/Vertex.hpp\" { header \"SFML/Graphics/Vertex.hpp\" export * }
  module \"SFML/Graphics/VertexArray.hpp\" { header \"SFML/Graphics/VertexArray.hpp\" export * }
  module \"SFML/Graphics/View.hpp\" { header \"SFML/Graphics/View.hpp\" export * }
  module \"SFML/Main.hpp\" { header \"SFML/Main.hpp\" export * }
  module \"SFML/Network.hpp\" { header \"SFML/Network.hpp\" export * }
  module \"SFML/Network/Export.hpp\" { header \"SFML/Network/Export.hpp\" export * }
  module \"SFML/Network/Ftp.hpp\" { header \"SFML/Network/Ftp.hpp\" export * }
  module \"SFML/Network/Http.hpp\" { header \"SFML/Network/Http.hpp\" export * }
  module \"SFML/Network/IpAddress.hpp\" { header \"SFML/Network/IpAddress.hpp\" export * }
  module \"SFML/Network/Packet.hpp\" { header \"SFML/Network/Packet.hpp\" export * }
  module \"SFML/Network/Socket.hpp\" { header \"SFML/Network/Socket.hpp\" export * }
  module \"SFML/Network/SocketHandle.hpp\" { header \"SFML/Network/SocketHandle.hpp\" export * }
  module \"SFML/Network/SocketSelector.hpp\" { header \"SFML/Network/SocketSelector.hpp\" export * }
  module \"SFML/Network/TcpListener.hpp\" { header \"SFML/Network/TcpListener.hpp\" export * }
  module \"SFML/Network/TcpSocket.hpp\" { header \"SFML/Network/TcpSocket.hpp\" export * }
  module \"SFML/Network/UdpSocket.hpp\" { header \"SFML/Network/UdpSocket.hpp\" export * }
  module \"SFML/OpenGL.hpp\" { header \"SFML/OpenGL.hpp\" export * }
  module \"SFML/System.hpp\" { header \"SFML/System.hpp\" export * }
  module \"SFML/System/Clock.hpp\" { header \"SFML/System/Clock.hpp\" export * }
  module \"SFML/System/Err.hpp\" { header \"SFML/System/Err.hpp\" export * }
  module \"SFML/System/Export.hpp\" { header \"SFML/System/Export.hpp\" export * }
  module \"SFML/System/FileInputStream.hpp\" { header \"SFML/System/FileInputStream.hpp\" export * }
  module \"SFML/System/InputStream.hpp\" { header \"SFML/System/InputStream.hpp\" export * }
  module \"SFML/System/Lock.hpp\" { header \"SFML/System/Lock.hpp\" export * }
  module \"SFML/System/MemoryInputStream.hpp\" { header \"SFML/System/MemoryInputStream.hpp\" export * }
  module \"SFML/System/Mutex.hpp\" { header \"SFML/System/Mutex.hpp\" export * }
  module \"SFML/System/NonCopyable.hpp\" { header \"SFML/System/NonCopyable.hpp\" export * }
  module \"SFML/System/Sleep.hpp\" { header \"SFML/System/Sleep.hpp\" export * }
  module \"SFML/System/String.hpp\" { header \"SFML/System/String.hpp\" export * }
  module \"SFML/System/Thread.hpp\" { header \"SFML/System/Thread.hpp\" export * }
  module \"SFML/System/ThreadLocal.hpp\" { header \"SFML/System/ThreadLocal.hpp\" export * }
  module \"SFML/System/ThreadLocalPtr.hpp\" { header \"SFML/System/ThreadLocalPtr.hpp\" export * }
  module \"SFML/System/Time.hpp\" { header \"SFML/System/Time.hpp\" export * }
  module \"SFML/System/Utf.hpp\" { header \"SFML/System/Utf.hpp\" export * }
  module \"SFML/System/Vector2.hpp\" { header \"SFML/System/Vector2.hpp\" export * }
  module \"SFML/System/Vector3.hpp\" { header \"SFML/System/Vector3.hpp\" export * }
  module \"SFML/Window.hpp\" { header \"SFML/Window.hpp\" export * }
  module \"SFML/Window/Context.hpp\" { header \"SFML/Window/Context.hpp\" export * }
  module \"SFML/Window/Event.hpp\" { header \"SFML/Window/Event.hpp\" export * }
  module \"SFML/Window/Export.hpp\" { header \"SFML/Window/Export.hpp\" export * }
  module \"SFML/Window/GlResource.hpp\" { header \"SFML/Window/GlResource.hpp\" export * }
  module \"SFML/Window/Joystick.hpp\" { header \"SFML/Window/Joystick.hpp\" export * }
  module \"SFML/Window/Keyboard.hpp\" { header \"SFML/Window/Keyboard.hpp\" export * }
  module \"SFML/Window/Mouse.hpp\" { header \"SFML/Window/Mouse.hpp\" export * }
  module \"SFML/Window/Sensor.hpp\" { header \"SFML/Window/Sensor.hpp\" export * }
  module \"SFML/Window/Touch.hpp\" { header \"SFML/Window/Touch.hpp\" export * }
  module \"SFML/Window/VideoMode.hpp\" { header \"SFML/Window/VideoMode.hpp\" export * }
  module \"SFML/Window/WindowHandle.hpp\" { header \"SFML/Window/WindowHandle.hpp\" export * }
  module \"SFML/Window/WindowStyle.hpp\" { header \"SFML/Window/WindowStyle.hpp\" export * }
}
")

file(WRITE "${ClangModules_UNPACK_FOLDER}/libc_full.modulemap" "// provides: libc
// after: stl
// It's not possible for now to modularize libc into a single
// libc module because that causes cyclic dependencies with
// the STL which overwrites some of the libc headers
// (such as stdlib.h which this is in turn referncing libc again).
module libc_ctype    [system] { header \"ctype.h\" export * }
module libc_errno    [system] { header \"errno.h\" export * }
module libc_fenv     [system] { header \"fenv.h\" export * }
module libc_inttypes [system] { header \"inttypes.h\" export * }
module libc_locale   [system] { header \"locale.h\" export * }
module libc_math     [system] { header \"math.h\" export * }
module libc_setjmp   [system] { header \"setjmp.h\" export * }
module libc_signal   [system] { header \"signal.h\" export * }
module libc_stdint   [system] { header \"stdint.h\" export * }
module libc_stdio    [system] { header \"stdio.h\" export * }
module libc_stdlib   [system] { header \"stdlib.h\" export * }
module libc_string   [system] { header \"string.h\" export * }
module libc_tgmath   [system] { header \"tgmath.h\" export * }
module libc_time     [system] { header \"time.h\" export * }
module libc_uchar    [system] { header \"uchar.h\" export * }
")

file(WRITE "${ClangModules_UNPACK_FOLDER}/sdl2.modulemap" "// after: stl
module sdl2 [system] {
  module \"SDL_assert.h\" { header \"SDL2/SDL_assert.h\" export * }
  module \"SDL_atomic.h\" { header \"SDL2/SDL_atomic.h\" export * }
  module \"SDL_audio.h\" { header \"SDL2/SDL_audio.h\" export * }
  module \"SDL_bits.h\" { header \"SDL2/SDL_bits.h\" export * }
  module \"SDL_blendmode.h\" { header \"SDL2/SDL_blendmode.h\" export * }
  module \"SDL_clipboard.h\" { header \"SDL2/SDL_clipboard.h\" export * }
  module \"SDL_config.h\" { header \"SDL2/SDL_config.h\" export * }
  module \"SDL_cpuinfo.h\" { header \"SDL2/SDL_cpuinfo.h\" export * }
  module \"SDL_endian.h\" { header \"SDL2/SDL_endian.h\" export * }
  module \"SDL_error.h\" { header \"SDL2/SDL_error.h\" export * }
  module \"SDL_events.h\" { header \"SDL2/SDL_events.h\" export * }
  module \"SDL_filesystem.h\" { header \"SDL2/SDL_filesystem.h\" export * }
  module \"SDL_gamecontroller.h\" { header \"SDL2/SDL_gamecontroller.h\" export * }
  module \"SDL_gesture.h\" { header \"SDL2/SDL_gesture.h\" export * }
  module \"SDL_haptic.h\" { header \"SDL2/SDL_haptic.h\" export * }
  module \"SDL.h\" { header \"SDL2/SDL.h\" export * }
  module \"SDL_hints.h\" { header \"SDL2/SDL_hints.h\" export * }
  module \"SDL_joystick.h\" { header \"SDL2/SDL_joystick.h\" export * }
  module \"SDL_keyboard.h\" { header \"SDL2/SDL_keyboard.h\" export * }
  module \"SDL_keycode.h\" { header \"SDL2/SDL_keycode.h\" export * }
  module \"SDL_loadso.h\" { header \"SDL2/SDL_loadso.h\" export * }
  module \"SDL_log.h\" { header \"SDL2/SDL_log.h\" export * }
  module \"SDL_main.h\" { header \"SDL2/SDL_main.h\" export * }
  module \"SDL_messagebox.h\" { header \"SDL2/SDL_messagebox.h\" export * }
  module \"SDL_mouse.h\" { header \"SDL2/SDL_mouse.h\" export * }
  module \"SDL_mutex.h\" { header \"SDL2/SDL_mutex.h\" export * }
  module \"SDL_name.h\" { header \"SDL2/SDL_name.h\" export * }
  module \"SDL_opengl.h\" { header \"SDL2/SDL_opengl.h\" export * }
  module \"SDL_pixels.h\" { header \"SDL2/SDL_pixels.h\" export * }
  module \"SDL_platform.h\" { header \"SDL2/SDL_platform.h\" export * }
  module \"SDL_power.h\" { header \"SDL2/SDL_power.h\" export * }
  module \"SDL_quit.h\" { header \"SDL2/SDL_quit.h\" export * }
  module \"SDL_rect.h\" { header \"SDL2/SDL_rect.h\" export * }
  module \"SDL_render.h\" { header \"SDL2/SDL_render.h\" export * }
  module \"SDL_revision.h\" { header \"SDL2/SDL_revision.h\" export * }
  module \"SDL_rwops.h\" { header \"SDL2/SDL_rwops.h\" export * }
  module \"SDL_scancode.h\" { header \"SDL2/SDL_scancode.h\" export * }
  module \"SDL_shape.h\" { header \"SDL2/SDL_shape.h\" export * }
  module \"SDL_stdinc.h\" { header \"SDL2/SDL_stdinc.h\" export * }
  module \"SDL_surface.h\" { header \"SDL2/SDL_surface.h\" export * }
  module \"SDL_system.h\" { header \"SDL2/SDL_system.h\" export * }
  module \"SDL_syswm.h\" { header \"SDL2/SDL_syswm.h\" export * }
  module \"SDL_thread.h\" { header \"SDL2/SDL_thread.h\" export * }
  module \"SDL_timer.h\" { header \"SDL2/SDL_timer.h\" export * }
  module \"SDL_touch.h\" { header \"SDL2/SDL_touch.h\" export * }
  module \"SDL_types.h\" { header \"SDL2/SDL_types.h\" export * }
  module \"SDL_version.h\" { header \"SDL2/SDL_version.h\" export * }
  module \"SDL_video.h\" { header \"SDL2/SDL_video.h\" export * }

}
")

file(WRITE "${ClangModules_UNPACK_FOLDER}/sys_types_only.modulemap" "// provides: sys
module sys [system] {
  module types { header \"sys/types.h\" export * }
}
")

file(WRITE "${ClangModules_UNPACK_FOLDER}/stl17.modulemap" "// provides: stl
module stl17 [system] {
  module \"algorithm\" { header \"algorithm\" export * } // Algorithms that operate on containers
  module \"any\" { header \"any\" export * } // (since C++17) std::any class template
  module \"array\" { header \"array\" export * } // (since C++11) std::array container
  module \"atomic\" { header \"atomic\" export * } // (since C++11)  Atomic operations library
  module \"bitset\" { header \"bitset\" export * } //  std::bitset class template
  module \"cassert\" { header \"cassert\" export * } // Conditionally compiled macro that compares its argument to zero
  module \"cctype\" { header \"cctype\" export * } //  functions to determine the type contained in character data
  module \"cerrno\" { header \"cerrno\" export * } //  Macro containing the last error number
  module \"cfenv\" { header \"cfenv\" export * } // (since C++11) Floating-point environment access functions
  module \"cfloat\" { header \"cfloat\" export * } //  limits of float types
  module \"chrono\" { header \"chrono\" export * } // (since C++11)  C++ time utilites
  module \"cinttypes\" { header \"cinttypes\" export * } // (since C++11) formatting macros , intmax_t and uintmax_t math and conversions
  module \"climits\" { header \"climits\" export * } // limits of integral types
  module \"clocale\" { header \"clocale\" export * } // C localization utilities
  module \"cmath\" { header \"cmath\" export * } // Common mathematics functions
  module \"codecvt\" { header \"codecvt\" export * } // (since C++11) Unicode conversion facilities
  module \"complex\" { header \"complex\" export * } // Complex number type
  module \"condition_variable\" { header \"condition_variable\" export * } // (since C++11)  thread waiting conditions
  module \"csetjmp\" { header \"csetjmp\" export * } // Macro (and function) that saves (and jumps) to an execution context
  module \"csignal\" { header \"csignal\" export * } // Functions and macro constants for signal management
  module \"cstdarg\" { header \"cstdarg\" export * } // Handling of variable length argument lists
  module \"cstddef\" { header \"cstddef\" export * } // typedefs for types such as size_t, NULL and others
  module \"cstdint\" { header \"cstdint\" export * } // (since C++11) fixed-size types and limits of other types
  module \"cstdio\" { header \"cstdio\" export * } //  C-style input-output functions
  module \"cstdlib\" { header \"cstdlib\" export * } // General purpose utilities: program control, dynamic memory allocation, random numbers, sort and search
  module \"cstring\" { header \"cstring\" export * } // various narrow character string handling functions
  module \"ctime\" { header \"ctime\" export * } // C-style time/date utilites
  module \"cuchar\" { header \"cuchar\" export * } // (since C++11)  C-style Unicode character conversion functions
  module \"cwchar\" { header \"cwchar\" export * } //  various wide and multibyte string handling functions
  module \"cwctype\" { header \"cwctype\" export * } // functions for determining the type of wide character data
  module \"deque\" { header \"deque\" export * } // std::deque container
  module \"exception\" { header \"exception\" export * } // Exception handling utilities
  module \"execution\" { header \"execution\" export * } // (C++17) Predefined execution policies for parallel versions of the algorithms
  module \"filesystem\" { header \"filesystem\" export * } // (since C++17)  std::path class and supporting functions
  module \"forward_list\" { header \"forward_list\" export * } // (since C++11)  std::forward_list container
  module \"fstream\" { header \"fstream\" export * } // std::basic_fstream, std::basic_ifstream, std::basic_ofstream class templates and several typedefs
  module \"functional\" { header \"functional\" export * } //  Function objects, designed for use with the standard algorithms
  module \"future\" { header \"future\" export * } // (since C++11)  primitives for asynchronous computations
  module \"initializer_list\" { header \"initializer_list\" export * } // (since C++11)  std::initializer_list class template
  module \"iomanip\" { header \"iomanip\" export * } // Helper functions to control the format or input and output
  module \"iosfwd\" { header \"iosfwd\" export * } //  forward declarations of all classes in the input/output library
  module \"ios\" { header \"ios\" export * } // std::ios_base class, std::basic_ios class template and several typedefs
  module \"iostream\" { header \"iostream\" export * } //  several standard stream objects
  module \"istream\" { header \"istream\" export * } // std::basic_istream class template and several typedefs
  module \"iterator\" { header \"iterator\" export * } //  Container iterators
  module \"limits\" { header \"limits\" export * } //  standardized way to query properties of arithmetic types
  module \"list\" { header \"list\" export * } //  std::list container
  module \"locale\" { header \"locale\" export * } //  Localization utilities
  module \"map\" { header \"map\" export * } // std::map and std::multimap associative containers
  module \"memory\" { header \"memory\" export * } //  Higher level memory management utilities
  module \"memory_resource\" { header \"memory_resource\" export * } // (since C++17) Polymorphic allocators and memory resources
  module \"mutex\" { header \"mutex\" export * } // (since C++11) mutual exclusion primitives
  module \"new\" { header \"new\" export * } // Low-level memory management utilities
  module \"numeric\" { header \"numeric\" export * } // Numeric operations on values in containers
  module \"optional\" { header \"optional\" export * } // (since C++17)  std::optional class template
  module \"ostream\" { header \"ostream\" export * } // std::basic_ostream, std::basic_iostream class templates and several typedefs
  module \"queue\" { header \"queue\" export * } // std::queue and std::priority_queue container adaptors
  module \"random\" { header \"random\" export * } // (since C++11)  Random number generators and distributions
  module \"ratio\" { header \"ratio\" export * } // (since C++11) Compile-time rational arithmetic
  module \"regex\" { header \"regex\" export * } // (since C++11) Classes, algorithms and iterators to support regular expression processing
  module \"scoped_allocator\" { header \"scoped_allocator\" export * } // (since C++11)  Nested allocator class
  module \"set\" { header \"set\" export * } // std::set and std::multiset associative containers
  module \"shared_mutex\" { header \"shared_mutex\" export * } // (since C++14)  shared mutual exclusion primitives
  module \"sstream\" { header \"sstream\" export * } // std::basic_stringstream, std::basic_istringstream, std::basic_ostringstream class templates and several typedefs
  module \"stack\" { header \"stack\" export * } // std::stack container adaptor
  module \"stdexcept\" { header \"stdexcept\" export * } // Standard exception objects
  module \"streambuf\" { header \"streambuf\" export * } // std::basic_streambuf class template
  module \"string\" { header \"string\" export * } //  std::basic_string class template
  module \"string_view\" { header \"string_view\" export * } // (since C++17) std::basic_string_view class template
  module \"system_error\" { header \"system_error\" export * } // (since C++11)  defines std::error_code, a platform-dependent error code
  module \"thread\" { header \"thread\" export * } // (since C++11)  std::thread class and supporting functions
  module \"tuple\" { header \"tuple\" export * } // (since C++11) std::tuple class template
  module \"typeindex\" { header \"typeindex\" export * } // (since C++11) std::type_index
  module \"typeinfo\" { header \"typeinfo\" export * } //  Runtime type information utilities
  module \"type_traits\" { header \"type_traits\" export * } // (since C++11) Compile-time type information
  module \"unordered_map\" { header \"unordered_map\" export * } // (since C++11) std::unordered_map and std::unordered_multimap unordered associative containers
  module \"unordered_set\" { header \"unordered_set\" export * } // (since C++11) std::unordered_set and std::unordered_multiset unordered associative containers
  module \"utility\" { header \"utility\" export * } // Various utility components
  module \"valarray\" { header \"valarray\" export * } //  Class for representing and manipulating arrays of values
  module \"variant\" { header \"variant\" export * } // (since C++17) std::variant class template
  module \"vector\" { header \"vector\" export * } //  std::vector container
}
")

file(WRITE "${ClangModules_UNPACK_FOLDER}/linux.modulemap" "// after: libc
module linux [system] {
  module \"uuid.h\" { header \"linux/uuid.h\" export * }
}
")

file(WRITE "${ClangModules_UNPACK_FOLDER}/stl03.modulemap" "// after: stl11
// provides: stl
module stl03 [system] {
  module \"algorithm\" { header \"algorithm\" export * } // Algorithms that operate on containers
  module \"bitset\" { header \"bitset\" export * } //  std::bitset class template
  module \"cassert\" { header \"cassert\" export * } // Conditionally compiled macro that compares its argument to zero
  module \"cctype\" { header \"cctype\" export * } //  functions to determine the type contained in character data
  module \"cerrno\" { header \"cerrno\" export * } //  Macro containing the last error number
  module \"cfloat\" { header \"cfloat\" export * } //  limits of float types
  module \"climits\" { header \"climits\" export * } // limits of integral types
  module \"clocale\" { header \"clocale\" export * } // C localization utilities
  module \"cmath\" { header \"cmath\" export * } // Common mathematics functions
  module \"complex\" { header \"complex\" export * } // Complex number type
  module \"csetjmp\" { header \"csetjmp\" export * } // Macro (and function) that saves (and jumps) to an execution context
  module \"csignal\" { header \"csignal\" export * } // Functions and macro constants for signal management
  module \"cstdarg\" { header \"cstdarg\" export * } // Handling of variable length argument lists
  module \"cstddef\" { header \"cstddef\" export * } // typedefs for types such as size_t, NULL and others
  module \"cstdio\" { header \"cstdio\" export * } //  C-style input-output functions
  module \"cstdlib\" { header \"cstdlib\" export * } // General purpose utilities: program control, dynamic memory allocation, random numbers, sort and search
  module \"cstring\" { header \"cstring\" export * } // various narrow character string handling functions
  module \"ctime\" { header \"ctime\" export * } // C-style time/date utilites
  module \"cwchar\" { header \"cwchar\" export * } //  various wide and multibyte string handling functions
  module \"cwctype\" { header \"cwctype\" export * } // functions for determining the type of wide character data
  module \"deque\" { header \"deque\" export * } // std::deque container
  module \"exception\" { header \"exception\" export * } // Exception handling utilities
  module \"fstream\" { header \"fstream\" export * } // std::basic_fstream, std::basic_ifstream, std::basic_ofstream class templates and several typedefs
  module \"functional\" { header \"functional\" export * } //  Function objects, designed for use with the standard algorithms
  module \"iomanip\" { header \"iomanip\" export * } // Helper functions to control the format or input and output
  module \"iosfwd\" { header \"iosfwd\" export * } //  forward declarations of all classes in the input/output library
  module \"ios\" { header \"ios\" export * } // std::ios_base class, std::basic_ios class template and several typedefs
  module \"iostream\" { header \"iostream\" export * } //  several standard stream objects
  module \"istream\" { header \"istream\" export * } // std::basic_istream class template and several typedefs
  module \"iterator\" { header \"iterator\" export * } //  Container iterators
  module \"limits\" { header \"limits\" export * } //  standardized way to query properties of arithmetic types
  module \"list\" { header \"list\" export * } //  std::list container
  module \"locale\" { header \"locale\" export * } //  Localization utilities
  module \"map\" { header \"map\" export * } // std::map and std::multimap associative containers
  module \"memory\" { header \"memory\" export * } //  Higher level memory management utilities
  module \"new\" { header \"new\" export * } // Low-level memory management utilities
  module \"numeric\" { header \"numeric\" export * } // Numeric operations on values in containers
  module \"ostream\" { header \"ostream\" export * } // std::basic_ostream, std::basic_iostream class templates and several typedefs
  module \"queue\" { header \"queue\" export * } // std::queue and std::priority_queue container adaptorsssion processing
  module \"set\" { header \"set\" export * } // std::set and std::multiset associative containers
  module \"sstream\" { header \"sstream\" export * } // std::basic_stringstream, std::basic_istringstream, std::basic_ostringstream class templates and several typedefs
  module \"stack\" { header \"stack\" export * } // std::stack container adaptor
  module \"stdexcept\" { header \"stdexcept\" export * } // Standard exception objects
  module \"streambuf\" { header \"streambuf\" export * } // std::basic_streambuf class template
  module \"string\" { header \"string\" export * } //  std::basic_string class template
  module \"typeinfo\" { header \"typeinfo\" export * } //  Runtime type information utilities
  module \"utility\" { header \"utility\" export * } // Various utility components
  module \"valarray\" { header \"valarray\" export * } //  Class for representing and manipulating arrays of values
  module \"vector\" { header \"vector\" export * } //  std::vector container
}
")

file(WRITE "${ClangModules_UNPACK_FOLDER}/libc_no_signal.modulemap" "// after: libc_full
// provides: libc
// It's not possible for now to modularize libc into a single
// libc module because that causes cyclic dependencies with
// the STL which overwrites some of the libc headers
// (such as stdlib.h which this is in turn referncing libc again).
module libc_ctype    [system] { header \"ctype.h\" export * }
module libc_errno    [system] { header \"errno.h\" export * }
module libc_fenv     [system] { header \"fenv.h\" export * }
module libc_inttypes [system] { header \"inttypes.h\" export * }
module libc_locale   [system] { header \"locale.h\" export * }
module libc_math     [system] { header \"math.h\" export * }
module libc_setjmp   [system] { header \"setjmp.h\" export * }
module libc_stdint   [system] { header \"stdint.h\" export * }
module libc_stdio    [system] { header \"stdio.h\" export * }
module libc_stdlib   [system] { header \"stdlib.h\" export * }
module libc_string   [system] { header \"string.h\" export * }
module libc_tgmath   [system] { header \"tgmath.h\" export * }
module libc_time     [system] { header \"time.h\" export * }
module libc_uchar    [system] { header \"uchar.h\" export * }
")

file(WRITE "${ClangModules_UNPACK_FOLDER}/bullet_old.modulemap" "// provides: bullet
// after: bullet_new
module bullet [system] {
  module \"BulletCollision/NarrowPhaseCollision/btVoronoiSimplexSolver.h\" { header \"BulletCollision/NarrowPhaseCollision/btVoronoiSimplexSolver.h\" export * }
  module \"BulletCollision/BroadphaseCollision/btAxisSweep3.h\" { header \"BulletCollision/BroadphaseCollision/btAxisSweep3.h\" export * }
  module \"BulletCollision/BroadphaseCollision/btBroadphaseInterface.h\" { header \"BulletCollision/BroadphaseCollision/btBroadphaseInterface.h\" export * }
  module \"BulletCollision/BroadphaseCollision/btBroadphaseProxy.h\" { header \"BulletCollision/BroadphaseCollision/btBroadphaseProxy.h\" export * }
  module \"BulletCollision/BroadphaseCollision/btCollisionAlgorithm.h\" { header \"BulletCollision/BroadphaseCollision/btCollisionAlgorithm.h\" export * }
  module \"BulletCollision/BroadphaseCollision/btDbvt.h\" { header \"BulletCollision/BroadphaseCollision/btDbvt.h\" export * }
  module \"BulletCollision/BroadphaseCollision/btDbvtBroadphase.h\" { header \"BulletCollision/BroadphaseCollision/btDbvtBroadphase.h\" export * }
  module \"BulletCollision/BroadphaseCollision/btDispatcher.h\" { header \"BulletCollision/BroadphaseCollision/btDispatcher.h\" export * }
  module \"BulletCollision/BroadphaseCollision/btOverlappingPairCache.h\" { header \"BulletCollision/BroadphaseCollision/btOverlappingPairCache.h\" export * }
  module \"BulletCollision/BroadphaseCollision/btQuantizedBvh.h\" { header \"BulletCollision/BroadphaseCollision/btQuantizedBvh.h\" export * }
  module \"BulletCollision/BroadphaseCollision/btSimpleBroadphase.h\" { header \"BulletCollision/BroadphaseCollision/btSimpleBroadphase.h\" export * }
  module \"BulletCollision/CollisionDispatch/SphereTriangleDetector.h\" { header \"BulletCollision/CollisionDispatch/SphereTriangleDetector.h\" export * }
  module \"BulletCollision/CollisionDispatch/btActivatingCollisionAlgorithm.h\" { header \"BulletCollision/CollisionDispatch/btActivatingCollisionAlgorithm.h\" export * }
  module \"BulletCollision/CollisionDispatch/btBox2dBox2dCollisionAlgorithm.h\" { header \"BulletCollision/CollisionDispatch/btBox2dBox2dCollisionAlgorithm.h\" export * }
  module \"BulletCollision/CollisionDispatch/btBoxBoxCollisionAlgorithm.h\" { header \"BulletCollision/CollisionDispatch/btBoxBoxCollisionAlgorithm.h\" export * }
  module \"BulletCollision/CollisionDispatch/btBoxBoxDetector.h\" { header \"BulletCollision/CollisionDispatch/btBoxBoxDetector.h\" export * }
  module \"BulletCollision/CollisionDispatch/btCollisionConfiguration.h\" { header \"BulletCollision/CollisionDispatch/btCollisionConfiguration.h\" export * }
  module \"BulletCollision/CollisionDispatch/btCollisionCreateFunc.h\" { header \"BulletCollision/CollisionDispatch/btCollisionCreateFunc.h\" export * }
  module \"BulletCollision/CollisionDispatch/btCollisionDispatcher.h\" { header \"BulletCollision/CollisionDispatch/btCollisionDispatcher.h\" export * }
  module \"BulletCollision/CollisionDispatch/btCollisionObject.h\" { header \"BulletCollision/CollisionDispatch/btCollisionObject.h\" export * }
  module \"BulletCollision/CollisionDispatch/btCollisionObjectWrapper.h\" { header \"BulletCollision/CollisionDispatch/btCollisionObjectWrapper.h\" export * }
  module \"BulletCollision/CollisionDispatch/btCollisionWorld.h\" { header \"BulletCollision/CollisionDispatch/btCollisionWorld.h\" export * }
  module \"BulletCollision/CollisionDispatch/btCompoundCollisionAlgorithm.h\" { header \"BulletCollision/CollisionDispatch/btCompoundCollisionAlgorithm.h\" export * }
  module \"BulletCollision/CollisionDispatch/btConvex2dConvex2dAlgorithm.h\" { header \"BulletCollision/CollisionDispatch/btConvex2dConvex2dAlgorithm.h\" export * }
  module \"BulletCollision/CollisionDispatch/btConvexConcaveCollisionAlgorithm.h\" { header \"BulletCollision/CollisionDispatch/btConvexConcaveCollisionAlgorithm.h\" export * }
  module \"BulletCollision/CollisionDispatch/btConvexConvexAlgorithm.h\" { header \"BulletCollision/CollisionDispatch/btConvexConvexAlgorithm.h\" export * }
  module \"BulletCollision/CollisionDispatch/btConvexPlaneCollisionAlgorithm.h\" { header \"BulletCollision/CollisionDispatch/btConvexPlaneCollisionAlgorithm.h\" export * }
  module \"BulletCollision/CollisionDispatch/btDefaultCollisionConfiguration.h\" { header \"BulletCollision/CollisionDispatch/btDefaultCollisionConfiguration.h\" export * }
  module \"BulletCollision/CollisionDispatch/btEmptyCollisionAlgorithm.h\" { header \"BulletCollision/CollisionDispatch/btEmptyCollisionAlgorithm.h\" export * }
  module \"BulletCollision/CollisionDispatch/btGhostObject.h\" { header \"BulletCollision/CollisionDispatch/btGhostObject.h\" export * }
  module \"BulletCollision/CollisionDispatch/btInternalEdgeUtility.h\" { header \"BulletCollision/CollisionDispatch/btInternalEdgeUtility.h\" export * }
  module \"BulletCollision/CollisionDispatch/btManifoldResult.h\" { header \"BulletCollision/CollisionDispatch/btManifoldResult.h\" export * }
  module \"BulletCollision/CollisionDispatch/btSimulationIslandManager.h\" { header \"BulletCollision/CollisionDispatch/btSimulationIslandManager.h\" export * }
  module \"BulletCollision/CollisionDispatch/btSphereBoxCollisionAlgorithm.h\" { header \"BulletCollision/CollisionDispatch/btSphereBoxCollisionAlgorithm.h\" export * }
  module \"BulletCollision/CollisionDispatch/btSphereSphereCollisionAlgorithm.h\" { header \"BulletCollision/CollisionDispatch/btSphereSphereCollisionAlgorithm.h\" export * }
  module \"BulletCollision/CollisionDispatch/btSphereTriangleCollisionAlgorithm.h\" { header \"BulletCollision/CollisionDispatch/btSphereTriangleCollisionAlgorithm.h\" export * }
  module \"BulletCollision/CollisionDispatch/btUnionFind.h\" { header \"BulletCollision/CollisionDispatch/btUnionFind.h\" export * }
  module \"BulletCollision/CollisionShapes/btBox2dShape.h\" { header \"BulletCollision/CollisionShapes/btBox2dShape.h\" export * }
  module \"BulletCollision/CollisionShapes/btBoxShape.h\" { header \"BulletCollision/CollisionShapes/btBoxShape.h\" export * }
  module \"BulletCollision/CollisionShapes/btBvhTriangleMeshShape.h\" { header \"BulletCollision/CollisionShapes/btBvhTriangleMeshShape.h\" export * }
  module \"BulletCollision/CollisionShapes/btCapsuleShape.h\" { header \"BulletCollision/CollisionShapes/btCapsuleShape.h\" export * }
  module \"BulletCollision/CollisionShapes/btCollisionMargin.h\" { header \"BulletCollision/CollisionShapes/btCollisionMargin.h\" export * }
  module \"BulletCollision/CollisionShapes/btCollisionShape.h\" { header \"BulletCollision/CollisionShapes/btCollisionShape.h\" export * }
  module \"BulletCollision/CollisionShapes/btCompoundShape.h\" { header \"BulletCollision/CollisionShapes/btCompoundShape.h\" export * }
  module \"BulletCollision/CollisionShapes/btConcaveShape.h\" { header \"BulletCollision/CollisionShapes/btConcaveShape.h\" export * }
  module \"BulletCollision/CollisionShapes/btConeShape.h\" { header \"BulletCollision/CollisionShapes/btConeShape.h\" export * }
  module \"BulletCollision/CollisionShapes/btConvex2dShape.h\" { header \"BulletCollision/CollisionShapes/btConvex2dShape.h\" export * }
  module \"BulletCollision/CollisionShapes/btConvexHullShape.h\" { header \"BulletCollision/CollisionShapes/btConvexHullShape.h\" export * }
  module \"BulletCollision/CollisionShapes/btConvexInternalShape.h\" { header \"BulletCollision/CollisionShapes/btConvexInternalShape.h\" export * }
  module \"BulletCollision/CollisionShapes/btConvexPointCloudShape.h\" { header \"BulletCollision/CollisionShapes/btConvexPointCloudShape.h\" export * }
  module \"BulletCollision/CollisionShapes/btConvexPolyhedron.h\" { header \"BulletCollision/CollisionShapes/btConvexPolyhedron.h\" export * }
  module \"BulletCollision/CollisionShapes/btConvexShape.h\" { header \"BulletCollision/CollisionShapes/btConvexShape.h\" export * }
  module \"BulletCollision/CollisionShapes/btConvexTriangleMeshShape.h\" { header \"BulletCollision/CollisionShapes/btConvexTriangleMeshShape.h\" export * }
  module \"BulletCollision/CollisionShapes/btCylinderShape.h\" { header \"BulletCollision/CollisionShapes/btCylinderShape.h\" export * }
  module \"BulletCollision/CollisionShapes/btEmptyShape.h\" { header \"BulletCollision/CollisionShapes/btEmptyShape.h\" export * }
  module \"BulletCollision/CollisionShapes/btHeightfieldTerrainShape.h\" { header \"BulletCollision/CollisionShapes/btHeightfieldTerrainShape.h\" export * }
  module \"BulletCollision/CollisionShapes/btMinkowskiSumShape.h\" { header \"BulletCollision/CollisionShapes/btMinkowskiSumShape.h\" export * }
  module \"BulletCollision/CollisionShapes/btMultiSphereShape.h\" { header \"BulletCollision/CollisionShapes/btMultiSphereShape.h\" export * }
  module \"BulletCollision/CollisionShapes/btMultimaterialTriangleMeshShape.h\" { header \"BulletCollision/CollisionShapes/btMultimaterialTriangleMeshShape.h\" export * }
  module \"BulletCollision/CollisionShapes/btOptimizedBvh.h\" { header \"BulletCollision/CollisionShapes/btOptimizedBvh.h\" export * }
  module \"BulletCollision/CollisionShapes/btPolyhedralConvexShape.h\" { header \"BulletCollision/CollisionShapes/btPolyhedralConvexShape.h\" export * }
  module \"BulletCollision/CollisionShapes/btScaledBvhTriangleMeshShape.h\" { header \"BulletCollision/CollisionShapes/btScaledBvhTriangleMeshShape.h\" export * }
  module \"BulletCollision/CollisionShapes/btShapeHull.h\" { header \"BulletCollision/CollisionShapes/btShapeHull.h\" export * }
  module \"BulletCollision/CollisionShapes/btSphereShape.h\" { header \"BulletCollision/CollisionShapes/btSphereShape.h\" export * }
  module \"BulletCollision/CollisionShapes/btStaticPlaneShape.h\" { header \"BulletCollision/CollisionShapes/btStaticPlaneShape.h\" export * }
  module \"BulletCollision/CollisionShapes/btStridingMeshInterface.h\" { header \"BulletCollision/CollisionShapes/btStridingMeshInterface.h\" export * }
  module \"BulletCollision/CollisionShapes/btTetrahedronShape.h\" { header \"BulletCollision/CollisionShapes/btTetrahedronShape.h\" export * }
  module \"BulletCollision/CollisionShapes/btTriangleBuffer.h\" { header \"BulletCollision/CollisionShapes/btTriangleBuffer.h\" export * }
  module \"BulletCollision/CollisionShapes/btTriangleCallback.h\" { header \"BulletCollision/CollisionShapes/btTriangleCallback.h\" export * }
  module \"BulletCollision/CollisionShapes/btTriangleIndexVertexArray.h\" { header \"BulletCollision/CollisionShapes/btTriangleIndexVertexArray.h\" export * }
  module \"BulletCollision/CollisionShapes/btTriangleIndexVertexMaterialArray.h\" { header \"BulletCollision/CollisionShapes/btTriangleIndexVertexMaterialArray.h\" export * }
  module \"BulletCollision/CollisionShapes/btTriangleInfoMap.h\" { header \"BulletCollision/CollisionShapes/btTriangleInfoMap.h\" export * }
  module \"BulletCollision/CollisionShapes/btTriangleMesh.h\" { header \"BulletCollision/CollisionShapes/btTriangleMesh.h\" export * }
  module \"BulletCollision/CollisionShapes/btTriangleMeshShape.h\" { header \"BulletCollision/CollisionShapes/btTriangleMeshShape.h\" export * }
  module \"BulletCollision/CollisionShapes/btTriangleShape.h\" { header \"BulletCollision/CollisionShapes/btTriangleShape.h\" export * }
  module \"BulletCollision/CollisionShapes/btUniformScalingShape.h\" { header \"BulletCollision/CollisionShapes/btUniformScalingShape.h\" export * }
  module \"BulletCollision/Gimpact/btBoxCollision.h\" { header \"BulletCollision/Gimpact/btBoxCollision.h\" export * }
  module \"BulletCollision/Gimpact/btClipPolygon.h\" { header \"BulletCollision/Gimpact/btClipPolygon.h\" export * }
  module \"BulletCollision/Gimpact/btContactProcessing.h\" { header \"BulletCollision/Gimpact/btContactProcessing.h\" export * }
  module \"BulletCollision/Gimpact/btGImpactBvh.h\" { header \"BulletCollision/Gimpact/btGImpactBvh.h\" export * }
  module \"BulletCollision/Gimpact/btGImpactCollisionAlgorithm.h\" { header \"BulletCollision/Gimpact/btGImpactCollisionAlgorithm.h\" export * }
  module \"BulletCollision/Gimpact/btGImpactMassUtil.h\" { header \"BulletCollision/Gimpact/btGImpactMassUtil.h\" export * }
  module \"BulletCollision/Gimpact/btGImpactQuantizedBvh.h\" { header \"BulletCollision/Gimpact/btGImpactQuantizedBvh.h\" export * }
  module \"BulletCollision/Gimpact/btGImpactShape.h\" { header \"BulletCollision/Gimpact/btGImpactShape.h\" export * }
  module \"BulletCollision/Gimpact/btGenericPoolAllocator.h\" { header \"BulletCollision/Gimpact/btGenericPoolAllocator.h\" export * }
  module \"BulletCollision/Gimpact/btQuantization.h\" { header \"BulletCollision/Gimpact/btQuantization.h\" export * }
  module \"BulletCollision/Gimpact/btTriangleShapeEx.h\" { header \"BulletCollision/Gimpact/btTriangleShapeEx.h\" export * }
  module \"BulletCollision/Gimpact/gim_array.h\" { header \"BulletCollision/Gimpact/gim_array.h\" export * }
  module \"BulletCollision/Gimpact/gim_basic_geometry_operations.h\" { header \"BulletCollision/Gimpact/gim_basic_geometry_operations.h\" export * }
  module \"BulletCollision/Gimpact/gim_bitset.h\" { header \"BulletCollision/Gimpact/gim_bitset.h\" export * }
  module \"BulletCollision/Gimpact/gim_box_collision.h\" { header \"BulletCollision/Gimpact/gim_box_collision.h\" export * }
  module \"BulletCollision/Gimpact/gim_box_set.h\" { header \"BulletCollision/Gimpact/gim_box_set.h\" export * }
  module \"BulletCollision/Gimpact/gim_contact.h\" { header \"BulletCollision/Gimpact/gim_contact.h\" export * }
  module \"BulletCollision/Gimpact/gim_geom_types.h\" { header \"BulletCollision/Gimpact/gim_geom_types.h\" export * }
  module \"BulletCollision/Gimpact/gim_geometry.h\" { header \"BulletCollision/Gimpact/gim_geometry.h\" export * }
  module \"BulletCollision/Gimpact/gim_linear_math.h\" { header \"BulletCollision/Gimpact/gim_linear_math.h\" export * }
  module \"BulletCollision/Gimpact/gim_math.h\" { header \"BulletCollision/Gimpact/gim_math.h\" export * }
  module \"BulletCollision/Gimpact/gim_memory.h\" { header \"BulletCollision/Gimpact/gim_memory.h\" export * }
  module \"BulletCollision/Gimpact/gim_radixsort.h\" { header \"BulletCollision/Gimpact/gim_radixsort.h\" export * }
  module \"BulletCollision/Gimpact/gim_tri_collision.h\" { header \"BulletCollision/Gimpact/gim_tri_collision.h\" export * }
  module \"BulletCollision/NarrowPhaseCollision/btContinuousConvexCollision.h\" { header \"BulletCollision/NarrowPhaseCollision/btContinuousConvexCollision.h\" export * }
  module \"BulletCollision/NarrowPhaseCollision/btConvexCast.h\" { header \"BulletCollision/NarrowPhaseCollision/btConvexCast.h\" export * }
  module \"BulletCollision/NarrowPhaseCollision/btConvexPenetrationDepthSolver.h\" { header \"BulletCollision/NarrowPhaseCollision/btConvexPenetrationDepthSolver.h\" export * }
  module \"BulletCollision/NarrowPhaseCollision/btDiscreteCollisionDetectorInterface.h\" { header \"BulletCollision/NarrowPhaseCollision/btDiscreteCollisionDetectorInterface.h\" export * }
  module \"BulletCollision/NarrowPhaseCollision/btGjkConvexCast.h\" { header \"BulletCollision/NarrowPhaseCollision/btGjkConvexCast.h\" export * }
  module \"BulletCollision/NarrowPhaseCollision/btGjkEpa2.h\" { header \"BulletCollision/NarrowPhaseCollision/btGjkEpa2.h\" export * }
  module \"BulletCollision/NarrowPhaseCollision/btGjkEpaPenetrationDepthSolver.h\" { header \"BulletCollision/NarrowPhaseCollision/btGjkEpaPenetrationDepthSolver.h\" export * }
  module \"BulletCollision/NarrowPhaseCollision/btGjkPairDetector.h\" { header \"BulletCollision/NarrowPhaseCollision/btGjkPairDetector.h\" export * }
  module \"BulletCollision/NarrowPhaseCollision/btManifoldPoint.h\" { header \"BulletCollision/NarrowPhaseCollision/btManifoldPoint.h\" export * }
  module \"BulletCollision/NarrowPhaseCollision/btMinkowskiPenetrationDepthSolver.h\" { header \"BulletCollision/NarrowPhaseCollision/btMinkowskiPenetrationDepthSolver.h\" export * }
  module \"BulletCollision/NarrowPhaseCollision/btPersistentManifold.h\" { header \"BulletCollision/NarrowPhaseCollision/btPersistentManifold.h\" export * }
  module \"BulletCollision/NarrowPhaseCollision/btPointCollector.h\" { header \"BulletCollision/NarrowPhaseCollision/btPointCollector.h\" export * }
  module \"BulletCollision/NarrowPhaseCollision/btPolyhedralContactClipping.h\" { header \"BulletCollision/NarrowPhaseCollision/btPolyhedralContactClipping.h\" export * }
  module \"BulletCollision/NarrowPhaseCollision/btRaycastCallback.h\" { header \"BulletCollision/NarrowPhaseCollision/btRaycastCallback.h\" export * }
  module \"BulletCollision/NarrowPhaseCollision/btSimplexSolverInterface.h\" { header \"BulletCollision/NarrowPhaseCollision/btSimplexSolverInterface.h\" export * }
  module \"BulletCollision/NarrowPhaseCollision/btSubSimplexConvexCast.h\" { header \"BulletCollision/NarrowPhaseCollision/btSubSimplexConvexCast.h\" export * }
  module \"BulletCollision/btBulletCollisionCommon.h\" { header \"BulletCollision/btBulletCollisionCommon.h\" export * }
  module \"BulletDynamics/Character/btCharacterControllerInterface.h\" { header \"BulletDynamics/Character/btCharacterControllerInterface.h\" export * }
  module \"BulletDynamics/Character/btKinematicCharacterController.h\" { header \"BulletDynamics/Character/btKinematicCharacterController.h\" export * }
  module \"BulletDynamics/ConstraintSolver/btConeTwistConstraint.h\" { header \"BulletDynamics/ConstraintSolver/btConeTwistConstraint.h\" export * }
  module \"BulletDynamics/ConstraintSolver/btConstraintSolver.h\" { header \"BulletDynamics/ConstraintSolver/btConstraintSolver.h\" export * }
  module \"BulletDynamics/ConstraintSolver/btContactConstraint.h\" { header \"BulletDynamics/ConstraintSolver/btContactConstraint.h\" export * }
  module \"BulletDynamics/ConstraintSolver/btContactSolverInfo.h\" { header \"BulletDynamics/ConstraintSolver/btContactSolverInfo.h\" export * }
  module \"BulletDynamics/ConstraintSolver/btGearConstraint.h\" { header \"BulletDynamics/ConstraintSolver/btGearConstraint.h\" export * }
  module \"BulletDynamics/ConstraintSolver/btGeneric6DofConstraint.h\" { header \"BulletDynamics/ConstraintSolver/btGeneric6DofConstraint.h\" export * }
  module \"BulletDynamics/ConstraintSolver/btGeneric6DofSpringConstraint.h\" { header \"BulletDynamics/ConstraintSolver/btGeneric6DofSpringConstraint.h\" export * }
  module \"BulletDynamics/ConstraintSolver/btHinge2Constraint.h\" { header \"BulletDynamics/ConstraintSolver/btHinge2Constraint.h\" export * }
  module \"BulletDynamics/ConstraintSolver/btHingeConstraint.h\" { header \"BulletDynamics/ConstraintSolver/btHingeConstraint.h\" export * }
  module \"BulletDynamics/ConstraintSolver/btJacobianEntry.h\" { header \"BulletDynamics/ConstraintSolver/btJacobianEntry.h\" export * }
  module \"BulletDynamics/ConstraintSolver/btPoint2PointConstraint.h\" { header \"BulletDynamics/ConstraintSolver/btPoint2PointConstraint.h\" export * }
  module \"BulletDynamics/ConstraintSolver/btSequentialImpulseConstraintSolver.h\" { header \"BulletDynamics/ConstraintSolver/btSequentialImpulseConstraintSolver.h\" export * }
  module \"BulletDynamics/ConstraintSolver/btSliderConstraint.h\" { header \"BulletDynamics/ConstraintSolver/btSliderConstraint.h\" export * }
  module \"BulletDynamics/ConstraintSolver/btSolve2LinearConstraint.h\" { header \"BulletDynamics/ConstraintSolver/btSolve2LinearConstraint.h\" export * }
  module \"BulletDynamics/ConstraintSolver/btSolverBody.h\" { header \"BulletDynamics/ConstraintSolver/btSolverBody.h\" export * }
  module \"BulletDynamics/ConstraintSolver/btSolverConstraint.h\" { header \"BulletDynamics/ConstraintSolver/btSolverConstraint.h\" export * }
  module \"BulletDynamics/ConstraintSolver/btTypedConstraint.h\" { header \"BulletDynamics/ConstraintSolver/btTypedConstraint.h\" export * }
  module \"BulletDynamics/ConstraintSolver/btUniversalConstraint.h\" { header \"BulletDynamics/ConstraintSolver/btUniversalConstraint.h\" export * }
  module \"BulletDynamics/Dynamics/btActionInterface.h\" { header \"BulletDynamics/Dynamics/btActionInterface.h\" export * }
  module \"BulletDynamics/Dynamics/btDiscreteDynamicsWorld.h\" { header \"BulletDynamics/Dynamics/btDiscreteDynamicsWorld.h\" export * }
  module \"BulletDynamics/Dynamics/btDynamicsWorld.h\" { header \"BulletDynamics/Dynamics/btDynamicsWorld.h\" export * }
  module \"BulletDynamics/Dynamics/btRigidBody.h\" { header \"BulletDynamics/Dynamics/btRigidBody.h\" export * }
  module \"BulletDynamics/Dynamics/btSimpleDynamicsWorld.h\" { header \"BulletDynamics/Dynamics/btSimpleDynamicsWorld.h\" export * }
  module \"BulletDynamics/Vehicle/btRaycastVehicle.h\" { header \"BulletDynamics/Vehicle/btRaycastVehicle.h\" export * }
  module \"BulletDynamics/Vehicle/btVehicleRaycaster.h\" { header \"BulletDynamics/Vehicle/btVehicleRaycaster.h\" export * }
  module \"BulletDynamics/Vehicle/btWheelInfo.h\" { header \"BulletDynamics/Vehicle/btWheelInfo.h\" export * }
  module \"BulletDynamics/btBulletDynamicsCommon.h\" { header \"BulletDynamics/btBulletDynamicsCommon.h\" export * }
  module \"BulletSoftBody/btDefaultSoftBodySolver.h\" { header \"BulletSoftBody/btDefaultSoftBodySolver.h\" export * }
  module \"BulletSoftBody/btSoftBody.h\" { header \"BulletSoftBody/btSoftBody.h\" export * }
  module \"BulletSoftBody/btSoftBodyConcaveCollisionAlgorithm.h\" { header \"BulletSoftBody/btSoftBodyConcaveCollisionAlgorithm.h\" export * }
  module \"BulletSoftBody/btSoftBodyData.h\" { header \"BulletSoftBody/btSoftBodyData.h\" export * }
  module \"BulletSoftBody/btSoftBodyHelpers.h\" { header \"BulletSoftBody/btSoftBodyHelpers.h\" export * }
  module \"BulletSoftBody/btSoftBodyInternals.h\" { header \"BulletSoftBody/btSoftBodyInternals.h\" export * }
  module \"BulletSoftBody/btSoftBodyRigidBodyCollisionConfiguration.h\" { header \"BulletSoftBody/btSoftBodyRigidBodyCollisionConfiguration.h\" export * }
  module \"BulletSoftBody/btSoftBodySolverVertexBuffer.h\" { header \"BulletSoftBody/btSoftBodySolverVertexBuffer.h\" export * }
  module \"BulletSoftBody/btSoftBodySolvers.h\" { header \"BulletSoftBody/btSoftBodySolvers.h\" export * }
  module \"BulletSoftBody/btSoftRigidCollisionAlgorithm.h\" { header \"BulletSoftBody/btSoftRigidCollisionAlgorithm.h\" export * }
  module \"BulletSoftBody/btSoftRigidDynamicsWorld.h\" { header \"BulletSoftBody/btSoftRigidDynamicsWorld.h\" export * }
  module \"BulletSoftBody/btSoftSoftCollisionAlgorithm.h\" { header \"BulletSoftBody/btSoftSoftCollisionAlgorithm.h\" export * }
  module \"BulletSoftBody/btSparseSDF.h\" { header \"BulletSoftBody/btSparseSDF.h\" export * }
  module \"LinearMath/btAabbUtil2.h\" { header \"LinearMath/btAabbUtil2.h\" export * }
  module \"LinearMath/btAlignedAllocator.h\" { header \"LinearMath/btAlignedAllocator.h\" export * }
  module \"LinearMath/btAlignedObjectArray.h\" { header \"LinearMath/btAlignedObjectArray.h\" export * }
  module \"LinearMath/btConvexHull.h\" { header \"LinearMath/btConvexHull.h\" export * }
  module \"LinearMath/btConvexHullComputer.h\" { header \"LinearMath/btConvexHullComputer.h\" export * }
  module \"LinearMath/btDefaultMotionState.h\" { header \"LinearMath/btDefaultMotionState.h\" export * }
  module \"LinearMath/btGeometryUtil.h\" { header \"LinearMath/btGeometryUtil.h\" export * }
  module \"LinearMath/btGrahamScan2dConvexHull.h\" { header \"LinearMath/btGrahamScan2dConvexHull.h\" export * }
  module \"LinearMath/btHashMap.h\" { header \"LinearMath/btHashMap.h\" export * }
  module \"LinearMath/btIDebugDraw.h\" { header \"LinearMath/btIDebugDraw.h\" export * }
  module \"LinearMath/btList.h\" { header \"LinearMath/btList.h\" export * }
  module \"LinearMath/btMatrix3x3.h\" { header \"LinearMath/btMatrix3x3.h\" export * }
  module \"LinearMath/btMinMax.h\" { header \"LinearMath/btMinMax.h\" export * }
  module \"LinearMath/btMotionState.h\" { header \"LinearMath/btMotionState.h\" export * }
  module \"LinearMath/btPolarDecomposition.h\" { header \"LinearMath/btPolarDecomposition.h\" export * }
  module \"LinearMath/btPoolAllocator.h\" { header \"LinearMath/btPoolAllocator.h\" export * }
  module \"LinearMath/btQuadWord.h\" { header \"LinearMath/btQuadWord.h\" export * }
  module \"LinearMath/btQuaternion.h\" { header \"LinearMath/btQuaternion.h\" export * }
  module \"LinearMath/btQuickprof.h\" { header \"LinearMath/btQuickprof.h\" export * }
  module \"LinearMath/btScalar.h\" { header \"LinearMath/btScalar.h\" export * }
  module \"LinearMath/btSerializer.h\" { header \"LinearMath/btSerializer.h\" export * }
  module \"LinearMath/btStackAlloc.h\" { header \"LinearMath/btStackAlloc.h\" export * }
  module \"LinearMath/btTransform.h\" { header \"LinearMath/btTransform.h\" export * }
  module \"LinearMath/btTransformUtil.h\" { header \"LinearMath/btTransformUtil.h\" export * }
  module \"LinearMath/btVector3.h\" { header \"LinearMath/btVector3.h\" export * }
  module \"btBulletCollisionCommon.h\" { header \"btBulletCollisionCommon.h\" export * }
  module \"btBulletDynamicsCommon.h\" { header \"btBulletDynamicsCommon.h\" export * }
}
")

file(WRITE "${ClangModules_UNPACK_FOLDER}/tinyxml.modulemap" "module tinyxml [system] { header \"tinyxml.h\" export * }
")

file(WRITE "${ClangModules_UNPACK_FOLDER}/sfml_newer.modulemap" "// provides: sfml
// after: stl
module sfml [system] {
  module \"SFML/Audio.hpp\" { header \"SFML/Audio.hpp\" export * }
  module \"SFML/Audio/Export.hpp\" { header \"SFML/Audio/Export.hpp\" export * }
  module \"SFML/Audio/Listener.hpp\" { header \"SFML/Audio/Listener.hpp\" export * }
  module \"SFML/Audio/Music.hpp\" { header \"SFML/Audio/Music.hpp\" export * }
  module \"SFML/Audio/Sound.hpp\" { header \"SFML/Audio/Sound.hpp\" export * }
  module \"SFML/Audio/SoundBuffer.hpp\" { header \"SFML/Audio/SoundBuffer.hpp\" export * }
  module \"SFML/Audio/SoundBufferRecorder.hpp\" { header \"SFML/Audio/SoundBufferRecorder.hpp\" export * }
  module \"SFML/Audio/SoundRecorder.hpp\" { header \"SFML/Audio/SoundRecorder.hpp\" export * }
  module \"SFML/Audio/SoundSource.hpp\" { header \"SFML/Audio/SoundSource.hpp\" export * }
  module \"SFML/Audio/SoundStream.hpp\" { header \"SFML/Audio/SoundStream.hpp\" export * }
  module \"SFML/Config.hpp\" { header \"SFML/Config.hpp\" export * }
  module \"SFML/Graphics.hpp\" { header \"SFML/Graphics.hpp\" export * }
  module \"SFML/Graphics/BlendMode.hpp\" { header \"SFML/Graphics/BlendMode.hpp\" export * }
  module \"SFML/Graphics/CircleShape.hpp\" { header \"SFML/Graphics/CircleShape.hpp\" export * }
  module \"SFML/Graphics/Color.hpp\" { header \"SFML/Graphics/Color.hpp\" export * }
  module \"SFML/Graphics/ConvexShape.hpp\" { header \"SFML/Graphics/ConvexShape.hpp\" export * }
  module \"SFML/Graphics/Drawable.hpp\" { header \"SFML/Graphics/Drawable.hpp\" export * }
  module \"SFML/Graphics/Export.hpp\" { header \"SFML/Graphics/Export.hpp\" export * }
  module \"SFML/Graphics/Font.hpp\" { header \"SFML/Graphics/Font.hpp\" export * }
  module \"SFML/Graphics/Glyph.hpp\" { header \"SFML/Graphics/Glyph.hpp\" export * }
  module \"SFML/Graphics/Image.hpp\" { header \"SFML/Graphics/Image.hpp\" export * }
  module \"SFML/Graphics/PrimitiveType.hpp\" { header \"SFML/Graphics/PrimitiveType.hpp\" export * }
  module \"SFML/Graphics/Rect.hpp\" { header \"SFML/Graphics/Rect.hpp\" export * }
  module \"SFML/Graphics/RectangleShape.hpp\" { header \"SFML/Graphics/RectangleShape.hpp\" export * }
  module \"SFML/Graphics/RenderStates.hpp\" { header \"SFML/Graphics/RenderStates.hpp\" export * }
  module \"SFML/Graphics/RenderTarget.hpp\" { header \"SFML/Graphics/RenderTarget.hpp\" export * }
  module \"SFML/Graphics/RenderTexture.hpp\" { header \"SFML/Graphics/RenderTexture.hpp\" export * }
  module \"SFML/Graphics/RenderWindow.hpp\" { header \"SFML/Graphics/RenderWindow.hpp\" export * }
  module \"SFML/Graphics/Shader.hpp\" { header \"SFML/Graphics/Shader.hpp\" export * }
  module \"SFML/Graphics/Shape.hpp\" { header \"SFML/Graphics/Shape.hpp\" export * }
  module \"SFML/Graphics/Sprite.hpp\" { header \"SFML/Graphics/Sprite.hpp\" export * }
  module \"SFML/Graphics/Text.hpp\" { header \"SFML/Graphics/Text.hpp\" export * }
  module \"SFML/Graphics/Texture.hpp\" { header \"SFML/Graphics/Texture.hpp\" export * }
  module \"SFML/Graphics/Transform.hpp\" { header \"SFML/Graphics/Transform.hpp\" export * }
  module \"SFML/Graphics/Transformable.hpp\" { header \"SFML/Graphics/Transformable.hpp\" export * }
  module \"SFML/Graphics/Vertex.hpp\" { header \"SFML/Graphics/Vertex.hpp\" export * }
  module \"SFML/Graphics/VertexArray.hpp\" { header \"SFML/Graphics/VertexArray.hpp\" export * }
  module \"SFML/Graphics/View.hpp\" { header \"SFML/Graphics/View.hpp\" export * }
  module \"SFML/Network.hpp\" { header \"SFML/Network.hpp\" export * }
  module \"SFML/Network/Export.hpp\" { header \"SFML/Network/Export.hpp\" export * }
  module \"SFML/Network/Ftp.hpp\" { header \"SFML/Network/Ftp.hpp\" export * }
  module \"SFML/Network/Http.hpp\" { header \"SFML/Network/Http.hpp\" export * }
  module \"SFML/Network/IpAddress.hpp\" { header \"SFML/Network/IpAddress.hpp\" export * }
  module \"SFML/Network/Packet.hpp\" { header \"SFML/Network/Packet.hpp\" export * }
  module \"SFML/Network/Socket.hpp\" { header \"SFML/Network/Socket.hpp\" export * }
  module \"SFML/Network/SocketHandle.hpp\" { header \"SFML/Network/SocketHandle.hpp\" export * }
  module \"SFML/Network/SocketSelector.hpp\" { header \"SFML/Network/SocketSelector.hpp\" export * }
  module \"SFML/Network/TcpListener.hpp\" { header \"SFML/Network/TcpListener.hpp\" export * }
  module \"SFML/Network/TcpSocket.hpp\" { header \"SFML/Network/TcpSocket.hpp\" export * }
  module \"SFML/Network/UdpSocket.hpp\" { header \"SFML/Network/UdpSocket.hpp\" export * }
  module \"SFML/OpenGL.hpp\" { header \"SFML/OpenGL.hpp\" export * }
  module \"SFML/System.hpp\" { header \"SFML/System.hpp\" export * }
  module \"SFML/System/Clock.hpp\" { header \"SFML/System/Clock.hpp\" export * }
  module \"SFML/System/Err.hpp\" { header \"SFML/System/Err.hpp\" export * }
  module \"SFML/System/Export.hpp\" { header \"SFML/System/Export.hpp\" export * }
  module \"SFML/System/InputStream.hpp\" { header \"SFML/System/InputStream.hpp\" export * }
  module \"SFML/System/Lock.hpp\" { header \"SFML/System/Lock.hpp\" export * }
  module \"SFML/System/Mutex.hpp\" { header \"SFML/System/Mutex.hpp\" export * }
  module \"SFML/System/NonCopyable.hpp\" { header \"SFML/System/NonCopyable.hpp\" export * }
  module \"SFML/System/Sleep.hpp\" { header \"SFML/System/Sleep.hpp\" export * }
  module \"SFML/System/String.hpp\" { header \"SFML/System/String.hpp\" export * }
  module \"SFML/System/Thread.hpp\" { header \"SFML/System/Thread.hpp\" export * }
  module \"SFML/System/ThreadLocal.hpp\" { header \"SFML/System/ThreadLocal.hpp\" export * }
  module \"SFML/System/ThreadLocalPtr.hpp\" { header \"SFML/System/ThreadLocalPtr.hpp\" export * }
  module \"SFML/System/Time.hpp\" { header \"SFML/System/Time.hpp\" export * }
  module \"SFML/System/Utf.hpp\" { header \"SFML/System/Utf.hpp\" export * }
  module \"SFML/System/Vector2.hpp\" { header \"SFML/System/Vector2.hpp\" export * }
  module \"SFML/System/Vector3.hpp\" { header \"SFML/System/Vector3.hpp\" export * }
  module \"SFML/Window.hpp\" { header \"SFML/Window.hpp\" export * }
  module \"SFML/Window/Context.hpp\" { header \"SFML/Window/Context.hpp\" export * }
  module \"SFML/Window/Event.hpp\" { header \"SFML/Window/Event.hpp\" export * }
  module \"SFML/Window/Export.hpp\" { header \"SFML/Window/Export.hpp\" export * }
  module \"SFML/Window/GlResource.hpp\" { header \"SFML/Window/GlResource.hpp\" export * }
  module \"SFML/Window/Joystick.hpp\" { header \"SFML/Window/Joystick.hpp\" export * }
  module \"SFML/Window/Keyboard.hpp\" { header \"SFML/Window/Keyboard.hpp\" export * }
  module \"SFML/Window/Mouse.hpp\" { header \"SFML/Window/Mouse.hpp\" export * }
  module \"SFML/Window/VideoMode.hpp\" { header \"SFML/Window/VideoMode.hpp\" export * }
  module \"SFML/Window/WindowHandle.hpp\" { header \"SFML/Window/WindowHandle.hpp\" export * }
  module \"SFML/Window/WindowStyle.hpp\" { header \"SFML/Window/WindowStyle.hpp\" export * }
}
")

file(WRITE "${ClangModules_UNPACK_FOLDER}/boost_min.modulemap" "// provides: boost
// after: stl
module boost [system] {
  module shared_ptr { header \"boost/shared_ptr.hpp\" export * }
  module thread { header \"boost/thread.hpp\" export * }
}
")

file(WRITE "${ClangModules_UNPACK_FOLDER}/stl11.modulemap" "// after: stl14
// provides: stl
module stl11 [system] {
  module \"algorithm\" { header \"algorithm\" export * } // Algorithms that operate on containers
  module \"array\" { header \"array\" export * } // (since C++11) std::array container
  module \"atomic\" { header \"atomic\" export * } // (since C++11)  Atomic operations library
  module \"bitset\" { header \"bitset\" export * } //  std::bitset class template
  module \"cassert\" { header \"cassert\" export * } // Conditionally compiled macro that compares its argument to zero
  module \"cctype\" { header \"cctype\" export * } //  functions to determine the type contained in character data
  module \"cerrno\" { header \"cerrno\" export * } //  Macro containing the last error number
  module \"cfenv\" { header \"cfenv\" export * } // (since C++11) Floating-point environment access functions
  module \"cfloat\" { header \"cfloat\" export * } //  limits of float types
  module \"chrono\" { header \"chrono\" export * } // (since C++11)  C++ time utilites
  module \"cinttypes\" { header \"cinttypes\" export * } // (since C++11) formatting macros , intmax_t and uintmax_t math and conversions
  module \"climits\" { header \"climits\" export * } // limits of integral types
  module \"clocale\" { header \"clocale\" export * } // C localization utilities
  module \"cmath\" { header \"cmath\" export * } // Common mathematics functions
  module \"codecvt\" { header \"codecvt\" export * } // (since C++11) Unicode conversion facilities
  module \"complex\" { header \"complex\" export * } // Complex number type
  module \"condition_variable\" { header \"condition_variable\" export * } // (since C++11)  thread waiting conditions
  module \"csetjmp\" { header \"csetjmp\" export * } // Macro (and function) that saves (and jumps) to an execution context
  module \"csignal\" { header \"csignal\" export * } // Functions and macro constants for signal management
  module \"cstdarg\" { header \"cstdarg\" export * } // Handling of variable length argument lists
  module \"cstddef\" { header \"cstddef\" export * } // typedefs for types such as size_t, NULL and others
  module \"cstdint\" { header \"cstdint\" export * } // (since C++11) fixed-size types and limits of other types
  module \"cstdio\" { header \"cstdio\" export * } //  C-style input-output functions
  module \"cstdlib\" { header \"cstdlib\" export * } // General purpose utilities: program control, dynamic memory allocation, random numbers, sort and search
  module \"cstring\" { header \"cstring\" export * } // various narrow character string handling functions
  module \"ctime\" { header \"ctime\" export * } // C-style time/date utilites
  module \"cuchar\" { header \"cuchar\" export * } // (since C++11)  C-style Unicode character conversion functions
  module \"cwchar\" { header \"cwchar\" export * } //  various wide and multibyte string handling functions
  module \"cwctype\" { header \"cwctype\" export * } // functions for determining the type of wide character data
  module \"deque\" { header \"deque\" export * } // std::deque container
  module \"exception\" { header \"exception\" export * } // Exception handling utilities
  module \"forward_list\" { header \"forward_list\" export * } // (since C++11)  std::forward_list container
  module \"fstream\" { header \"fstream\" export * } // std::basic_fstream, std::basic_ifstream, std::basic_ofstream class templates and several typedefs
  module \"functional\" { header \"functional\" export * } //  Function objects, designed for use with the standard algorithms
  module \"future\" { header \"future\" export * } // (since C++11)  primitives for asynchronous computations
  module \"initializer_list\" { header \"initializer_list\" export * } // (since C++11)  std::initializer_list class template
  module \"iomanip\" { header \"iomanip\" export * } // Helper functions to control the format or input and output
  module \"iosfwd\" { header \"iosfwd\" export * } //  forward declarations of all classes in the input/output library
  module \"ios\" { header \"ios\" export * } // std::ios_base class, std::basic_ios class template and several typedefs
  module \"iostream\" { header \"iostream\" export * } //  several standard stream objects
  module \"istream\" { header \"istream\" export * } // std::basic_istream class template and several typedefs
  module \"iterator\" { header \"iterator\" export * } //  Container iterators
  module \"limits\" { header \"limits\" export * } //  standardized way to query properties of arithmetic types
  module \"list\" { header \"list\" export * } //  std::list container
  module \"locale\" { header \"locale\" export * } //  Localization utilities
  module \"map\" { header \"map\" export * } // std::map and std::multimap associative containers
  module \"memory\" { header \"memory\" export * } //  Higher level memory management utilities
  module \"mutex\" { header \"mutex\" export * } // (since C++11) mutual exclusion primitives
  module \"new\" { header \"new\" export * } // Low-level memory management utilities
  module \"numeric\" { header \"numeric\" export * } // Numeric operations on values in containers
  module \"ostream\" { header \"ostream\" export * } // std::basic_ostream, std::basic_iostream class templates and several typedefs
  module \"queue\" { header \"queue\" export * } // std::queue and std::priority_queue container adaptors
  module \"random\" { header \"random\" export * } // (since C++11)  Random number generators and distributions
  module \"ratio\" { header \"ratio\" export * } // (since C++11) Compile-time rational arithmetic
  module \"regex\" { header \"regex\" export * } // (since C++11) Classes, algorithms and iterators to support regular expression processing
  module \"scoped_allocator\" { header \"scoped_allocator\" export * } // (since C++11)  Nested allocator class
  module \"set\" { header \"set\" export * } // std::set and std::multiset associative containers
  module \"sstream\" { header \"sstream\" export * } // std::basic_stringstream, std::basic_istringstream, std::basic_ostringstream class templates and several typedefs
  module \"stack\" { header \"stack\" export * } // std::stack container adaptor
  module \"stdexcept\" { header \"stdexcept\" export * } // Standard exception objects
  module \"streambuf\" { header \"streambuf\" export * } // std::basic_streambuf class template
  module \"string\" { header \"string\" export * } //  std::basic_string class template
  module \"system_error\" { header \"system_error\" export * } // (since C++11)  defines std::error_code, a platform-dependent error code
  module \"thread\" { header \"thread\" export * } // (since C++11)  std::thread class and supporting functions
  module \"tuple\" { header \"tuple\" export * } // (since C++11) std::tuple class template
  module \"typeindex\" { header \"typeindex\" export * } // (since C++11) std::type_index
  module \"typeinfo\" { header \"typeinfo\" export * } //  Runtime type information utilities
  module \"type_traits\" { header \"type_traits\" export * } // (since C++11) Compile-time type information
  module \"unordered_map\" { header \"unordered_map\" export * } // (since C++11) std::unordered_map and std::unordered_multimap unordered associative containers
  module \"unordered_set\" { header \"unordered_set\" export * } // (since C++11) std::unordered_set and std::unordered_multiset unordered associative containers
  module \"utility\" { header \"utility\" export * } // Various utility components
  module \"valarray\" { header \"valarray\" export * } //  Class for representing and manipulating arrays of values
  module \"vector\" { header \"vector\" export * } //  std::vector container
}
")


  endfunction()

  message(STATUS "Configuring ClangModules")
  ClangModules_UnpackFiles()
  
  get_property(ClangModules_CURRENT_COMPILE_OPTIONS DIRECTORY PROPERTY COMPILE_OPTIONS)
  
  set(ClangModules_IncArg ":")
  get_property(ClangModules_dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
  foreach(inc ${ClangModules_dirs})
    set(ClangModules_IncArg "${ClangModules_IncArg}:${inc}")
  endforeach()

  if(NOT ClangModules_CheckOnlyFor)
    set(ClangModules_CheckOnlyFor ";")
  endif()

  if(NOT ClangModules_CustomModulemapFolders)
    set(ClangModules_CustomModulemapFolders ";")
  endif()

  if(NOT ClangModules_RequiredModules)
    set(ClangModules_RequiredModules ";")
  endif()

  if(NOT ClangModules_OutputVFSFile)
    set(ClangModules_OutputVFSFile "-")
  endif()

  if(NOT ClangModules_ModulesCache)
    set(ClangModules_ModulesCache "${CMAKE_BINARY_DIR}/pcm")
  endif()

  set(ClangModules_ClangInvocation "${CMAKE_CXX_COMPILER} ${CMAKE_CXX_COMPILER_ARG1} ${CMAKE_CXX_FLAGS} ${ClangModules_CURRENT_COMPILE_OPTIONS}")
  message(STATUS "Using clang invocation: ${ClangModules_ClangInvocation}")
  execute_process(COMMAND ${PYTHON_EXECUTABLE}
                 "${ClangModules_UNPACK_FOLDER}/ClangModules.py"
                 --modulemap-dir "${ClangModules_UNPACK_FOLDER}"
                 --modulemap-dir "${ClangModules_CustomModulemapFolders}"
                 --output-dir "${ClangModules_UNPACK_FOLDER}"
                 -I "${ClangModules_IncArg}"
                 ${ClangModules_ClanglessArg}
                 --vfs-output "${ClangModules_OutputVFSFile}"
                 --required-modules "${ClangModules_RequiredModules}"
                 --check-only "${ClangModules_CheckOnlyFor}"
                 --invocation "${ClangModules_ClangInvocation}"
                 WORKING_DIRECTORY "${ClangModules_UNPACK_FOLDER}"
                 RESULT_VARIABLE ClangModules_py_exitcode
                 OUTPUT_VARIABLE ClangModules_CXX_FLAGS_RAW
                 OUTPUT_STRIP_TRAILING_WHITESPACE
                 )
                 
  if(NOT "${ClangModules_py_exitcode}" EQUAL 0)
    message(FATAL_ERROR "ClangModules failed with exit code ${ClangModules_py_exitcode}!")
  endif()

  if(ClangModules_ExtraFlags)
    set(ClangModules_CXX_FLAGS "${ClangModules_CXX_FLAGS_RAW} ${ClangModules_EXTRA_FLAGS} -fmodules-cache-path=${ClangModules_ModulesCache}")
  else()
    set(ClangModules_CXX_FLAGS "${ClangModules_CXX_FLAGS_RAW} -fmodules-cache-path=${ClangModules_ModulesCache}")
  endif()
  string(REPLACE "-fcxx-modules" "" ClangModules_C_FLAGS "${ClangModules_CXX_FLAGS}")

  if(ClangModules_IsClang AND NOT ClangModules_WithoutClang)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ClangModules_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ClangModules_CXX_FLAGS}")
  endif()
endif()
endif()
