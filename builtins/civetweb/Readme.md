# Patched version of civetweb master

Unfortunately last civetweb released version 1.16 does not include necessary changes,
therefore unreleased master version is used plus several patches.

One strategy to build civetweb is checkout github repository, patch it at place and then build.
Main disadvantege of such approach - any ROOT build will require checkout of full repository
which is about 100MB of files. This possibility is commented out in cmake file -
see `GIT_REPOSITORY` and `GIT_TAG` parameters.

Instead minimalistic tarball file is created and placed on LCG download site.
Such minimalistic tarball includes following files:

    * `cmake/`    - cmake files
    * `include/`  - headers
    * `src/`      - sources without `src/third_party`
    * CMakeLists.txt - main cmake file
    * CREDITS.md, LICENSE.md, README.md, RELEASE_NOTES.md, SECURITY.md - misc

Full command to create tarball is:

    git clone https://github.com/civetweb/civetweb.git
    cd civetweb
    tar -czf civetweb-<githash>.tar.gz --exclude='src/third_party'  cmake include src CMakeLists.txt CREDITS.md LICENSE.md README.md RELEASE_NOTES.md SECURITY.md


