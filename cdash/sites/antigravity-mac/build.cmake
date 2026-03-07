#############################################
# Site-Specific Build Configuration Script  #
#############################################

SET(DO_UPLOAD OFF)
SET(MINVER 11.0)
SET(FWDIR "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk")

SETCOND(ARCH arm64 CONFIG arm64rel)

ENV_ADD(MAKEFLAGS "-j 8")

CACHE_ADD("MAKECOMMAND:STRING=/usr/bin/make -j 8")
CACHE_ADD("CMAKE_MAKE_PROGRAM:FILEPATH=/usr/bin/make")
CACHE_ADD("CMAKE_BUILD_TYPE:STRING=Release" CONFIG ".*rel")
CACHE_ADD("CMAKE_C_FLAGS:STRING=-mmacosx-version-min=${MINVER} -Wno-deprecated -Wno-implicit-function-declaration")
CACHE_ADD("CMAKE_CXX_FLAGS:STRING=-mmacosx-version-min=${MINVER} -Wno-deprecated -Wno-implicit-function-declaration --std=c++17")
CACHE_ADD("ARCH:STRING=${ARCH}")
CACHE_ADD("CMAKE_OSX_ARCHITECTURES:STRING=${ARCH}")

CACHE_ADD("BUILD_SHARED_LIBS:BOOL=ON")

# Qt6 Configuration
CACHE_ADD("CMAKE_PREFIX_PATH:STRING=/opt/homebrew/lib/cmake/Qt6")

# VTK Configuration
CACHE_ADD("VTK_MODULE_USE_EXTERNAL_VTK_zlib:BOOL=ON" PRODUCT "vtk")
CACHE_ADD("VTK_MODULE_ENABLE_VTK_RenderingExternal:STRING=YES" PRODUCT "vtk")
