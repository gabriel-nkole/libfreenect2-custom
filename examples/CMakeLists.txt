CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12.1)

if(WIN32 AND NOT MINGW)
  if(NOT DEFINED CMAKE_DEBUG_POSTFIX)
    set(CMAKE_DEBUG_POSTFIX "d")
  endif()
endif()

IF(NOT DEFINED CMAKE_BUILD_TYPE)
  # No effect for multi-configuration generators (e.g. for Visual Studio)
  SET(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose: RelWithDebInfo Release Debug MinSizeRel None")
ENDIF()

PROJECT(libfreenect2_examples)

SET(MY_DIR ${libfreenect2_examples_SOURCE_DIR})
SET(DEPENDS_DIR "${MY_DIR}/../depends" CACHE STRING "Dependency directory")

OPTION(ENABLE_OPENGL "Enable OpenGL support" ON)

# The example build system is standalone and will work out-of-tree with these files copied
SET(freenect2_ROOT_DIR ${MY_DIR}/..)
SET(flextGL_SOURCES ${freenect2_ROOT_DIR}/src/flextGL.cpp)
SET(flextGL_INCLUDE_DIRS ${freenect2_ROOT_DIR}/src) # for flextGL.h

FIND_PACKAGE(PkgConfig)    # try find PKGConfig as it will be used if found
LIST(APPEND CMAKE_MODULE_PATH ${freenect2_ROOT_DIR}/cmake_modules) # FindGLFW3.cmake

IF(TARGET freenect2)
  MESSAGE(STATUS "Using in-tree freenect2 target")
  SET(freenect2_LIBRARIES freenect2)
  SET(freenect2_DLLS ${LIBFREENECT2_DLLS})
ELSE()
  FIND_PACKAGE(freenect2 REQUIRED)
  # Out-of-tree build will have to have DLLs manually copied.
ENDIF()

INCLUDE_DIRECTORIES(
  ${freenect2_INCLUDE_DIR}
)

SET(Protonect_src
  Protonect.cpp
)

SET(Protonect_LIBRARIES
  ${freenect2_LIBRARIES}
)

SET(Protonect_DLLS
  ${freenect2_DLLS}
)

IF(ENABLE_OPENGL)
  FIND_PACKAGE(GLFW3)
  FIND_PACKAGE(OpenGL)
  find_package(OpenCV REQUIRED) # MEEE
  find_package(OpenMP REQUIRED) # MEEE
  include_directories(${OpenCV_INCLUDE_DIRS}) # MEEE
  include_directories(
    imgui
    imgui/backends
  )
  IF(GLFW3_FOUND AND OPENGL_FOUND)
    INCLUDE_DIRECTORIES(
      ${GLFW3_INCLUDE_DIRS}
      #${flextGL_INCLUDE_DIRS}
    )

    LIST(APPEND Protonect_DLLS ${GLFW3_DLL})
    LIST(APPEND Protonect_src
      #${flextGL_SOURCES}
      glad.c
      imgui/imgui.cpp
      imgui/imgui_demo.cpp
      imgui/imgui_draw.cpp
      imgui/imgui_tables.cpp
      imgui/imgui_widgets.cpp
      imgui/backends/imgui_impl_glfw.cpp
      imgui/backends/imgui_impl_opengl3.cpp
    )
    LIST(APPEND Protonect_LIBRARIES
      ${GLFW3_LIBRARIES}
      ${OPENGL_gl_LIBRARY}
    )
    ADD_DEFINITIONS(-DEXAMPLES_WITH_OPENGL_SUPPORT=1)
  ENDIF()
ENDIF(ENABLE_OPENGL)

ADD_EXECUTABLE(Protonect
  ${Protonect_src}
)

TARGET_LINK_LIBRARIES(Protonect
  ${Protonect_LIBRARIES}
)

target_link_libraries(Protonect ${OpenCV_LIBS}) # MEEE
target_link_libraries(Protonect OpenMP::OpenMP_CXX)

IF(WIN32)
  INSTALL(TARGETS Protonect DESTINATION bin)
  LIST(REMOVE_DUPLICATES Protonect_DLLS)
  FOREACH(FILEI ${Protonect_DLLS})
    ADD_CUSTOM_COMMAND(TARGET Protonect POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${FILEI} $<TARGET_FILE_DIR:Protonect>
    )
  ENDFOREACH(FILEI)
  INSTALL(FILES ${Protonect_DLLS} DESTINATION bin)
ENDIF()
