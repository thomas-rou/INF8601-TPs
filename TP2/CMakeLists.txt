cmake_minimum_required(VERSION 3.13.5)
project(tp2 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-fno-rtti")

set(CMAKE_C_FLAGS "${CMAKE_CFLAGS} -g -Wall")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_CFLAGS_RELEASE} -O2")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fsanitize=address")
set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer")
set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fsanitize=address")

include_directories(include)
add_executable(sinoscope)
target_link_libraries(sinoscope -lm -lpthread)
target_sources(sinoscope PUBLIC
    source/color.c
    source/headless.c
    source/image.c
    source/main.c
    source/opencl.c
    source/sinoscope.c
    source/sinoscope-serial.c
    source/sinoscope-openmp.c
    source/sinoscope-opencl.c
)

add_executable(sinoscope-nocl)
target_link_libraries(sinoscope-nocl -lm -lpthread)
target_sources(sinoscope-nocl PUBLIC
    source/color.c
    source/headless.c
    source/image.c
    source/main.c
    source/sinoscope.c
    source/sinoscope-serial.c
    source/sinoscope-openmp.c
)

add_executable(sinoscope-nomp)
target_link_libraries(sinoscope-nomp -lm -lpthread)
target_sources(sinoscope-nomp PUBLIC
    source/color.c
    source/headless.c
    source/image.c
    source/main.c
    source/opencl.c
    source/sinoscope.c
    source/sinoscope-serial.c
    source/sinoscope-opencl.c
)

include(FindOpenMP)
if(OpenMP_C_FOUND)
    target_link_libraries(sinoscope ${OpenMP_C_LIBRARIES})
    target_link_libraries(sinoscope-nocl ${OpenMP_C_LIBRARIES})
else()
    message(FATAL_ERROR "openmp is required for building the application")
endif()

if (DEFINED ROCM)
  target_link_libraries(sinoscope ${ROCM}/lib/libamdocl64.so)
  target_link_libraries(sinoscope-nomp ${ROCM}/lib/libamdocl64.so)
else()

  include(FindOpenCL)
  if(OpenCL_FOUND)
      target_link_libraries(sinoscope ${OpenCL_LIBRARY})
      target_include_directories(sinoscope PUBLIC ${OpenCL_INCLUDE_DIR})
      target_link_libraries(sinoscope-nomp ${OpenCL_LIBRARY})
      target_include_directories(sinoscope-nomp PUBLIC ${OpenCL_INCLUDE_DIR})
  else()
      if(EXISTS /usr/lib64/libOpenCL.so.1)
          target_link_libraries(sinoscope /usr/lib64/libOpenCL.so.1)
          target_link_libraries(sinoscope-nomp /usr/lib64/libOpenCL.so.1)
      else()
          message(FATAL_ERROR "opencl is required for building the application")
      endif()
  endif()

  add_definitions(-DWITH_VIEWER)
  target_sources(sinoscope PUBLIC
      source/viewer.c
  )

  include(FindOpenGL)
  if(OPENGL_FOUND)
      if(NOT OPENGL_GLU_FOUND)
          message(FATAL_ERROR "glu is required for building the application")
      endif()

      target_link_libraries(sinoscope ${OPENGL_LIBRARIES})
      target_include_directories(sinoscope PUBLIC ${OPENGL_INCLUDE_DIR})
      target_link_libraries(sinoscope-nomp ${OPENGL_LIBRARIES})
      target_include_directories(sinoscope-nomp PUBLIC ${OPENGL_INCLUDE_DIR})
      target_link_libraries(sinoscope-nocl ${OPENGL_LIBRARIES})
      target_include_directories(sinoscope-nocl PUBLIC ${OPENGL_INCLUDE_DIR})
  else()
      message(FATAL_ERROR "opengl is required for building the application")
  endif()

  include(FindGLUT)
  if(GLUT_FOUND)
     target_link_libraries(sinoscope ${GLUT_LIBRARIES})
     target_include_directories(sinoscope PUBLIC ${GLUT_INCLUDE_DIR})
     target_link_libraries(sinoscope-nocl ${GLUT_LIBRARIES})
     target_include_directories(sinoscope-nocl PUBLIC ${GLUT_INCLUDE_DIR})
     target_link_libraries(sinoscope-nomp ${GLUT_LIBRARIES})
     target_include_directories(sinoscope-nomp PUBLIC ${GLUT_INCLUDE_DIR})
  else()
     message(FATAL_ERROR "glut is required for building the application")
  endif()

  include(FindGLEW)
  if(GLEW_FOUND)
     target_link_libraries(sinoscope ${GLEW_LIBRARIES})
     target_include_directories(sinoscope PUBLIC ${GLEW_INCLUDE_DIRS})
     target_link_libraries(sinoscope-nomp ${GLEW_LIBRARIES})
     target_include_directories(sinoscope-nomp PUBLIC ${GLEW_INCLUDE_DIRS})
     target_link_libraries(sinoscope-nocl ${GLEW_LIBRARIES})
     target_include_directories(sinoscope-nocl PUBLIC ${GLEW_INCLUDE_DIRS})
  else()
     message(FATAL_ERROR "glew is required for building the application")
  endif()
endif()

include(FindPNG)
if(PNG_FOUND)
    target_link_libraries(sinoscope ${PNG_LIBRARIES})
    target_include_directories(sinoscope PUBLIC ${PNG_INCLUDE_DIRS})
    target_link_libraries(sinoscope-nocl ${PNG_LIBRARIES})
    target_include_directories(sinoscope-nocl PUBLIC ${PNG_INCLUDE_DIRS})
    target_link_libraries(sinoscope-nomp ${PNG_LIBRARIES})
    target_include_directories(sinoscope-nomp PUBLIC ${PNG_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "libpng is required for building the application")
endif()

set(OpenCLRoot ${PROJECT_SOURCE_DIR}/source/kernel)

set_source_files_properties(source/sinoscope-openmp.c PROPERTIES COMPILE_FLAGS -fopenmp)
add_definitions(-D__KERNEL_FILE__="${OpenCLRoot}/sinoscope.cl")
add_definitions(-D__OPENCL_INCLUDE__="${OpenCLRoot}")
add_definitions(-DCL_TARGET_OPENCL_VERSION=220)

if(DEFINED CLANG_INCLUDE_DIR)

add_executable(
    source-checker
    source/matcher/main.cpp
    source/matcher/file_search.cpp
    source/matcher/matchers.cpp
)

target_link_libraries(source-checker
    clangTooling
    clangFrontend
    clangSerialization
    clangDriver
    clangRewriteFrontend
    clangRewrite
    clangParse
    clangSema
    clangAnalysis
    clangAST
    clangASTMatchers
    clangEdit
    clangLex
    clangBasic
    LLVMSupport
    LLVMFrontendOpenMP
)

add_custom_target(check-source
  COMMAND ${CMAKE_CURRENT_BINARY_DIR}/source-checker
  -d ${CMAKE_CURRENT_SOURCE_DIR}
  -v ${VARIANT}
  2> /dev/null
)
add_dependencies(check-source source-checker)

endif()

add_custom_target(format
    COMMAND clang-format -i --verbose `find source -type f -iname '*.cl'`
    COMMAND clang-format -i --verbose `find source -type f -iname '*.c'` `find include -type f -iname '*.h'`
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)

add_custom_target(remise
    COMMAND tar -zcvf remise.tar.gz source/sinoscope-openmp.c source/sinoscope-opencl.c source/kernel/sinoscope.cl
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)

add_custom_target(check
    COMMAND ./sinoscope --check cl
    COMMAND ./sinoscope --check mp
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
add_dependencies(check sinoscope-nocl sinoscope-nomp)

install(TARGETS sinoscope-nocl sinoscope-nomp)
install(DIRECTORY source/kernel/ DESTINATION share/cl/sinoscope
        FILES_MATCHING PATTERN "*.cl")
