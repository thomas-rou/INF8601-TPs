cmake_minimum_required(VERSION 3.13.5)
project(tp1 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-fno-rtti")

include_directories(include)

add_executable(pipeline)
target_link_libraries(pipeline -lm -pthread -lpng -ltbb)
target_sources(pipeline PUBLIC
    source/filter.c
    source/image.c
    source/main.c
    source/pipeline-pthread.c
    source/pipeline-serial.c
    source/pipeline-tbb.cpp
    source/queue.c
)
# For macros with __FILE__
target_compile_options(pipeline PUBLIC "-fmacro-prefix-map=${CMAKE_SOURCE_DIR}/=")

add_executable(pipeline-notbb)
target_link_libraries(pipeline-notbb -lm -pthread -lpng)
target_sources(pipeline-notbb PUBLIC
    source/filter.c
    source/image.c
    source/main.c
    source/pipeline-pthread.c
    source/pipeline-serial.c
    source/queue.c
)
# For macros with __FILE__
target_compile_options(pipeline-notbb PUBLIC "-fmacro-prefix-map=${CMAKE_SOURCE_DIR}/=")

if (DEFINED CLANG_INCLUDE_DIR)
add_executable(source-checker
    matcher/main.cpp
    matcher/matchers.cpp
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
endif()

add_custom_target(format
    COMMAND clang-format -i `find source -type f -iname '*.c'` `find include -type f -iname '*.h'`
    COMMAND clang-format -i `find source -type f -iname '*.cpp'` `find include -type f -iname '*.hpp'`
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)

add_custom_target(remise
    COMMAND tar -zcvf remise.tar.gz source/pipeline-pthread.c source/pipeline-tbb.cpp
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)

add_custom_target(run-serial
    COMMAND time ${CMAKE_CURRENT_BINARY_DIR}/pipeline --directory ${PROJECT_SOURCE_DIR}/data --pipeline serial
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)
add_dependencies(run-serial pipeline)

add_custom_target(run-pthread
    COMMAND time ${CMAKE_CURRENT_BINARY_DIR}/pipeline --directory ${PROJECT_SOURCE_DIR}/data --pipeline pthread
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)
add_dependencies(run-pthread pipeline)

add_custom_target(run-tbb
    COMMAND time ${CMAKE_CURRENT_BINARY_DIR}/pipeline --directory ${PROJECT_SOURCE_DIR}/data --pipeline tbb
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)
add_dependencies(run-tbb pipeline)

add_custom_target(run-all
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)
add_dependencies(run-all run-serial run-pthread run-tbb)

add_custom_target(generate-image
    COMMAND ./data/generate-random ./data/0000.png
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)

if (DEFINED CLANG_INCLUDE_DIR)
add_custom_target(check-source
  COMMAND ${CMAKE_CURRENT_BINARY_DIR}/source-checker
  ${PROJECT_SOURCE_DIR}/source/pipeline-tbb.cpp
  --extra-arg=-I${CLANG_INCLUDE_DIR}
  -p ${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json
  -v ${VARIANT}
)
add_dependencies(check-source source-checker)
endif()

add_custom_target(check
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/pipeline-notbb --directory ${PROJECT_SOURCE_DIR}/data --pipeline pthread
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/pipeline --directory ${PROJECT_SOURCE_DIR}/data --pipeline tbb
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/pipeline --directory ${PROJECT_SOURCE_DIR}/data --pipeline serial
    COMMAND ./data/check.sh
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)
if (DEFINED CLANG_INCLUDE_DIR)
add_dependencies(check check-source generate-image)
else()
add_dependencies(check generate-image)
endif()

install(TARGETS pipeline pipeline-notbb)
