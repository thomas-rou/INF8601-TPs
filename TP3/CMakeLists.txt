cmake_minimum_required(VERSION 3.16.0)
project(tp3 LANGUAGES C CXX)

set(CMAKE_C_FLAGS "${CMAKE_CFLAGS} -std=c99 -Wall")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_CFLAGS_RELEASE} -O2 -g")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fsanitize=address")
set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer")
set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fsanitize=address")

include_directories(heatsim PUBLIC include)
add_executable(heatsim
    source/cart.c
    source/color.c
    source/grid.c
    source/heatsim.c
    source/heatsim-mpi.c
    source/image.c
    source/main.c
)
target_link_libraries(heatsim PUBLIC -lm -lpng)

find_package(MPI REQUIRED)
separate_arguments(MPI_C_COMPILE_OPTIONS UNIX_COMMAND "${MPI_C_COMPILE_OPTIONS}")
separate_arguments(MPI_C_COMPILE_DEFINITIONS UNIX_COMMAND "${MPI_C_COMPILE_DEFINITIONS}")
separate_arguments(MPI_C_INCLUDE_DIRS UNIX_COMMAND "${MPI_C_INCLUDE_DIRS}")
separate_arguments(MPI_C_LINK_FLAGS UNIX_COMMAND "${MPI_C_LIBRARIES}")
separate_arguments(MPI_C_LIBRARIES UNIX_COMMAND "${MPI_C_LINK_FLAGS}")
target_compile_options(heatsim PUBLIC ${MPI_C_COMPILE_OPTIONS})
target_compile_definitions(heatsim PUBLIC ${MPI_C_COMPILE_DEFINITIONS})
target_include_directories(heatsim PUBLIC ${MPI_C_INCLUDE_DIRS})
target_link_options(heatsim PUBLIC ${MPI_C_LINK_FLAGS})
target_link_libraries(heatsim PUBLIC ${MPI_C_LIBRARIES})

add_custom_target(format
    COMMAND clang-format -i `find source -type f -iname '*.c'`
    COMMAND clang-format -i `find include -type f -iname '*.h'`
    COMMAND clang-format -i `find tests -type f -iname '*.c'`
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)

add_custom_target(remise
    COMMAND tar -zcvf remise.tar.gz source/heatsim-mpi.c
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)

add_custom_target(check
    COMMAND ${PROJECT_SOURCE_DIR}/scripts/run-check ${CMAKE_CURRENT_BINARY_DIR}/heatsim
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)

# ----- Source checker ----- #

if(DEFINED CLANG_INCLUDE_DIR)

add_executable(source-checker
    source/matcher/main.cpp
    source/matcher/file_search.cpp
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
)

add_dependencies(check-source source-checker)
add_dependencies(check check-source)

endif()

install(TARGETS heatsim)
