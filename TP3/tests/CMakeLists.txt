add_executable(grid
    ../source/grid.c
    grid.c
)
add_custom_target(test_grid
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/grid
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)
add_dependencies(test_grid grid)

add_executable(cart
    ../source/grid.c
    ../source/cart.c
    cart.c
)
add_custom_target(test_cart
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/cart
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)
add_dependencies(test_cart cart)

add_custom_target(tests)
add_dependencies(tests test_grid test_cart)
