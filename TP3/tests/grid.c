#include <assert.h>

#include "grid.h"

#define SIZE_X 3231
#define SIZE_Y 1234

int main() {
    grid_t* grid1 = grid_create(SIZE_X, SIZE_Y, 0);
    assert(grid1 != NULL);
    assert(grid1->data != NULL);
    assert(grid1->width == SIZE_X);
    assert(grid1->height == SIZE_Y);
    assert(grid1->padding == 0);
    assert(grid1->width_padded == SIZE_X);
    assert(grid1->height_padded == SIZE_Y);

    for (int j = 0; j < SIZE_Y; j++) {
        for (int i = 0; i < SIZE_X; i++) {
            *grid_get_cell(grid1, i, j) = i + j * SIZE_X;
        }
    }

    grid_t* grid2 = grid_clone(grid1);
    assert(grid2 != NULL);
    assert(grid2->data != NULL);
    assert(grid2->width == SIZE_X);
    assert(grid2->height == SIZE_Y);
    assert(grid2->padding == 0);
    assert(grid2->width_padded == SIZE_X);
    assert(grid2->height_padded == SIZE_Y);

    for (int j = 0; j < SIZE_Y; j++) {
        for (int i = 0; i < SIZE_X; i++) {
            assert(*grid_get_cell(grid2, i, j) == i + j * SIZE_X);
        }
    }

    grid_t* grid3 = grid_clone_with_padding(grid2, 1);
    assert(grid3 != NULL);
    assert(grid3->data != NULL);
    assert(grid3->width == SIZE_X);
    assert(grid3->height == SIZE_Y);
    assert(grid3->padding == 1);
    assert(grid3->width_padded == SIZE_X + 2);
    assert(grid3->height_padded == SIZE_Y + 2);

    for (int j = 0; j < SIZE_Y; j++) {
        for (int i = 0; i < SIZE_X; i++) {
            assert(*grid_get_cell(grid3, i, j) == i + j * SIZE_X);
        }
    }

    grid_t* grid4 = grid_create(SIZE_X / 2, SIZE_Y / 2, 1);
    assert(grid4 != NULL);
    assert(grid4->data != NULL);
    assert(grid4->width == SIZE_X / 2);
    assert(grid4->height == SIZE_Y / 2);
    assert(grid4->padding == 1);
    assert(grid4->width_padded == SIZE_X / 2 + 2);
    assert(grid4->height_padded == SIZE_Y / 2 + 2);

    assert(grid_copy_block(grid3, 0, 0, SIZE_X / 2, SIZE_Y / 2, grid4, 0, 0) == 0);

    for (int j = 0; j < SIZE_Y / 2; j++) {
        for (int i = 0; i < SIZE_X / 2; i++) {
            assert(*grid_get_cell(grid4, i, j) == i + j * SIZE_X);
        }
    }

    assert(grid_set(grid4, 1420.0) == 0);

    for (int j = 0; j < SIZE_Y / 2 + 2; j++) {
        for (int i = 0; i < SIZE_X / 2 + 2; i++) {
            assert(*grid_get_cell_padded(grid4, i, j) == 1420.0);
        }
    }

    for (int j = 0; j < SIZE_Y / 2; j++) {
        for (int i = 0; i < SIZE_X / 2; i++) {
            *grid_get_cell(grid4, i, j) = 420.0;
        }
    }

    *grid_get_cell_padded(grid4, 0, 0)                           = 420.0;
    *grid_get_cell_padded(grid4, SIZE_X / 2 + 1, 0)              = 420.0;
    *grid_get_cell_padded(grid4, SIZE_X / 2 + 1, SIZE_Y / 2 + 1) = 420.0;
    *grid_get_cell_padded(grid4, 0, SIZE_Y / 2 + 1)              = 420.0;

    assert(grid_set_padding_from_inner_bound(grid4) == 0);

    for (int j = 0; j < SIZE_Y / 2 + 2; j++) {
        for (int i = 0; i < SIZE_X / 2 + 2; i++) {
            assert(*grid_get_cell_padded(grid4, i, j) == 420.0);
        }
    }

    assert(grid_multiply(grid4, 2.0) == 0);

    for (int j = 0; j < SIZE_Y / 2 + 2; j++) {
        for (int i = 0; i < SIZE_X / 2 + 2; i++) {
            assert(*grid_get_cell_padded(grid4, i, j) == 420.0 * 2.0);
        }
    }

    *grid_get_cell(grid4, SIZE_X / 4, SIZE_Y / 4) = 2000.0;
    assert(grid_max(grid4) == 2000.0);

    /* TODO: test `grid_copy_inner_border` */

    grid_destroy(grid1);
    grid_destroy(grid2);
    grid_destroy(grid3);
    grid_destroy(grid4);
}