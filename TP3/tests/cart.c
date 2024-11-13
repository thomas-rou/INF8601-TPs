#include <assert.h>

#include "cart.h"

#define DIM_X 11
#define DIM_Y 23

#define SIZE_X 2132
#define SIZE_Y 4421

int main() {
    cart2d_t* cart1 = cart2d_create(DIM_X, DIM_Y, SIZE_X, SIZE_Y);
    assert(cart1 != NULL);
    assert(cart1->grids != NULL);
    assert(cart1->grid_x_count == DIM_X);
    assert(cart1->grid_y_count == DIM_Y);
    assert(cart1->total_width == SIZE_X);
    assert(cart1->total_height == SIZE_Y);

    for (int dj = 0; dj < DIM_Y; dj++) {
        size_t total_width = 0;
        for (int di = 0; di < DIM_X; di++) {
            total_width += cart2d_get_grid(cart1, di, dj)->width;
        }
        assert(total_width == SIZE_X);
    }

    for (int di = 0; di < DIM_X; di++) {
        size_t total_height = 0;
        for (int dj = 0; dj < DIM_Y; dj++) {
            total_height += cart2d_get_grid(cart1, di, dj)->height;
        }
        assert(total_height == SIZE_Y);
    }

    for (int dj = 0; dj < DIM_Y; dj++) {
        for (int di = 0; di < DIM_X; di++) {
            grid_t* grid = cart2d_get_grid(cart1, di, dj);
            grid_set(grid, di + dj * DIM_X);
        }
    }

    grid_t* grid1 = cart2d_to_grid(cart1);
    assert(grid1 != NULL);
    assert(grid1->width == SIZE_X);
    assert(grid1->height == SIZE_Y);
    assert(grid1->padding == 0);

    for (int dj = 0; dj < DIM_Y; dj++) {
        for (int di = 0; di < DIM_X; di++) {
            size_t x_off = cart1->x_offsets[di];
            size_t y_off = cart1->y_offsets[dj];

            grid_t* grid = cart2d_get_grid(cart1, di, dj);

            for (int j = 0; j < grid->height; j++) {
                for (int i = 0; i < grid->width; i++) {
                    double value = *grid_get_cell(grid1, x_off + i, y_off + j);
                    assert(value == di + dj * DIM_X);
                }
            }
        }
    }

    cart2d_destroy(cart1);
    grid_destroy(grid1);
}
