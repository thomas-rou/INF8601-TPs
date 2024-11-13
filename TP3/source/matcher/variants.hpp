/** \file variants.hpp
 * \brief Variants types definitions
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#pragma once

#include <iostream>

constexpr auto NUM_VARIANTS = 64u;

enum class SendType {
    Sync,
    Async,
};

enum class MpiType {
    Contiguous,
    Vector,
    Double,
    Struct,
    Unsigned,
};

struct Variant {
    struct Borders {
        SendType send_type;
        MpiType north_south; // Only Double or contiguous
        MpiType east_west;   // Always vector
    };

    struct Parameters {
        SendType send_type;
        MpiType params_type;
        MpiType grid_type;
    };

    struct Grid {
        SendType send_type;
        MpiType type;
    };

    Parameters parameters;
    Borders borders;
    Grid grid;
};

std::ostream& operator<<(std::ostream&, const Variant&);
