/** \file file_search.hpp
 * \brief Variant checker
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#pragma once

#include <string>
#include <tuple>

enum class OmpConstruct {
    ParallelWithFor,
    ParallelWithForSIMD,
    ParallelFor,
    ParallelForSIMD
};

enum class OmpSchedule { Static, Dynamic, Guided };

using OmpImpl = std::tuple<OmpConstruct, OmpSchedule>;

enum class OclParam {
    IntStructFloatStruct,
    OneByOne,
    Struct,
    IntStructFloatOneByOne,
    FloatStructIntOneByOne
};

enum class OclDim { OneD, TwoD };

using OclImpl = std::tuple<OclParam, OclDim>;

using Requirements = std::pair<OmpImpl, OclImpl>;

/** \fn assertVariant
 * \brief Verifies that a variant is respected, or throws
 */
Requirements getVariantRequirements(unsigned int variant);

void assertOmpVariant(const OmpImpl& requirements,
                      const std::string& file_path);

void assertOclVariant(const OclImpl& requirements,
                      const std::string& file_path);
