/** \file file_search.hpp
 * \brief Variant checker
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#pragma once

#include "variants.hpp"
#include <string>

/** \fn getVariantRequirements
 * \brief returns the variant requirements
 */
Variant getVariantRequirements(unsigned int variant);

void assertVariant(const Variant& variant, const std::string& path);
