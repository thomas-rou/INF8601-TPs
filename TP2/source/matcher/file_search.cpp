/** \file file_search.hpp
 * \brief Variant checker
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "file_search.hpp"

#include <fstream>
#include <iostream>
#include <regex>

std::string loadFile(const std::string& file_path) {
    std::ifstream f{file_path};
    if (!f.is_open()) {
        throw std::runtime_error("assertVariant() : Could not open file " +
                                 file_path);
    }

    std::stringstream ss;
    ss << f.rdbuf();

    return ss.str();
}

Requirements getVariantRequirements(unsigned int variant) {
    switch (variant) {
    case 1:
        return {{OmpConstruct::ParallelWithFor, OmpSchedule::Static},
                {OclParam::IntStructFloatStruct, OclDim::OneD}};
    case 2:
        return {{OmpConstruct::ParallelWithFor, OmpSchedule::Dynamic},
                {OclParam::IntStructFloatStruct, OclDim::TwoD}};
    case 3:
        return {{OmpConstruct::ParallelWithForSIMD, OmpSchedule::Static},
                {OclParam::Struct, OclDim::OneD}};
    case 4:
        return {{OmpConstruct::ParallelWithForSIMD, OmpSchedule::Dynamic},
                {OclParam::Struct, OclDim::TwoD}};
    case 5:
        return {{OmpConstruct::ParallelFor, OmpSchedule::Static},
                {OclParam::IntStructFloatOneByOne, OclDim::OneD}};
    case 6:
        return {{OmpConstruct::ParallelFor, OmpSchedule::Dynamic},
                {OclParam::IntStructFloatOneByOne, OclDim::TwoD}};
    case 7:
        return {{OmpConstruct::ParallelForSIMD, OmpSchedule::Static},
                {OclParam::FloatStructIntOneByOne, OclDim::OneD}};
    case 8:
        return {{OmpConstruct::ParallelForSIMD, OmpSchedule::Dynamic},
                {OclParam::FloatStructIntOneByOne, OclDim::TwoD}};
    default:
        throw std::runtime_error("Unexpected variant number");
    }
}

inline void assert(bool cond, const std::string& mess) {
    if (!cond) {
        throw std::runtime_error("Failed assertion, " + mess);
    }
}

void assertOmpVariant(const OmpImpl& requirements,
                      const std::string& file_path) {
    const std::regex omp_construct{
        R"(((?:#pragma omp)|(?:\[\[omp)) ((.+ ?)+))"};

    std::smatch matches;

    auto file = loadFile(file_path);

    bool has_simd = false;
    bool has_parallel_for = false;
    bool has_parallel = false;
    bool has_collapse = false;
    bool has_dynamic = false;
    bool has_static = false;
    bool has_guided = false;

    int num_pragmas = 0;

    auto begin = file.cbegin();
    while (std::regex_search(begin, file.cend(), matches, omp_construct)) {

        // std::cout << matches[0].str() << '\n';

        for (auto& match : matches) {
            auto str = match.str();

            has_simd |= (str.find("simd") != std::string::npos);
            has_parallel |= (str.find("parallel") != std::string::npos);
            has_parallel_for |= (str.find("parallel for") != std::string::npos);
            has_collapse |= (str.find("collapse") != std::string::npos);
            has_dynamic |= (str.find("dynamic") != std::string::npos);
            has_static |= (str.find("static") != std::string::npos);
            has_guided |= (str.find("has_guided") != std::string::npos);
        }

        ++num_pragmas;

        begin = matches.suffix().first;
    }

    switch (std::get<0>(requirements)) {
    case OmpConstruct::ParallelWithFor:
        assert(!has_simd && !has_parallel_for && has_parallel,
               "omp parallel attendu");
        break;
    case OmpConstruct::ParallelWithForSIMD:
        assert(has_simd && !has_parallel_for && has_parallel,
               "omp parallel attendu, avec for SIMD");
        break;
    case OmpConstruct::ParallelFor:
        assert(!has_simd && has_parallel_for && has_parallel,
               "omp parallel for attendu");
        break;
    case OmpConstruct::ParallelForSIMD:
        assert(has_simd && has_parallel_for && has_parallel,
               "omp parallel for attendu, avec for SIMD");
        break;
    }

    switch (std::get<1>(requirements)) {
    case OmpSchedule::Dynamic:
        assert(has_dynamic, "ordonnancement dynamique attendu");
        break;
    case OmpSchedule::Guided:
        assert(has_dynamic, "ordonnancement guidé attendu");
        break;
    case OmpSchedule::Static:
        assert(has_static || (!has_dynamic && !has_guided),
               "ordonnancement statique attendu");
        break;
    }
}

void assertOclVariant(const OclImpl& requirements,
                      const std::string& file_path) {
    const std::regex ocl_dim{R"(get_global_id\((.+)\);)"};

    std::smatch matches;

    auto file = loadFile(file_path);

    auto num_matches = 0u;
    std::string last_match;
    bool different = false; // True if two subsequent calls to get_global_id
                            // have different args

    auto begin = file.cbegin();
    while (std::regex_search(begin, file.cend(), matches, ocl_dim)) {

        auto dim = matches[1].str();

        if (num_matches == 0) {
            last_match = dim;
        }

        if (last_match != dim) {
            different = true;
        }

        ++num_matches;
        begin = matches.suffix().first;
    }

    switch (std::get<1>(requirements)) {
    case OclDim::OneD:
        assert(!different,
               "OpenCL: le calcul doit s'effectuer en une dimension");
        break;
    case OclDim::TwoD:
        assert(different,
               "OpenCL: le calcul doit s'effectuer en deux dimensions");
        break;
    }
}
