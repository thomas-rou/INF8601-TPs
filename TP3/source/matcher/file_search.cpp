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

inline void assert(bool cond, const std::string& mess) {
    if (!cond) {
        throw std::runtime_error("Failed assertion, " + mess);
    }
}

std::ostream& operator<<(std::ostream& out, const MpiType& ty) {
    switch (ty) {
    case MpiType::Contiguous:
        out << "`MPI_Type_contiguous`";
        break;
    case MpiType::Vector:
        out << "`MPI_Type_vector`";
        break;
    case MpiType::Double:
        out << "`MPI_DOUBLE`";
        break;
    case MpiType::Struct:
        out << "`MPI_Type_struct`";
        break;
    case MpiType::Unsigned:
        out << "`MPI_UNSIGNED`";
        break;
    }

    return out;
}

std::ostream& operator<<(std::ostream& out, const SendType& s_ty) {
    switch (s_ty) {
    case SendType::Sync:
        out << "`MPI_Send` et `MPI_Recv`";
        break;
    case SendType::Async:
        out << "`MPI_Isend` et `MPI_Irecv`";
        break;
    }

    return out;
}

std::ostream& operator<<(std::ostream& out, const Variant& v) {
    out << "L'envoi/réception des grilles initiales doit être effectué avec "
        << v.parameters.send_type << '\n';
    out << "Vous devez envoyer les paramètres `width`, `height` et `padding` "
           "en une seule requête de type "
        << v.parameters.params_type << '\n';
    out << "Les données (`data`) de la grille doivent être envoyées en une "
           "seule requête de type défini avec "
        << v.parameters.grid_type << '\n';

    out << "\nL'échange des bordures doit être effectué avec "
        << v.borders.send_type << '\n';
    out << "Les bordures nord et sud doivent être de type défini avec  "
        << v.borders.north_south << '\n';
    out << "Les bordures est et ouest doivent être de type défini avec  "
        << v.borders.east_west << '\n';

    out << "\nL'envoi et la réception de la grille finale doit être effectué "
           "avec "
        << v.grid.send_type << '\n';
    out << "Les données (`data`) de la grille doivent être de type "
        << v.grid.type << '\n';

    return out;
}

Variant getVariantRequirements(unsigned int variant) {
    Variant requirements;

    --variant;
    assert(variant < NUM_VARIANTS,
           "le numéro de variant ne correspond à aucune valeur connue");

    // Constants
    requirements.parameters.grid_type = MpiType::Struct;
    requirements.borders.east_west = MpiType::Vector;

    // Scan through all the bits set by the variant since we're dealing with
    // binary options

    auto bit = 1u;

    requirements.grid.type =
        (variant & bit) ? MpiType::Double : MpiType::Struct;

    bit <<= 1u;
    requirements.grid.send_type =
        (variant & bit) ? SendType::Async : SendType::Sync;

    bit <<= 1u;
    requirements.borders.north_south =
        (variant & bit) ? MpiType::Double : MpiType::Contiguous;

    bit <<= 1u;
    requirements.borders.send_type =
        (variant & bit) ? SendType::Async : SendType::Sync;

    bit <<= 1u;
    requirements.parameters.params_type =
        (variant & bit) ? MpiType::Struct : MpiType::Unsigned;

    bit <<= 1u;
    requirements.parameters.send_type =
        (variant & bit) ? SendType::Async : SendType::Sync;

    return requirements;
}

inline size_t countMatches(const std::string& text, const std::regex& regex) {
    return std::distance(std::sregex_iterator(text.begin(), text.end(), regex),
                         std::sregex_iterator());
}

void assertVariant(const Variant& variant, const std::string& path) {
    auto content = loadFile(path);

    std::map<MpiType, unsigned int> expected_nums;
    auto types = {variant.parameters.params_type,
                  variant.parameters.grid_type, variant.borders.east_west,
                  variant.borders.north_south, variant.grid.type};

    for (auto& ty : types) {
        ++expected_nums[ty];
    }

    const std::regex contiguous{"MPI_Type_contiguous"};
    const std::regex vector{R"(MPI_Type_vector)"};
    const std::regex create_struct{R"(MPI_Type_create_struct)"};

    const std::regex sync{R"(MPI_Recv)"};
    const std::regex async{R"(MPI_Irecv)"};

    // We won't count scalar types as they are reused throughout the code to
    // create composite types (struct, vector, ..)

    auto checker = [&content, &expected_nums](auto val, const auto& regex,
                                              const std::string& err) {
        auto expected = expected_nums[val];
        auto actual = countMatches(content, regex);
        try {
            assert(actual >= expected, err);
        } catch (std::runtime_error& e) {
            std::cerr << "\nType error for " << val
                      << "\n\tExpected : " << expected << ", got " << actual
                      << '\n';
            throw e;
        }
    };

    checker(MpiType::Contiguous, contiguous,
            "Unexpected number of contiguous data types");
    checker(MpiType::Vector, vector, "Unexpected number of vector data types");
    checker(MpiType::Struct, create_struct,
            "Unexpected number of struct data types");

    // Sync checker

    std::map<SendType, unsigned int> expected_sync;
    auto send = {variant.parameters.send_type, variant.borders.send_type,
                 variant.grid.send_type};
    for (auto& ty : send) {
        ++expected_sync[ty];
    }

    auto sync_checker = [&content, &expected_sync](auto val, const auto& regex,
                                                   const std::string& err) {
        auto expected = expected_sync[val];
        auto actual = countMatches(content, regex);
        try {
            assert(actual >= expected, err);
        } catch (std::runtime_error& e) {
            std::cerr << "\nSync error for " << val
                      << "\n\tExpected : " << expected << ", got " << actual
                      << '\n';
            throw e;
        }
    };

    sync_checker(SendType::Sync, sync,
                 "Unexpected number of MPI_Send/MPI_Recv");
    sync_checker(SendType::Async, async,
                 "Unexpected number of MPI_Isend/MPI_Irecv");
}
