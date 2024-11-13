/** \file main.cpp
 * \brief Program checker entry point
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"

#include "llvm/Support/CommandLine.h"

#include "file_search.hpp"

#include <array>
#include <iostream>

static llvm::cl::OptionCategory llvmClCategory("Options");

static llvm::cl::opt<unsigned int> variant("v",
                                           llvm::cl::desc("Variant number"),
                                           llvm::cl::value_desc("variant"),
                                           llvm::cl::Required);

static llvm::cl::opt<std::string> directory("d",
                                            llvm::cl::desc("Lab directory"),
                                            llvm::cl::value_desc("dir"),
                                            llvm::cl::Required);

int main(int argc, const char** argv) {
    llvm::cl::ParseCommandLineOptions(argc, argv);

    // Grep-like search for some constructs

    auto path = directory.getValue() + "/source/heatsim-mpi.c";

    auto req = getVariantRequirements(variant.getValue());

    std::cout << req;

    assertVariant(req, path);

    llvm::outs() << "\n\nChecker OK\n";
    return 0;
}
