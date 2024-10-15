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
#include "matchers.hpp"

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

static llvm::cl::opt<std::string> extra_args("extra-arg", llvm::cl::init(""));

int main(int argc, const char** argv) {
    llvm::cl::ParseCommandLineOptions(argc, argv);

    // First pass : grep-like search for some constructs

    auto omp_path = directory.getValue() + "/source/sinoscope-openmp.c";
    auto kernel_path = directory.getValue() + "/source/kernel/sinoscope.cl";

    auto req = getVariantRequirements(variant.getValue());

    try {
        assertOmpVariant(req.first, omp_path);
        assertOclVariant(req.second, kernel_path);
    } catch (std::exception& e) {
        std::cout << "Consigne : " << e.what();
        return -1;
    }

    // Second pass : AST matchers for arg passing

    int fake_argc = 7;
    const char* fake_argv[] = {"clang-tool",
                               "--",
                               "clang",
                               "-Xclang",
                               "-finclude-default-header",
                               "-c",
                               kernel_path.c_str()};
    std::string err;

    auto db = clang::tooling::FixedCompilationDatabase::loadFromCommandLine(
        fake_argc, fake_argv, err);

    clang::tooling::ClangTool tool(*db, kernel_path);

    clang::ast_matchers::MatchFinder finder;

    KernelCallback kernelChecker;

    finder.addMatcher(kernelMatcher, &kernelChecker);

    tool.run(clang::tooling::newFrontendActionFactory(&finder).get());

    try {
        kernelChecker.assertVariant(req.second);
    } catch (std::exception& e) {
        std::cout << "Consigne : " << e.what();
        return -1;
    }

    llvm::outs() << "Checker OK\n";
    return 0;
}
