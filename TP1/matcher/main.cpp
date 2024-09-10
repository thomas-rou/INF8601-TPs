/** \file main.cpp
 * \brief Program checker entry point
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */


#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"

#include "llvm/Support/CommandLine.h"

#include "matchers.hpp"

static llvm::cl::OptionCategory llvmClCategory("Options");

static llvm::cl::opt<unsigned int> variant("v",
                                           llvm::cl::desc("Variant number"),
                                           llvm::cl::value_desc("variant"),
                                           llvm::cl::Required);

int main(int argc, const char** argv) {
    auto parser =
        clang::tooling::CommonOptionsParser::create(argc, argv, llvmClCategory);

    if (!parser) {
        llvm::errs() << parser.takeError();
        return -1;
    }

    auto& options_parser = parser.get();
    auto& db = options_parser.getCompilations();

    clang::tooling::ClangTool tool(db, options_parser.getSourcePathList());

    clang::ast_matchers::MatchFinder finder;

    FilterCallback filterChecker;

    finder.addMatcher(filterWithLambdaMatcher, &filterChecker);
    finder.addMatcher(lambdaMatcher, &filterChecker);
    finder.addMatcher(filterWithFunctorMatcher, &filterChecker);
    finder.addMatcher(filterInherits, &filterChecker);
    finder.addMatcher(functorDeclaration, &filterChecker);

    auto ret =
        tool.run(clang::tooling::newFrontendActionFactory(&finder).get());

    if (ret != 0) {
        throw std::runtime_error("Could not parse input file");
    }

    return filterChecker.assertVariant(variant);
}

