/** \file matchers.cpp
 * \brief Clang AST Matchers for automatic correction :: implementation
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include <array>

#include "matchers.hpp"

using namespace clang;
using namespace clang::ast_matchers;

// Variants that require a lambda

std::set<unsigned int> requires_lambdas = {2u, 4u, 5u, 8u, 9u};

// AST matchers

clang::ast_matchers::StatementMatcher lambdaMatcher =
    expr(anyOf(declRefExpr(
                   to(varDecl(hasInitializer(lambdaExpr())).bind("decl"))),
               lambdaExpr()),
         hasAncestor(functionDecl(hasName("pipeline_tbb"))))
        .bind("lambda");

clang::ast_matchers::StatementMatcher filterWithLambdaMatcher =
    cxxConstructExpr(
        hasDeclaration(namedDecl(hasName("filter_t"))),
        hasArgument(1,
                    lambdaMatcher)) // declRefExpr(to(cxxRecordDecl()))
        .bind("filterLambda");

clang::ast_matchers::StatementMatcher filterWithFunctorMatcher =
    cxxConstructExpr(
        hasDeclaration(namedDecl(hasName("filter_t"))),
        hasArgument(
            1, expr(anyOf(cxxConstructExpr(),
                          declRefExpr(
                              to(varDecl(hasInitializer(cxxConstructExpr()))
                                     .bind("decl")))))
                   .bind("expr"))) // declRefExpr(to(cxxRecordDecl()))
        .bind("filterFunctor");

clang::ast_matchers::DeclarationMatcher filterInherits =
    cxxRecordDecl(isDerivedFrom("tbb::filter"),
                  unless(anyOf(hasName("concrete_filter"),
                               hasName("thread_bound_filter"))))
        .bind("filterInherits");

clang::ast_matchers::DeclarationMatcher functorDeclaration =
    cxxRecordDecl(hasMethod(hasOverloadedOperatorName("()")),
                  unless(isLambda()))
        .bind("functorDeclaration");

// FilterCallback

bool FilterCallback::usesLambdas() const {
    return lambda_count > functor_count;
}

int FilterCallback::assertVariant(unsigned int variant) const {
    if (functor_count == 0 && lambda_count == 0) {
        llvm::errs()
            << "Impossible d'identifier la construction de tbb::filter_t\n";
        return 1;
    }

    auto variant_requires_lambdas = requires_lambdas.contains(variant);

    // Checks that it both expects lambdas and uses them, or neither
    if (variant_requires_lambdas == usesLambdas()) {
        return 0;
    }

    if (variant_requires_lambdas) {
        llvm::errs() << "Votre énoncé requiert l'utilisation de lambdas, mais "
                        "votre code n'en utilise pas";
    } else {
        llvm::errs() << "Votre énoncé requiert l'utilisation de classes, mais "
                        "votre code utilise des lambdas";
    }

    llvm::errs() << ". (" << lambda_count << " lambdas, " << functor_count
                 << " fonctors)\n";

    return -1;
}

void FilterCallback::run(
    const clang::ast_matchers::MatchFinder::MatchResult& result) {
    if (const auto* match =
            result.Nodes.getNodeAs<CXXConstructExpr>("filterLambda")) {

#ifdef DEBUG
        if (const auto* decl_ref = result.Nodes.getNodeAs<Decl>("lambda")) {
            // decl_ref->dump();
        }
        llvm::errs() << "LAMBDA : ";
        match->dump();
#endif

        ++lambda_count;
    } else if (const auto* match = result.Nodes.getNodeAs<Expr>("lambda")) {
#ifdef DEBUG
        llvm::errs() << "ORPHAN LAMBDA : ";
        match->dump();
#endif
        ++lambda_count;
    } else if (const auto* match =
                   result.Nodes.getNodeAs<CXXConstructExpr>("filterFunctor")) {
#ifdef DEBUG
        llvm::errs() << "FUNCTOR : ";
        match->dump();
#endif
        ++functor_count;
    } else if (const auto* match =
                   result.Nodes.getNodeAs<CXXRecordDecl>("filterInherits")) {
#ifdef DEBUG
        llvm::errs() << "FUNCTOR : ";
        match->dump();
#endif
        ++functor_count;
    } else if (const auto* match = result.Nodes.getNodeAs<CXXRecordDecl>(
                   "functorDeclaration")) {
        if (result.Context->getSourceManager().isInMainFile(
                match->getBeginLoc())) {
#ifdef DEBUG
            llvm::errs() << "FUNCTOR : ";
            match->dump();
#endif
            ++functor_count;
        }
    } else {
        llvm::errs() << "Match non reconnu, contactez votre chargé de lab\n";
    }
}
