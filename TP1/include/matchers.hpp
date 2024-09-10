/** \file matchers.hpp
 * \brief Clang AST Matchers for automatic correction
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#pragma once

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

extern clang::ast_matchers::StatementMatcher filterWithLambdaMatcher;
extern clang::ast_matchers::StatementMatcher lambdaMatcher;
extern clang::ast_matchers::StatementMatcher filterWithFunctorMatcher;
extern clang::ast_matchers::DeclarationMatcher filterInherits;
extern clang::ast_matchers::DeclarationMatcher functorDeclaration;

class FilterCallback : public clang::ast_matchers::MatchFinder::MatchCallback {
  public:
    virtual void
    run(const clang::ast_matchers::MatchFinder::MatchResult& result) override;

    virtual ~FilterCallback() = default;

    /** \fn usesLambdas
     * \brief Returns true if the pipeline is built using only lambdas
     */
    bool usesLambdas() const;

    size_t lambdaCount() const { return lambda_count; }
    size_t functorCount() const { return functor_count; }

    /** \fn assertVariant
     * \brief Checks if the variant is respected
     */
    int assertVariant(unsigned int variant) const;

  private:
    size_t lambda_count = 0u;
    size_t functor_count = 0u;
};

