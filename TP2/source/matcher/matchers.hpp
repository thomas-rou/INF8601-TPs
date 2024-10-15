/** \file matchers.hpp
 * \brief Clang AST Matchers for automatic correction
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#pragma once

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

#include "file_search.hpp"

extern clang::ast_matchers::DeclarationMatcher kernelMatcher;

enum class ArgType { Aggregate, Pointer, Floating, Integral };

class KernelCallback : public clang::ast_matchers::MatchFinder::MatchCallback {
  public:
    virtual void
    run(const clang::ast_matchers::MatchFinder::MatchResult& result) override;

    virtual ~KernelCallback() = default;

    /** \fn assertVariant
     * \brief Checks if the variant is respected
     */
    void assertVariant(const OclImpl& requirements) const;

  private:
    bool has_matched = false;

    std::vector<ArgType> args;
};
