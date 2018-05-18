/*
thot package for statistical machine translation
Copyright (C) 2013 Daniel Ortiz-Mart\'inez
 
This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License
as published by the Free Software Foundation; either version 3
of the License, or (at your option) any later version.
 
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.
 
You should have received a copy of the GNU Lesser General Public License
along with this program; If not, see <http://www.gnu.org/licenses/>.
*/

/**
 * @file printAligFuncs.h
 * 
 * @brief Functions to print alignments.
 */

#ifndef _printAligFuncs
#define _printAligFuncs

//--------------- Include files --------------------------------------

#if HAVE_CONFIG_H
#  include <thot_config.h>
#endif /* HAVE_CONFIG_H */

#ifdef THOT__LARGEFILE_SOURCE
#ifndef _LARGEFILE_SOURCE
#define _LARGEFILE_SOURCE 1
#endif
#endif

#ifdef THOT__FILE_OFFSET_BITS
#ifndef _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS  THOT__FILE_OFFSET_BITS
#endif
#endif

#ifdef THOT__LARGE_FILES
#ifndef _LARGE_FILES 
#define _LARGE_FILES
#endif
#endif

#include <string>
#include <vector>
#include "StatModelDefs.h"
#include "WordAligMatrix.h"

//--------------- Constants ------------------------------------------


//--------------- typedefs -------------------------------------------


//--------------- function declarations ------------------------------

void printAlignmentInGIZAFormat(std::ostream &outS,
                                const std::vector<std::string> &ns,
                                const std::vector<std::string> &t,
                                WordAligMatrix waMatrix,
                                const char *header);
void printAlignmentInMyFormat(std::ostream &outS,
                              const std::vector<std::string> &ns,
                              const std::vector<std::string> &t,
                              WordAligMatrix waMatrix,
                              unsigned int numReps=1);

void printAlignmentInGIZAFormat(FILE* outf,
                                const std::vector<std::string> &ns,
                                const std::vector<std::string> &t,
                                WordAligMatrix waMatrix,
                                const char *header);
void printAlignmentInMyFormat(FILE* outf,
                              const std::vector<std::string> &ns,
                              const std::vector<std::string> &t,
                              WordAligMatrix waMatrix,
                              unsigned int numReps=1);

#endif
