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
 * @file ModelDescriptorUtils.h
 * @brief Defines string processing utilities
 */

#ifndef _ModelDescriptorUtils_h
#define _ModelDescriptorUtils_h

#if HAVE_CONFIG_H
#  include <thot_config.h>
#endif /* HAVE_CONFIG_H */

#include "ErrorDefs.h"
#include "awkInputStream.h"
#include <stdio.h>
#include "myVector.h"
#include <string>

//--------------- typedefs -------------------------------------------

struct ModelDescriptorEntry
{
  std::string modelType;
  std::string modelFileName;
  std::string statusStr;
  std::string absolutizedModelFileName;
};

//--------------- Functions ------------------------------------------

std::string absolutizeModelFileName(std::string descFileName,
                                    std::string modelFileName);
std::string extractDirName(std::string filePath);
bool fileIsDescriptor(std::string fileName,
                      std::string& mainFileName);
bool extractModelEntryInfo(const char *fileName,
                           Vector<ModelDescriptorEntry>& modelDescEntryVec);

#endif
