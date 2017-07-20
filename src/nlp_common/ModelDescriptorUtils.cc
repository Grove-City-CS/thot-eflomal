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
 

#include "ModelDescriptorUtils.h"

//---------------
std::string absolutizeModelFileName(std::string descFileName,
                                    std::string modelFileName)
{
  if(modelFileName.empty())
    return modelFileName;
  else
  {
        // Check if path is already absolute
    if(modelFileName[0]=='/')
    {
          // Path is absolute
      return modelFileName;
    }
    else
    {
          // Path is not absolute
      if(descFileName.empty())
      {
        return modelFileName;
      }
      else
      {
            // Absolutize model file name using directory name contained
            // in descriptor file path
        return extractDirName(descFileName)+modelFileName;
      }
    }
  }
}

//---------------
std::string extractDirName(std::string filePath)
{
  if(filePath.empty())
  {
    std::string dirName;
    return dirName;
  }
  else
  {
        // Provided file path is not empty
    int last_slash_pos=-1;

        // Find last position of slash symbol
    for(unsigned int i=0;i<filePath.size();++i)
      if(filePath[i]=='/')
        last_slash_pos=i;

        // Check if any slash symbols were found
    if(last_slash_pos==-1)
    {
      std::string dirName;
      return dirName;
    }
    else
    {
          // The last slash symbol was found at "last_slash_pos"
      return filePath.substr(0,last_slash_pos+1);
    }
  }
}

//---------------
bool fileIsDescriptor(std::string fileName,
                      std::string& mainFileName)
{
  awkInputStream awk;
  if(awk.open(fileName.c_str())==ERROR)
    return false;
  else
  {
    if(awk.getln())
    {
      if(awk.NF>=3 && awk.dollar(1)=="thot" && (awk.dollar(2)=="tm" || awk.dollar(2)=="lm") && awk.dollar(3)=="descriptor")
      {
            // Process descriptor (main file will be read)
        while(awk.getln())
        {
          if(awk.NF>=3 && awk.dollar(3)=="main")
          {
                // File is a descriptor and main file was found
            mainFileName=awk.dollar(2);
            awk.close();
            return true;
          }
          else
          {
                // Check if old descriptor file is being processed
            if(awk.NF>=3 && awk.dollar(2)=="main")
            {
                  // File is an old descriptor and main file was found
              mainFileName=awk.dollar(1);
              awk.close();
              return true;
            }
          }
        }
            // File is not a descriptor since it does not incorporate a
            // main model
        return false;
      }
      else
      {
            // File is not a descriptor
        awk.close();
        return false;
      }
    }
    else
    {
          // File is empty
      awk.close();
      return false;
    }
  }
}

//---------------
bool extractModelEntryInfo(const char *fileName,
                           Vector<ModelDescriptorEntry>& modelDescEntryVec)
{
  std::string mainFileName;
  if(fileIsDescriptor(fileName,mainFileName))
  {
    awkInputStream awk;
    if(awk.open(fileName)==ERROR)
    {
      return ERROR;
    }
    else
    {
          // Discard first line (it is used to identify the file as a
          // descriptor)
      awk.getln();
    
          // Read entries for each language model
      while(awk.getln())
      {
        if(awk.dollar(1)!="#")
        {
          if(awk.NF>=3)
          {
                // Read entry
            ModelDescriptorEntry modelDescEntry;
            modelDescEntry.modelType=awk.dollar(1);
            modelDescEntry.modelFileName=awk.dollar(2);
            modelDescEntry.statusStr=awk.dollar(3);
            modelDescEntry.absolutizedModelFileName=absolutizeModelFileName(fileName,modelDescEntry.modelFileName);
            modelDescEntryVec.push_back(modelDescEntry);
          }
        }
      }
      return OK;
    }
  }
  else
    return ERROR;
}
