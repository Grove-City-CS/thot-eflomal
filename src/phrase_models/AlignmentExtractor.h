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
 
/* ------------------------------------------ */
/*                                            */
/* AlignmentExtractor class                   */
/*                                            */
/* ------------------------------------------ */

#ifndef _AlignmentExtractor_h
#define _AlignmentExtractor_h

//--------------- Include files --------------------------------------

#if HAVE_CONFIG_H
#  include <thot_config.h>
#endif /* HAVE_CONFIG_H */

#include "PhraseDefs.h"
#include "myVector.h"
#include <string.h>
#include <string>
#include <fstream>
#include <iostream>
#include "awkInputStream.h"
#include "WordAligMatrix.h"
#include "printAligFuncs.h"

//--------------- Constants ------------------------------------------

#define ENDOFFILE 0
#define NO_ERRORS 1
#define ALIG_OP_FILE_ERROR -1
#define GIZA_ALIG_FILE_FORMAT 0
#define ALIG_OP_FILE_FORMAT 1


//--------------- typedefs -------------------------------------------

struct aligOpDescription
{
 string aligOperator;
 string GizaAligFile;
};

//--------------- Classes -----------------------------------------------

class AlignmentExtractor;

//--------------- function declarations----------------------------------

ostream& operator << (ostream &outS,AlignmentExtractor &ae);

//--------------- AlignmentExtractor class: class for extracting
//                sentence pair alignments from a GIZA xxx.A3.final file

class AlignmentExtractor
{
 public:

        // Constructor
	AlignmentExtractor(void);
    AlignmentExtractor(const AlignmentExtractor &alExt);
    AlignmentExtractor& operator= (const AlignmentExtractor &alExt);

        // Functions to manipulate files
    bool open(const char *str,
              unsigned int _fileFormat=GIZA_ALIG_FILE_FORMAT);
    bool open_stream(FILE *stream,
                     unsigned int _fileFormat=GIZA_ALIG_FILE_FORMAT);
	void close(void);
	bool rewind(void);
	bool getNextAlignment(void);

        // Functions to access alignment information
    Vector<string>& get_ns(void);
    Vector<string>& get_t(void);
    WordAligMatrix& get_wamatrix(void);
    float get_numReps(void);

        // Functions to operate alignments
    void transposeAlig(void);
	bool join(const char *GizaAligFileName,
              const char *outFileName,
              bool transpose=0,
              bool verbose=0); 
	    // joins the alignment matrices given in the GIZA file (one to
        // one correspondence between alignments is assumed and prints
        // it to outF.
	bool intersect(const char *GizaAligFileName,
                   const char *outFileName,
                   bool transpose=0,
                   bool verbose=0);
        // intersects the alignment matrixes
    bool sum(const char *GizaAligFileName,
             const char *outFileName,
             bool transpose=0,
             bool verbose=0);
        // Obtains the sum of the alignment matrixes
	bool symmetr1(const char *GizaAligFileName,
                  const char *outFileName,
                  bool transpose=0,
                  bool verbose=0);
	bool symmetr2(const char *GizaAligFileName,
                  const char *outFileName,
                  bool transpose=0,
                  bool verbose=0);
	bool growDiagFinal(const char *GizaAligFileName,
                       const char *outFileName,
                       bool transpose=0,
                       bool verbose=0);
	
        // Destructor
	~AlignmentExtractor();
	
 private:
    Vector<string> ns;
    Vector<string> t;
    WordAligMatrix wordAligMatrix;	
    float numReps;
    unsigned int fileFormat;
    FILE* fileStream;
    awkInputStream awkInpStrm;
    
    bool getNextAlignInGIZAFormat(void);
    bool getNextAlignInAlignOpFormat(void);
	
};

#endif
