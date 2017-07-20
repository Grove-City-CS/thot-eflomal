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
 
/********************************************************************/
/*                                                                  */
/* Module: IncrPhraseModel                                          */
/*                                                                  */
/* Prototype file: IncrPhraseModel.h                                */
/*                                                                  */
/* Description: Defines the IncrPhraseModel class.                  */
/*              IncrPhraseModel implements a phrase model derived   */
/*              from _incrPhraseModel class.                        */
/*                                                                  */
/********************************************************************/

#ifndef _IncrPhraseModel_h
#define _IncrPhraseModel_h

//--------------- Include files --------------------------------------

#if HAVE_CONFIG_H
#  include <thot_config.h>
#endif /* HAVE_CONFIG_H */

#include "_incrPhraseModel.h"

using namespace std;

//--------------- Constants ------------------------------------------

	 
//--------------- function declarations ------------------------------


//--------------- Classes --------------------------------------------


//--------------- IncrPhraseModel class

class IncrPhraseModel: public _incrPhraseModel
{
 public:

    typedef _incrPhraseModel::SrcTableNode SrcTableNode;
    typedef _incrPhraseModel::TrgTableNode TrgTableNode;

        // Constructor
    IncrPhraseModel(void):_incrPhraseModel()
      {
        basePhraseTablePtr=new PhraseTable;
      }

        // Destructor
	~IncrPhraseModel();
	
 protected:

};

#endif
