/*
thot package for statistical machine translation
Copyright (C) 2017 Adam Harasimowicz

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
/* Module: LevelDbNgramTableTest                                    */
/*                                                                  */
/* Prototypes file: LevelDbNgramTableTest.h                         */
/*                                                                  */
/* Description: Declares the LevelDbNgramTableTest class            */
/*              implementing unit tests for the LevelDbNgramTable   */
/*              class.                                              */
/*                                                                  */
/********************************************************************/

/**
 * @file LevelDbNgramTableTest.h
 *
 * @brief Declares the LevelDbNgramTable class implementing unit tests
 * for the LevelDbNgramTable class.
 */

#ifndef _LevelDbNgramTableTest_h
#define _LevelDbNgramTableTest_h

//--------------- Include files --------------------------------------

#include "ErrorDefs.h"
#include "MathDefs.h"

#if HAVE_CONFIG_H
#  include <thot_config.h>
#endif /* HAVE_CONFIG_H */

#include "im_pair.h"
#include "LevelDbNgramTable.h"
#include <cppunit/extensions/HelperMacros.h>

//--------------- Constants ------------------------------------------

//--------------- typedefs -------------------------------------------


//--------------- Classes --------------------------------------------

//--------------- LevelDbNgramTable template class

/**
 * @brief Class implementing tests for LevelDbNgramTable.
 */

class LevelDbNgramTableTest: public CppUnit::TestFixture
{
        CPPUNIT_TEST_SUITE( LevelDbNgramTableTest );
        CPPUNIT_TEST( testAddTrgWithEmptySrc );
        CPPUNIT_TEST( testCTrg );
        CPPUNIT_TEST( testLogCountRetrieving );
        CPPUNIT_TEST( testStoreFloatValues );
        CPPUNIT_TEST( testKeyVectorConversion );
        CPPUNIT_TEST( testAddTableEntry );
        CPPUNIT_TEST( testIncrCountsOfEntryLog );
        CPPUNIT_TEST( testStoreAndRestoreSrcInfo );
        CPPUNIT_TEST( testGetEntriesForTarget );
        CPPUNIT_TEST( testRetrievingSubphrase );
        CPPUNIT_TEST( testGetEntriesForSource );
        CPPUNIT_TEST( testRetrievingEntriesWithCountEqualZero );
        CPPUNIT_TEST( testGetNbestForSrc );
        CPPUNIT_TEST( testGetNbestForTrg );
        CPPUNIT_TEST( testIteratorsLoop );
        CPPUNIT_TEST( testIteratorsOperatorsPlusPlusStar );
        CPPUNIT_TEST( testIteratorsOperatorsEqualNotEqual );
        CPPUNIT_TEST( testSize );
        CPPUNIT_TEST( testLoadingLevelDb );
        CPPUNIT_TEST( testLoadedDataCorrectness );
        CPPUNIT_TEST( testLoadedDataNullCount );
        CPPUNIT_TEST_SUITE_END();

    private:
        LevelDbNgramTable *tab;
        std::string dbName = "/tmp/thot_leveldb_unit_test";
  
        std::vector<WordIndex> getVector(std::string phrase);

    public:
        void setUp();
        void tearDown();

        void testAddTrgWithEmptySrc();
        void testCTrg();
        void testLogCountRetrieving();
        void testKeyVectorConversion();
        void testStoreFloatValues();
        void testAddTableEntry();
        void testIncrCountsOfEntryLog();
        void testStoreAndRestoreSrcInfo();
        void testGetEntriesForTarget();
        void testRetrievingSubphrase();
        void testGetEntriesForSource();
        void testRetrievingEntriesWithCountEqualZero();
        void testGetNbestForSrc();
        void testGetNbestForTrg();
        void testIteratorsLoop();
        void testIteratorsOperatorsPlusPlusStar();
        void testIteratorsOperatorsEqualNotEqual();
        void testSize();
        void testLoadingLevelDb();
        void testLoadedDataCorrectness();
        void testLoadedDataNullCount();
};

#endif
