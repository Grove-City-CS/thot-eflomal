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

#include "sw_models/IncrLexTable.h"

#include "_incrLexTableTest.h"

#include <gtest/gtest.h>

template <> _incrLexTable* CreateIncrLexTable<IncrLexTable>()
{
  return new IncrLexTable;
}

INSTANTIATE_TYPED_TEST_SUITE_P(IncrLexTableTest, _incrLexTableTest, IncrLexTable);
