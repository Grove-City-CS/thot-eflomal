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
 * @file WordAligMatrix.cc
 * 
 * @brief Definitions file for WordAligMatrix.h
 */

#include "WordAligMatrix.h"

//-------------------------
WordAligMatrix::WordAligMatrix(void)
{
  I=0;
  J=0;
}
//-------------------------
WordAligMatrix::WordAligMatrix(unsigned int I_dims,unsigned int J_dims,int val)
{
  I=0;
  J=0;
  init(I_dims,J_dims,val);
}

//-------------------------
WordAligMatrix::WordAligMatrix(const WordAligMatrix  &waMatrix)
{
  unsigned int i,j;
 
  I=0;
  J=0;
  init(waMatrix.I,waMatrix.J);
	
  for(i=0;i<I;++i)
    for(j=0;j<J;++j)
      matrix[i][j]=waMatrix.matrix[i][j];
}

//-------------------------
unsigned int WordAligMatrix::get_I(void)const
{
  return I;
}

//-------------------------
unsigned int WordAligMatrix::get_J(void)const
{
  return J;
}

//-------------------------
bool WordAligMatrix::empty(void)const
{
  return I==0 || J==0;
}

//-------------------------
int WordAligMatrix::getValue(unsigned int i,
                             unsigned int j)const
{
  return matrix[i][j];
}

//-------------------------
void WordAligMatrix::init(unsigned int I_dims,unsigned int J_dims,int val)
{
  unsigned int i;

  if(I!=I_dims || J!=J_dims)
  {
    clear();
    I=I_dims;
    J=J_dims;
	 
    matrix=(int**) calloc(I,sizeof(int*));
    for(i=0;i<I;++i)
      matrix[i]=(int*) calloc(J,sizeof(int));
 
    setValues(val);	
  }
  else setValues(val);
}

//-------------------------
void WordAligMatrix::putAligVec(std::vector<PositionIndex> aligVec)
{
  unsigned int j;
	
  if(aligVec.size()==J)
  {
    for(j=0;j<aligVec.size();++j)
    {
      if(aligVec[j]>0) matrix[aligVec[j]-1][j]=1;
    }
  }
}

//-------------------------
void WordAligMatrix::getAligVec(std::vector<PositionIndex>& aligVec)const
{
  aligVec.clear();
  for(unsigned int j=0;j<J;++j)
  {
    aligVec.push_back(0);
    for(unsigned int i=0;i<I;++i)
    {
      if(matrix[i][j]>0)
      {
        aligVec[j]=i+1;
        break;
      }
      if(matrix[i][j]<0)
      {
        aligVec[j]=UNKNOWN_POSITION;
        break;
      }
    }
  }
}

//-------------------------
void WordAligMatrix::reset(void)
{
  setValues(0);
}

//-------------------------
void WordAligMatrix::set(void)
{
  setValues(1);
}

//-------------------------
void WordAligMatrix::setValues(int val)
{
  unsigned int i,j;
	
  for(i=0;i<I;++i)
    for(j=0;j<J;++j)
      matrix[i][j]=val;
}

//-------------------------
void WordAligMatrix::set(unsigned int i,unsigned int j)
{
  if(i<I && j<J) matrix[i][j]=1;	
}



//-------------------------
void WordAligMatrix::setValue(unsigned int i,unsigned int j,int val)
{
  if(i<I && j<J) matrix[i][j]=val;		
}

//-------------------------
void WordAligMatrix::transpose(void)
{
  WordAligMatrix wam;
  unsigned int i,j;
	
  wam.init(J,I);
 
  for(i=0;i<I;++i)	
	for(j=0;j<J;++j)
	{
      wam.matrix[j][i]=matrix[i][j];	
	}
  *this=wam;
}

//-------------------------
WordAligMatrix& WordAligMatrix::operator= (const WordAligMatrix &waMatrix)
{
  unsigned int i,j;

  init(waMatrix.I,waMatrix.J);
  for(i=0;i<I;++i)
    for(j=0;j<J;++j)
      matrix[i][j]=waMatrix.matrix[i][j];
	 
  return *this;	 
}

//-------------------------
bool WordAligMatrix::operator== (const WordAligMatrix &waMatrix)
{
  unsigned int i,j;
	
  if(waMatrix.I!=I || waMatrix.J!=J) return 0;
  for(i=0;i<I;++i)
    for(j=0;j<J;++j)
    {
      if(waMatrix.matrix[i][j]!=matrix[i][j])
        return 0;
    }
 
  return 1;
}

//-------------------------
WordAligMatrix& WordAligMatrix::flip(void)
{
  unsigned int i,j;
  
  for(i=0;i<I;++i)
    for(j=0;j<J;++j)
    {
      if(matrix[i][j]==0)
        matrix[i][j]=1;
      else
        matrix[i][j]=0; 
    }
  return *this; 	
}

//-------------------------
WordAligMatrix& WordAligMatrix::operator&= (const WordAligMatrix &waMatrix)
{
  unsigned int i,j;

	
  if(I==waMatrix.I && J==waMatrix.J)
  {
    for(i=0;i<I;++i)
      for(j=0;j<J;++j)
      {
        if(!(matrix[i][j]>0 && waMatrix.matrix[i][j]>0))	 
		      matrix[i][j]=0; 
      }
  }
  return *this; 
}

//-------------------------
WordAligMatrix& WordAligMatrix::operator|= (const WordAligMatrix &waMatrix)
{
  unsigned int i,j;
	
  if(I==waMatrix.I && J==waMatrix.J)
  {
    for(i=0;i<I;++i)
      for(j=0;j<J;++j)
        if(matrix[i][j]>0 || waMatrix.matrix[i][j]>0)	 
		      matrix[i][j]=1;
  }	
  return *this;
}

//-------------------------
WordAligMatrix& WordAligMatrix::operator^= (const WordAligMatrix &waMatrix)
{
  unsigned int i,j;
	
  if(I==waMatrix.I && J==waMatrix.J)
  {
    for(i=0;i<I;++i)
      for(j=0;j<J;++j)
      {
        if((matrix[i][j]>0 && waMatrix.matrix[i][j]==0) || (matrix[i][j]==0 && waMatrix.matrix[i][j]>0))	 
		      matrix[i][j]=1;
        else matrix[i][j]=0;
      }
  }	
  return *this;
}
//-------------------------
WordAligMatrix& WordAligMatrix::operator+= (const WordAligMatrix &waMatrix)
{
  unsigned int i,j;
	
  if(I==waMatrix.I && J==waMatrix.J)
  {
    for(i=0;i<I;++i)
      for(j=0;j<J;++j)
        matrix[i][j]=matrix[i][j] + waMatrix.matrix[i][j];
  }	
  return *this;
}

//-------------------------
WordAligMatrix& WordAligMatrix::operator-= (const WordAligMatrix &waMatrix)
{
  unsigned int i,j;
	
  if(I==waMatrix.I && J==waMatrix.J)
  {
    for(i=0;i<I;++i)
      for(j=0;j<J;++j)
        if(matrix[i][j]>0 && waMatrix.matrix[i][j]==0)	 
		  matrix[i][j]=1;
  }	
  return *this;
}

//-------------------------
WordAligMatrix& WordAligMatrix::symmetr1 (const WordAligMatrix &waMatrix)
{
  unsigned int i,j;
  WordAligMatrix	aux,prev;
		
  if(I==waMatrix.I && J==waMatrix.J)
  {
    aux=*this;
    *this&=waMatrix;	 
    prev.reset();
	 
    while(!(*this==prev))
    {
      prev=*this;
      for(i=0;i<I;++i)
        for(j=0;j<J;++j)
        {
          if((waMatrix.matrix[i][j]>0 || aux.matrix[i][j]>0) && this->matrix[i][j]==0)	 
          {
            if(jAligned(j)==0 && iAligned(i)==0) this->matrix[i][j]=1;
            else
            {
              if(this->ijInNeighbourhood(i,j)) this->matrix[i][j]=1;
            }
          }
        }
    }
  }	
  return *this;	
}

//-------------------------
WordAligMatrix& WordAligMatrix::symmetr2(const WordAligMatrix &waMatrix)
{
  unsigned int i,j;
  WordAligMatrix aux;
  WordAligMatrix prev;
		
  if(I==waMatrix.I && J==waMatrix.J)
  {
    aux=*this;
    *this&=waMatrix;	 
    prev.reset();
	 
    while(!(*this==prev))
    {
      prev=*this;
      for(i=0;i<I;++i)
        for(j=0;j<J;++j)
        {
          if((waMatrix.matrix[i][j]>0 || aux.matrix[i][j]>0) && this->matrix[i][j]==0)	 
          {
            if(jAligned(j)==0 && iAligned(i)==0) this->matrix[i][j]=1;
            else
            {
              if(this->ijInNeighbourhood(i,j)) 
                if(!(this->ijHasHorizNeighbours(i,j) && this->ijHasVertNeighbours(i,j))) this->matrix[i][j]=1;
            }
          }
        }
    }
  }	
  return *this;		
}

//-------------------------
WordAligMatrix& WordAligMatrix::growDiagFinal(const WordAligMatrix &waMatrix)
{

      // Check that the matrices can be operated
  if(I!=waMatrix.I || J!=waMatrix.J)
  {
    return *this;
  }
  else
  {
        // Matrices can be operated

        // Obtain copy of source matrix
    WordAligMatrix sourceMatAux=*this;

        // Obtain intersection between source and target
    *this&=waMatrix;

        // Obtain union between source and target
    WordAligMatrix joinMat=sourceMatAux;
    joinMat|=waMatrix;
    
        // Grow-diag
    bool end=false;
    WordAligMatrix prevMat=*this;

        // Iterate until no new points added
    while(!end)
    {
      for(unsigned int i=0;i<I;++i)
      {
        for(unsigned int j=0;j<J;++j)
        {
              // Check if (i,j) is aligned
          if(getValue(i,j)>0)
          {
                // Explore neighbourhood
            std::vector<std::pair<unsigned int,unsigned int> > neighboursVec=obtainAdjacentCells(i,j);
            for(unsigned int k=0;k<neighboursVec.size();++k)
            {
              unsigned int ip=neighboursVec[k].first;
              unsigned int jp=neighboursVec[k].second;
              if((iAligned(ip)==0 || jAligned(jp)==0) && joinMat.getValue(ip,jp)>0)
              {
                set(ip,jp);
              }
            }
          }
        }
      }
      if(prevMat==*this)
        end=true;
      else
        prevMat=*this;
    }

        // Final source
    for(unsigned int i=0;i<I;++i)
    {
      for(unsigned int j=0;j<J;++j)
      {
        if((iAligned(i)==0 || jAligned(j)==0) && sourceMatAux.getValue(i,j)>0)
        {
          set(i,j);
        }
      }
    }

        // Final target
    for(unsigned int i=0;i<I;++i)
    {
      for(unsigned int j=0;j<J;++j)
      {
        if((iAligned(i)==0 || jAligned(j)==0) && waMatrix.getValue(i,j)>0)
        {
          set(i,j);
        }
      }
    }
    
        // Return result
    return *this;
  }
}

//-------------------------
std::vector<std::pair<unsigned int,unsigned int> >
WordAligMatrix::obtainAdjacentCells(unsigned int i,
                                    unsigned int j)
{
      // Initialize variables
  std::vector<std::pair<unsigned int,unsigned int> > puintVec;
  std::pair<unsigned int,unsigned int> puint;

      // Add neighbour points
  for(int delta_i=-1;delta_i<=1;++delta_i)
  {
    for(int delta_j=-1;delta_j<=1;++delta_j)
    {
      if(delta_i!=0 || delta_j!=0)
      {
        int ip=i+delta_i;
        int jp=j+delta_j;
        if(ip<(int) I && jp<(int) J && ip>=0 && jp>=0)
        {
          puintVec.push_back(std::make_pair((unsigned int) ip,(unsigned int) jp));
        }
      }
    }
  }
     // Return result
  return puintVec;
}

//-------------------------
bool WordAligMatrix::ijInNeighbourhood(unsigned int i,unsigned int j)
{
 if(i>0) if(matrix[i-1][j]>0) return 1;
 if(j>0) if(matrix[i][j-1]>0) return 1;
 if(i<I-1) if(matrix[i+1][j]>0) return 1;
 if(j<J-1) if(matrix[i][j+1]>0) return 1;
	 
 return 0; 
}

//-------------------------
bool WordAligMatrix::ijHasHorizNeighbours(unsigned int i,unsigned int j)
{
 if(j>0) if(matrix[i][j-1]>0) return 1;
 if(j<J-1) if(matrix[i][j+1]>0) return 1;
	 
 return 0; 
}

//-------------------------
bool WordAligMatrix::ijHasVertNeighbours(unsigned int i,unsigned int j)
{
 if(i>0) if(matrix[i-1][j]>0) return 1;
 if(i<I-1) if(matrix[i+1][j]>0) return 1;	 
 
 return 0; 
}

//-------------------------
int WordAligMatrix::jAligned(unsigned int j)const
{
  unsigned int i;

  for(i=0;i<I;++i)
  {
    if(matrix[i][j]>0)
      return 1;
    if(matrix[i][j]<0)
      return -1;
  }
		 
  return 0;	 	
}

//-------------------------
int WordAligMatrix::iAligned(unsigned int i)const
{
  unsigned int j;
	
  for(j=0;j<J;++j)
  {
    if(matrix[i][j]>0)
      return 1;
    if(matrix[i][j]<0)
      return -1;
  }
		 
  return 0;	 
}

//-------------------------
void WordAligMatrix::clear(void)
{
   unsigned int i;
	      
   if(I>0)
   {
     for(i=0;i<I;++i)
       free(matrix[i]);
     free(matrix);	
   }
   I=0; J=0;
}

//-------------------------
std::ostream& operator << (std::ostream &outS,const WordAligMatrix &waMatrix)
{
  unsigned int j;
  int i;
	
  for(i=(int)waMatrix.I-1;i>=0;--i)	
  {
    for(j=0;j<waMatrix.J;++j)
    {
      if(waMatrix.matrix[i][j]<0)
        outS<<"U";
      else
        outS<<waMatrix.matrix[i][j];
      outS<<" ";
    }
    outS<<std::endl;	 
  }
  return outS;	
}
//-------------------------
void WordAligMatrix::print(FILE* f)
{
  unsigned int j;
  int i;
	
  for(i=(int)this->I-1;i>=0;--i)	
  {
    for(j=0;j<this->J;++j)	
      fprintf(f,"%d ",this->matrix[i][j]);
    fprintf(f,"\n");	 
  }	
}

//-------------------------
void WordAligMatrix::wordAligAsVectors(std::vector<std::pair<unsigned int,unsigned int> >& sourceSegm,
                                       std::vector<unsigned int>& targetCuts)
{
  std::pair<unsigned int,unsigned int> prevIntPair,intPair;
  unsigned int i,j;

  targetCuts.clear();
	
  prevIntPair.first=0;
  prevIntPair.second=0;	
  for(i=0;i<I;++i)	
  {
    intPair.first=0;
    intPair.second=0;
    for(j=0;j<J;++j)
    {
	  if(matrix[i][j]>0 && intPair.first==0) intPair.first=j+1;
	  if(matrix[i][j]==0 && intPair.first!=0 && intPair.second==0) intPair.second=j;
    }
    if(intPair.second==0) intPair.second=j;
    if(intPair!=prevIntPair)
    {
      sourceSegm.push_back(intPair);
      if(i!=0) targetCuts.push_back(i);		 
	    prevIntPair=intPair;
    }
  }
  targetCuts.push_back(i);	
}
//-------------------------
WordAligMatrix::~WordAligMatrix()
{
  clear();
}
