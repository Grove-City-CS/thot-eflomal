/*
thot package for statistical machine translation
Copyright (C) 2013-2017 Daniel Ortiz-Mart\'inez, Adam Harasimowicz
 
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
 * @file BasicSocketUtils.cc
 * 
 * @brief Definitions file for BasicSocketUtils.h
 */

#include "BasicSocketUtils.h"

namespace BasicSocketUtils
{
  //---------------
  int init(void)
  {
#ifdef THOT_MINGW
    WSADATA wsadata;
    if (WSAStartup(MAKEWORD(1,1), &wsadata) == SOCKET_ERROR)
    {
      fprintf(stderr,"Error initializing sockets!");
      return THOT_ERROR;
    }
    else return THOT_OK;
#else
    return THOT_OK;
#endif
  }

  //---------------
  int recvStr(int s,char *str)
  {
    int  numbytes;

    numbytes=recvInt(s);
    if(numbytes>0)
    {
      if ((numbytes=recv(s,str,numbytes,0)) == -1)
      {
            // recv() call
        std::cerr<<"recv() error!"<<std::endl;
        throw std::runtime_error("Socket error: Cannot read string");
      }
    }
    str[numbytes] = '\0';
    return numbytes;
  }

  //---------------
  int recvStlStr(int s,std::string& stlstr)
  {
    int  numbytes;
    char* str=NULL;

    numbytes=recvInt(s);
    str=(char*) mem_alloc_utils::my_realloc(str,(numbytes+1)*sizeof(char));
    if(numbytes>0)
    {
      if ((numbytes=recv(s,str,numbytes,0)) == -1)
      {
            // recv() call
        std::cerr<<"recv() error!"<<std::endl;
        free(str);
        throw std::runtime_error("Socket error: Cannot read STL string");
      }
    }
    str[numbytes] = '\0';
    stlstr=str;
    free(str);
    return numbytes;
  }

  //---------------
  int recvInt(int s)
  {
    int numbytes;
    int receivedInt;

    if ((numbytes=recv(s,(char*)&receivedInt,sizeof(int),0)) == -1)
    {
          // recv() call
      std::cerr<<"recv() error!"<<std::endl;
      throw std::runtime_error("Socket error: Cannot read integer");
    }
    else
    {
      receivedInt=ntohl(receivedInt);
    }
    return receivedInt;
  }

  //---------------
  int recvFloat(int s,float& f)
  {
    int dp;
    int sign;
    char auxstr[NETWORK_BUFF_SIZE];
    int numbytes;

    dp=recvInt(s);
    numbytes=sizeof(int);
    sign=recvInt(s);
    numbytes+=sizeof(int);
    numbytes+=recvStr(s,auxstr);
    if(dp>=0) f=(atof(auxstr)/(double)((double)pow((double)10,(double)5-dp)));
    else f=(atof(auxstr)*(double)((double)pow((double)10,(double)dp-5)));
    if(sign!=0) f=-f;
    return numbytes;
  }

  //--------------------------
  int writeInt(int fd,int i)
  {
    int ret;

    i=htonl(i);
    if((ret=write(fd,(char*) &i,sizeof(i)))==-1)
    {
      std::cerr<<"write() error"<<std::endl;
      throw std::runtime_error("Socket error: Cannot write integer");
    }
    return ret;
  }
  //--------------------------
  int writeStr(int fd,const char* s)
  {
    int numbytes;
    int ret=0;

    numbytes=strlen(s);
    ret+=writeInt(fd,numbytes);
    if(numbytes>0)
    {
     if((ret+=write(fd,(char*) s,numbytes))==-1)
     {
       std::cerr<<"write() error"<<std::endl;
       throw std::runtime_error("Socket error: Cannot write string");
     }
    }
    return ret;
  }

  //---------------
  void connect(const char *dirServ,
               unsigned int port,
               int& fileDesc)
  {
     struct hostent *he;
         /* Data structure containing information about remote host */

     struct sockaddr_in server;
         /* server address */

     if ((he=gethostbyname(dirServ))==NULL)
     {
       std::cerr<<"gethostbyname() error\n";
       throw std::runtime_error("Error while establishing connection");
     }

     if ((fileDesc=socket(AF_INET, SOCK_STREAM, 0))==-1)
     {
       std::cerr<<"socket() error\n";
       throw std::runtime_error("Error while establishing connection");
     }

     server.sin_family = AF_INET;
     server.sin_port = htons(port);
         // htons() is used to convert from hardware data representation to network representation
     server.sin_addr = *((struct in_addr *)he->h_addr);
         // he->h_addr changes the data type from``*he'' to "h_addr"
//     bzero(&(server.sin_zero),8);
     memset(&(server.sin_zero),0,8);

         // Connect to server
     if(connect(fileDesc, (struct sockaddr *)&server,sizeof(struct sockaddr))==-1)
     {
       std::cerr<<"connect() error\n";
       throw std::runtime_error("Error while establishing connection");
     }
  }

  //---------------
  void clean(void)
  {
#ifdef THOT_MINGW
    WSACleanup();
#else
#endif
  }
}
