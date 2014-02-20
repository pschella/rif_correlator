/*
Copyright (C) 2014 Pim Schellart <P.Schellart@astro.ru.nl>

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
*/

#include <time.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <stdio.h>
#include <strings.h>
#include <stdlib.h>
#include <arpa/inet.h>

/* UDP port */
#define UDP_PORT_NUMBER 32000

#define PACKET_SIZE 1024

int main(int argc, char* argv[])
{
  FILE *fp;
  int sockfd,n;
  struct sockaddr_in servaddr,cliaddr;
  char sendline[1000];
  char recvline[1000];
  char *buffer;
  int i;
  
  if (argc != 3)
  {
    printf("usage: rif_correlator_sender <IP address> <file>\n");
    exit(1);
  }
  
  buffer = (char*)malloc(PACKET_SIZE*sizeof(char));

  /* Open input file */
  fp = fopen(argv[2], "rb");
  fseek(fp, 0, 0);
  if (fp == NULL) {
    fprintf(stderr, "Error: could not open input file!\n");
    return 1;
  }

  i=0;
  printf("test %d\n", fread(buffer, sizeof(char), PACKET_SIZE, fp));

  while (fread(buffer, sizeof(char), PACKET_SIZE, fp) == PACKET_SIZE*sizeof(char)) {
    printf("sending block %d\n", i++);
    sockfd=socket(AF_INET, SOCK_DGRAM, 0);
    
    bzero(&servaddr, sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    servaddr.sin_addr.s_addr=inet_addr(argv[1]);
    servaddr.sin_port=htons(UDP_PORT_NUMBER);
    
    sendto(sockfd, buffer, PACKET_SIZE*sizeof(char), 0, (struct sockaddr *)&servaddr, sizeof(servaddr));
    usleep(1000000);
  }

  fclose(fp);
}

