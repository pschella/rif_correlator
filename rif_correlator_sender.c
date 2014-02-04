#include <sys/socket.h>
#include <netinet/in.h>
#include <stdio.h>
#include <strings.h>
#include <stdlib.h>
#include <arpa/inet.h>

/* UDP port */
#define UDP_PORT_NUMBER 32000

/* Number of samples to average per channel per time bin */
#define N 19999744
/*20e6*/

int main(int argc, char* argv[])
{
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
  
  buffer = (char*)malloc(2*N*sizeof(char));

  /* Open input file */
  fp = fopen(argv[2], "rb");
  if (fp == NULL) {
    fprintf(stderr, "Error: could not open input file!\n");
    return 1;
  }

  while (fread(buffer, sizeof(char), 2*N, fp) == 2*N*sizeof(char)) {
    sockfd=socket(AF_INET, SOCK_DGRAM, 0);
    
    bzero(&servaddr, sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    servaddr.sin_addr.s_addr=inet_addr(argv[1]);
    servaddr.sin_port=htons(UDP_PORT_NUMBER);
    
    sendto(sockfd, buffer, 2*N*sizeof(char), 0, (struct sockaddr *)&servaddr, sizeof(servaddr));
  }
}

