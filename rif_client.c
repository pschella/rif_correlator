/* Sample UDP client */

#include <sys/socket.h>
#include <netinet/in.h>
#include <stdio.h>
#include <strings.h>
#include <stdlib.h>
#include <arpa/inet.h>

#define N 1024

void readdata(char* filename, char* data, long start, long n)
{
  FILE *fp;

  /* Open file */
  fp = fopen(filename, "rb");
  if (fp == NULL)
  {
    printf("Error: could not open file!\n");
    return;
  }

  /* Skip to correct position */
  fseek(fp, start * sizeof(char), SEEK_SET);

  /* Read data */
  fread(data, sizeof(char), n, fp);

  /* Close file */
  fclose(fp);
}

long nsamples(char* filename)
{
  FILE *fp;
  long n;

  /* Open file */
  fp = fopen(filename, "rb");

  /* Find end of file */
  fseek(fp, 0, SEEK_END);
  
  /* Get size */
  n = ftell(fp) / sizeof(char);
  
  /* Close file */
  fclose(fp);
  
  return n;
}

int main(int argc, char**argv)
{
   int sockfd;
   struct sockaddr_in servaddr,cliaddr;
   char sendline[1000];
   char recvline[1000];
   char *buffer;
   int i;
   long n;

   if (argc != 3)
   {
      printf("usage:  udpcli <IP address> <file>\n");
      exit(1);
   }

   buffer = (char*)malloc(2*N*sizeof(char));

   n = nsamples(argv[2]);

   sockfd=socket(AF_INET,SOCK_DGRAM,0);

   bzero(&servaddr,sizeof(servaddr));
   servaddr.sin_family = AF_INET;
   servaddr.sin_addr.s_addr=inet_addr(argv[1]);
   servaddr.sin_port=htons(32000);

   for (i=0; i<n/(2*N); i++) {
       readdata(argv[2], buffer, 2*N*i, 2*N);

       sendto(sockfd, buffer, 2*N*sizeof(char), 0, (struct sockaddr *)&servaddr, sizeof(servaddr));
   }
}

