/* Sample UDP server */

#include <sys/socket.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>

#define N 1024

int main(int argc, char**argv)
{
   int sockfd,n;
   struct sockaddr_in servaddr,cliaddr;
   socklen_t len;
   char *buffer;
   int i;

   buffer = (char*)malloc(2*N*sizeof(char));

   sockfd=socket(AF_INET,SOCK_DGRAM,0);

   bzero(&servaddr,sizeof(servaddr));
   servaddr.sin_family = AF_INET;
   servaddr.sin_addr.s_addr=htonl(INADDR_ANY);
   servaddr.sin_port=htons(32000);
   bind(sockfd,(struct sockaddr *)&servaddr,sizeof(servaddr));

   for (;;)
   {
      len = sizeof(cliaddr);

      n = recvfrom(sockfd, buffer, 2*N*sizeof(char), 0, (struct sockaddr *)&cliaddr, &len);

      for (i=0; i<2*N; i++) {
        printf("%d\n", (int)buffer[i]);
      }
   }
}

