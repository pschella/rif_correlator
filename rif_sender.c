#include <sys/socket.h>
#include <netinet/in.h>
#include <stdio.h>
#include <strings.h>
#include <stdlib.h>
#include <arpa/inet.h>
#include <stdlib.h>

#define PACKET_SIZE 1024

int main(int argc, char* argv[])
{
	int i;
	FILE *fp;
	char buffer[PACKET_SIZE];
	int sockfd;
	struct sockaddr_in servaddr, cliaddr;

	if (argc != 3)
	{
		printf("usage: rif_sender <IP address> <file>\n");
		exit(1);
	}

	/* Open input file */
	fp = fopen(argv[2], "rb");
	if (fp == NULL) {
		fprintf(stderr, "Error: could not open input file!\n");
		return 1;
	}

	sockfd = socket(AF_INET, SOCK_DGRAM, 0);

	bzero(&servaddr, sizeof(servaddr));
	servaddr.sin_family = AF_INET;
	servaddr.sin_addr.s_addr = inet_addr(argv[1]);
	servaddr.sin_port = htons(32000);

	i = 0;
	while (fread(buffer, sizeof(char), PACKET_SIZE, fp) == PACKET_SIZE*sizeof(char)) {
		printf("%d\n", i);

		sendto(sockfd, buffer, PACKET_SIZE*sizeof(char), 0, (struct sockaddr *)&servaddr, sizeof(servaddr));

		i++;
	}

	/* Close file */
	fclose(fp);
	return 0;
}

