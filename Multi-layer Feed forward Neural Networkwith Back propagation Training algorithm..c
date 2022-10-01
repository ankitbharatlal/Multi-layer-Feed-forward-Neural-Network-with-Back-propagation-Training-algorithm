//Multi-layer Feed forward Neural Network with Back propagation Training algorithm.//
#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<time.h>
#define tolerance pow(10,-3)
#define kmax pow(10,5)
int main()
{
	FILE *op1,*ip,*op2,*op3;
	ip=fopen("input.txt","r");
	op1=fopen("MSEvsITERATIONS.txt","w");
	op2=fopen("OUTPUTofNETWORK.txt","w");
	op3=fopen("ERRORinPREDICTION.txt","w");
	int p,P,L,M=3,N,i,j,k,iteration=0,P1,P2;
	float I[200][200],V[200][200],W[200][200],IH[200][200],OH[200][200],IO[200][200],OO[200][200],TO[200][200];
	float delW[200][200],delV[200][200],error,MSE,eta=0.5;
	float maxI[200],minI[200],maxTO[200],minTO[200];
	printf("Enter No of Input Patterns:");
	scanf("%d",&L);
	printf("Enter No of Output Patterns:");
	scanf("%d",&N);
	printf("Enter No of Training Patterns:");
	scanf("%d",&P1);
	printf("Enter No of Testing Patterns:");
	scanf("%d",&P2);
	P=P1+P2;

	for(p=1;p<=P;p++)
	{
		for(i=1;i<=L;i++)
		{
			fscanf(ip,"%f",&I[i][p]);
		}
	}
	printf("\n\nI[L][P] (%dX%d) matrix",L,P);
	fprintf(op2,"\n\nI[L][P] (%dX%d) matrix",L,P);
	printf("\n");
	fprintf(op2,"\n");
	for(i=1;i<=L;i++)
	{
		for(p=1;p<=P;p++)
		{
			printf("%f\t",I[i][p]);
			fprintf(op2,"%f\t",I[i][p]);
		}
		printf("\n");
		fprintf(op2,"\n");
	}
	for(i=1;i<=L;i++)
	{
		maxI[i]=-10000;minI[i]=10000;
		for(p=1;p<=P;p++)
		{
			if(I[i][p]>maxI[i])
			maxI[i]=I[i][p];
			if(I[i][p]<minI[i])
			minI[i]=I[i][p];
		}
	}
	for(p=1;p<=P;p++)
	{
		for(i=1;i<=L;i++)
		{
			I[i][p]=0.1+0.8*((I[i][p]-minI[i])/(maxI[i]-minI[i]));
		}
	}
	printf("\n\nNormaalized I[L][P] (%dX%d) matrix",L,P);
	fprintf(op2,"\n\nNormalized I[L][P] (%dX%d) matrix",L,P);
	printf("\n");
	fprintf(op2,"\n");
	for(i=1;i<=L;i++)
	{
		for(p=1;p<=P;p++)
		{
			printf("%f\t",I[i][p]);
			fprintf(op2,"%f\t",I[i][p]);
		}
		printf("\n");
		fprintf(op2,"\n");
	}	
	printf("\nTO[P][N] (%dX%d) matrix\n\n",P,N);
	fprintf(op2,"\nTO[P][N] (%dX%d) matrix\n\n",P,N);
	for(p=1;p<=P;p++)
	{
		for(k=1;k<N+1;k++)
		{
			fscanf(ip,"%f",&TO[k][p]);
			printf("TO[%d][%d]:%f\t",k,p,TO[k][p]);
			fprintf(op2,"TO[%d][%d]:%f\t",k,p,TO[k][p]);
		}
		printf("\n");
		fprintf(op2,"\n");
	}
	for(k=1;k<N+1;k++)
	{
		maxTO[k]=-1000;minTO[k]=1000;
		for(p=1;p<=P;p++)
		{
			if(TO[k][p]>maxTO[k])
			maxTO[k]=TO[k][p];
			if(TO[k][p]<minTO[k])
			minTO[k]=TO[k][p];
		}
	}
	for(p=1;p<=P;p++)
	{
		for(k=1;k<N+1;k++)
		{
			TO[k][p]=0.1+0.8*((TO[k][p]-minTO[k])/(maxTO[k]-minTO[k]));
		}
	}
	printf("\nNOrmalized TO[P][N] (%dX%d) matrix\n\n",P,N);
	fprintf(op2,"\nNormalized TO[P][N] (%dX%d) matrix\n\n",P,N);
	for(p=1;p<=P;p++)
	{
		for(k=1;k<N+1;k++)
		{
			printf("TO[%d][%d]:%f\t",k,p,TO[k][p]);
			fprintf(op2,"TO[%d][%d]:%f\t",k,p,TO[k][p]);
		}
		printf("\n");
		fprintf(op2,"\n");
	}
	
	printf("\n\nV[L+1][M] (%dX%d)\n",L+1,M);
	fprintf(op2,"\n\nV[L+1][M] (%dX%d)\n",L+1,M);
	srand(time(NULL));
	for(i=0;i<L+1;i++)
	{
		for(j=1;j<=M;j++)
		{
			V[i][j]=1.0*rand()/ RAND_MAX;
		}
	}
	
	for(i=0;i<=L;i++)
	{
		for(j=1;j<=M;j++)
		{
			printf("V[%d][%d]:%f\t",i,j,V[i][j]);
			fprintf(op2,"V[%d][%d]:%f\t",i,j,V[i][j]);
		}
		printf("\n");
		fprintf(op2,"\n");
	}
	printf("\n");
	fprintf(op2,"\n");
	for(j=0;j<M+1;j++)
	{
		for(k=1;k<=N;k++)
		{
			W[j][k]=1.0*rand()/RAND_MAX;
//			fscanf(ip,"%f",&W[j][k]);
		}
	}
	for(j=0;j<M+1;j++)
	{
		for(k=1;k<=N;k++)
		{
			printf("W[%d][%d]:%f\t",j,k,W[j][k]);
			fprintf(op2,"W[%d][%d]:%f\t",j,k,W[j][k]);
		}
		printf("\n");
		fprintf(op2,"\n");
	}
	fprintf(op1,"Iteration\tMSE\n");
	do
	{
		iteration++;
		// forward pass calculation
		for(p=1;p<=P1;p++)
		{
			IH[j][p]=0;
			for(j=1;j<M+1;j++)
			{
				for(i=1;i<L+1;i++)
				{
					IH[j][p]=IH[j][p]+(I[i][p]*V[i][j]);
				}
				IH[j][p]=IH[j][p]+(1*V[0][j]);
				OH[j][p]=1.0/(1.0+exp(-IH[j][p]));
//				printf("\nIH[%d][%d]:%f\tOH[%d][%d]:%f",j,p,IH[j][p],j,p,OH[j][p]);
//				fprintf(op1,"\nIH[%d][%d]:%f\tOH[%d][%d]:%f",j,p,IH[j][p],j,p,OH[j][p]);
				IH[j][p]=0;
			}
			
		}
//		printf("\n\n");
//		fprintf(op1,"\n\n");
		//output of output layer
		for(p=1;p<=P1;p++)
		{
			IO[k][p]=0;
			for(k=1;k<N+1;k++)
			{
				for(j=1;j<M+1;j++)
				{
					IO[k][p]+=OH[j][p]*W[j][k];
				}
				IO[k][p]=IO[k][p]+(1*W[0][k]);
				OO[k][p]=(exp(IO[k][p])-exp(-1*IO[k][p]))/(exp(IO[k][p])+exp(-1*IO[k][p]));
//				printf("\nIO[%d][%d]:%f\tOO[%d][%d]%f",k,p,IO[k][p],k,p,OO[k][p]);
//				fprintf(op1,"\nIO[%d][%d]:%f\tOO[%d][%d]%f",k,p,IO[k][p],k,p,OO[k][p]);
				IO[k][p]=0;
			}
		}
//		printf("\n");
//		fprintf(op1,"\n");
		//delWJK calculations
		for(j=0;j<=M;j++)
		{
			for(k=1;k<=N;k++)
			{
				delW[j][k]=0;
				for(p=1;p<=P1;p++)
				{
					OH[0][p]=1.0;
					delW[j][k]=delW[j][k]+((eta/P)*(TO[k][p]-OO[k][p])*(1-(OO[k][p]*OO[k][p]))*OH[j][p]);
				}
				printf("delW[%d][%d]=%f\t",j,k,delW[j][k]);
//				fprintf(op1,"delW[%d][%d]=%f\t",j,k,delW[j][k]);
			}
			printf("\n");
//			fprintf(op1,"\n");
		}
		printf("\n");
//		fprintf(op1,"\n");
		//delVIJ calculations
		for(i=0;i<=L;i++)
		{
			for(j=1;j<=M;j++)
			{
				delV[i][j]=0;
				for(p=1;p<=P1;p++)
				{
					for(k=1;k<=N;k++)
					{
						I[0][p]=1.0;
						delV[i][j]=delV[i][j]+((eta/(P*N))*((TO[k][p]-OO[k][p])*(1-(OO[k][p]*OO[k][p]))*W[j][k]*OH[j][p]*(1-OH[j][p])*I[i][p]));
					}
				}
				printf("delV[%d][%d]=%f\t",i,j,delV[i][j]);
//				fprintf(op1,"delV[%d][%d]=%f\t",i,j,delV[i][j]);
			}
			printf("\n");
//			fprintf(op1,"\n");
		}
		//error calculations
		MSE=0;
		for(p=1;p<=P1;p++)
		{
			for(k=1;k<=N;k++)
			{
				error=pow((TO[k][p]-OO[k][p]),2)/2;
				MSE=MSE+error;
			}
		}
		MSE=MSE/P1;
		printf("iteration:%d\tMSE:%f\n",iteration,MSE);
		fprintf(op1,"%d\t%f\n",iteration,MSE);
		
		//updating Vij values
		for(i=0;i<=L;i++)
		{
			for(j=1;j<=M;j++)
			{
				V[i][j]=V[i][j]+delV[i][j];
				printf("V[%d][%d]:%f\t",i,j,V[i][j]);
//				fprintf(op1,"V[%d][%d]:%f\t",i,j,V[i][j]);
			}
			printf("\n");
//			fprintf(op1,"\n");
		}
		printf("\n");
//		fprintf(op1,"\n");
		//updating Wjk values
		for(j=0;j<=M;j++)
		{
			for(k=1;k<=N;k++)
			{
				W[j][k]=W[j][k]+delW[j][k];
				printf("W[%d][%d]:%f\t",j,k,W[j][k]);
//				fprintf(op1,"W[%d][%d]:%f\t",j,k,W[j][k]);
			}
			printf("\n");
//			fprintf(op1,"\n");
		}
	}while(MSE>tolerance && iteration<kmax);
	fprintf(op2,"Optimum Connection Weights V,W:\n");
	for(i=0;i<=L;i++)
		{
			for(j=1;j<=M;j++)
			{
				fprintf(op2,"V[%d][%d]:%f\t",i,j,V[i][j]);
			}
			fprintf(op2,"\n");
		}
		fprintf(op2,"\n");
		//updating Wjk values
		for(j=0;j<=M;j++)
		{
			for(k=1;k<=N;k++)
			{
				fprintf(op2,"W[%d][%d]:%f\t",j,k,W[j][k]);
			}
			fprintf(op2,"\n");
		}
		//TESTING
	// forward pass calculation
		for(p=P1+1;p<=P;p++)
		{
			IH[j][p]=0;
			for(j=1;j<M+1;j++)
			{
				for(i=1;i<L+1;i++)
				{
					IH[j][p]=IH[j][p]+(I[i][p]*V[i][j]);
				}
				IH[j][p]=IH[j][p]+(1*V[0][j]);
				OH[j][p]=1/(1+exp(-IH[j][p]));
				printf("\nIH[%d][%d]:%f\tOH[%d][%d]:%f",j,p,IH[j][p],j,p,OH[j][p]);
				fprintf(op2,"\nIH[%d][%d]:%f\tOH[%d][%d]:%f",j,p,IH[j][p],j,p,OH[j][p]);
				IH[j][p]=0;
			}
			
		}
		printf("\n");
		fprintf(op2,"\n");
		fprintf(op3,"ERROR in PREDICTION (%d Patterns):",P2);
		//output of output layer
		error=0;
		for(p=P1+1;p<=P;p++)
		{
			IO[k][p]=0;
			for(k=1;k<N+1;k++)
			{
				for(j=1;j<=M+1;j++)
				{
					IO[k][p]+=OH[j][p]*W[j][k];
				}
				IO[k][p]+=1.0*W[0][k];
				OO[k][p]=(exp(IO[k][p])-exp(-1*IO[k][p]))/(exp(IO[k][p])+exp(-1*IO[k][p]));
				error+=fabs(OO[k][p]-TO[k][p]);
				printf("\nIO[%d][%d]:%f\tOO[%d][%d]:%f\tTO[%d][%d]:%f\tError:%f",k,p,IO[k][p],k,p,OO[k][p],k,p,TO[k][p],fabs(OO[k][p]-TO[k][p]));
				fprintf(op2,"\nIO[%d][%d]:%f\tOO[%d][%d]:%f\tTO[%d][%d]:%f",k,p,IO[k][p],k,p,OO[k][p],k,p,TO[k][p]);
				fprintf(op3,"\n%f",fabs(OO[k][p]-TO[k][p]));
				IO[k][p]=0;
			}
		}
		fprintf(op3,"\nAverage Error:%f",error/P2);
		MSE=0;
		for(p=P1+1;p<=P;p++)
		{
			for(k=1;k<=N;k++)
			{
				error=pow((TO[k][p]-OO[k][p]),2)/2;
				MSE=MSE+error;
			}
		}
		MSE=MSE/P2;
		printf("\nEnergy of error of Predicted Output:%f",MSE);
		fprintf(op3,"\nEnergy of error of Predicted Output:%f",MSE);
	fclose(ip);
	fclose(op1);
	fclose(op2);
	fclose(op3);
	return 0;
}
