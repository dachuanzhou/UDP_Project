#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <math.h>
#include <time.h>
#include <cstring>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

#define PI 3.14159265358979323846
#define N 2048
#define M 256 //接收阵元到发射阵元的最大距离（阵元个数），所以接收孔径为2*M+1
#define ele_no 2048
#define od 64
#define Nsample 3750

//const int sample_point_num=5000;

__device__ float dev_ele_position_center[ele_no];
__device__ float dev_ele_position_height[ele_no]; //写到纹理内存里面
float ele_position_center[ele_no] = {0};
float ele_position_height[ele_no] = {0};

float rfocus = 112.0 / 1000;
float first_one = 2 * PI * (45 - 43.4695) / 360;        //第一个阵元角度
float ele_angle = (2 * PI * 43.4695 / (256 - 1)) / 360; //阵元间隔角度

short uct_data_board0[Nsample * ele_no] = {0}; //写到页锁定主机内存
short before_convert_data;
//float trans_sdata[sample_point_num*ele_no]={0};

float allimage_data[N * N] = {0};
int allpoint_count[N * N] = {0};

void get_ele_position();
void write_txtfile();
//cudaError_t calcWithCuda(int i,float *dev_sumdata,int *dev_sumpoint,float *dev_filterdata);

__device__ float db[od]; //filter parameter
//__device__ float da[31];
float b[od];
//float a[31];
//int i;
void get_code();
cudaError_t precalcWithCuda(short *uct_data_board0, int i, float *dev_sumdata, int *dev_sumpoint);

//short sample_data[sample_point_num][ele_no]={0};

//double v[3344][256]={0};;
//double u_test[1024]={0};
__global__ void kernel3(float *filterdata, short *data1_1024_output)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int id = blockDim.x * bid + tid;
    short data[Nsample];
    float fdata[Nsample];

    if (id < ele_no)
    {
        //da[0]=0.0;
        memset(fdata, 0, Nsample * sizeof(float));
        for (int i = 0; i < Nsample; i++)
        {
            data[i] = data1_1024_output[i * ele_no + id];
            for (int j = 0; i >= j && j < od; j++)
            {
                //fdata[i] += (db[j]*data[i-j]-da[j]*fdata[i-j]);
                fdata[i] += (db[j] * data[i - j]);
            }
        }
        //da[0]=1.0;

        for (int i = 0; i < Nsample; i++)
        {
            filterdata[i * ele_no + id] = fdata[i];
        }
    }
}
__global__ void kernel(int i, float *image_data, int *point_count, float *trans_sdata)
{
    int c = 1520;
    float fs = 25e6;
    float image_width = 200.0 / 1000;
    float image_length = 200.0 / 1000;
    float data_diameter = 220.0 / 1000;
    int point_length = data_diameter / c * fs + 0.5;

    const int no_lines = N;
    const int no_point = N;

    float d_x = image_width / (no_lines - 1);
    float d_z = image_length / (no_point - 1);

    int middot = -160; //发射前1us开始接收，也就是约为12.5个点之后发射,数据显示约16个点
                       //const int ele_no=1024;

    int k_line = blockIdx.y;                       //线
    int nn = blockIdx.x;                           //点
                                                   //blockIdx.x+blockIdx.y * gridDimx.x
    int y = threadIdx.x;                           //接收阵元
    int j = i - 1 - M + y;                         //接收阵元
    int bid = blockIdx.x + blockIdx.y * gridDim.x; //线程块的索引
    int tid = blockDim.x * bid + threadIdx.x;      //线程的索引

    __shared__ float cache_image[2 * M];
    __shared__ int cache_point[2 * M];
    /*__shared__ float cache_image2[512];*/
    int cacheIndex = threadIdx.x;

    if (k_line < N && nn < N && y < 2 * M)
    {
        float u = 0;
        int point_count_1 = 0;
        float z1 = -image_length / 2 + d_z * nn;
        float x = -image_length / 2 + d_x * k_line;
        float xg = 0.0014;

        // for(int jj=1;jj<=ele_no/M;jj++)
        // {
        //  int j=y*ele_no/M+jj;
        j = (j + ele_no) % ele_no;

        //int num= (sqrt(pow((dev_ele_position_center[i-1]-x),2) + pow((z1-dev_ele_position_height[i-1]),2))+sqrt(pow((dev_ele_position_center[j-1]-x),2) + pow((z1-dev_ele_position_height[j-1]),2)))/c*fs+0.5;
        float disi = sqrt((dev_ele_position_center[i - 1] - x) * (dev_ele_position_center[i - 1] - x) + (z1 - dev_ele_position_height[i - 1]) * (z1 - dev_ele_position_height[i - 1]));
        float disj = sqrt((dev_ele_position_center[j] - x) * (dev_ele_position_center[j] - x) + (z1 - dev_ele_position_height[j]) * (z1 - dev_ele_position_height[j]));
        float ilength = 112.0 / 1000;
        float imagelength = sqrt(x * x + z1 * z1);
        float angle = acos((ilength * ilength + disi * disi - imagelength * imagelength) / 2 / ilength / disi);
        if ((disi >= 0.1 * 2 / 3 && (abs(i - j - 1) < 256 || abs(i - j - 1) > 1024 - 256)) || (disi >= 0.1 * 1 / 3 && (abs(i - j - 1) < 200 || abs(i - j - 1) > 1024 - 200)) || (disi >= 0 && (abs(i - j - 1) < 100 || abs(i - j - 1) > 1024 - 100)))
        {
            int num = (disi + disj) / c * fs + 0.5;
            //if (((num+middot)>200)&&(num<=point_length)&&(angle<PI/9))
            // if (((num+middot+(od-1-1)/2)>100)&&((num+middot+(od-1-1)/2)<=point_length))
            if (((num + middot + (od - 1 - 1) / 2) > 100) && ((num + middot + (od - 1 - 1) / 2) <= point_length) && (angle < PI / 9))
            {
                //u= trans_sdata[(num+middot)*ele_no+(j-1)];
                // u= trans_sdata[(num+middot)*ele_no+j]*exp(xg*(num-1));
                // u= trans_sdata[(num+middot)*ele_no+j];
                // u= trans_sdata[(num+middot+(od-1-1)/2)*ele_no+j];
                u = trans_sdata[(num + middot + (od - 1 - 1) / 2) * ele_no + j] * exp(xg * (num - 1));
                // u= trans_sdata[(num+middot+(127-1)/2)*ele_no+j]*exp(xg*(num-1));
                //u= trans_sdata[(num+middot+(127-1)/2)*ele_no+j]*exp(xg*(num-1));
                // cout<<u<<endl;
                point_count_1 = 1;
            }
        }

        //}
        cache_image[cacheIndex] = u;
        cache_point[cacheIndex] = point_count_1;

        __syncthreads();

        /*int jj=1;
        while((cacheIndex + jj)< blockDim.x){
        cache_image2[cacheIndex]+=cache_image[cacheIndex]*cache_image[cacheIndex + jj];
         __syncthreads();
                 jj= jj+1;
        }*/

        int i = blockDim.x / 2;
        while (i != 0)
        {
            if (cacheIndex < i)
            {
                cache_image[cacheIndex] += cache_image[cacheIndex + i];
                cache_point[cacheIndex] += cache_point[cacheIndex + i];
            }
            __syncthreads();
            i /= 2;
        }

        if (cacheIndex == 0)
        {
            image_data[bid] = cache_image[0];
            point_count[bid] = cache_point[0];
        }
    }
}

__global__ void add(float *sumdata, int *sumpoint, float *imagedata, int *point_count)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < N * N)
    {
        sumdata[tid] += imagedata[tid];
        sumpoint[tid] += point_count[tid];
        tid += blockDim.x * gridDim.x;
    }
}

int main()
{
    time_t start, over;
    start = time(NULL);
    //float *db;

    cudaError_t cudaStatus;

    ifstream inRco;
    std::string s1 = "/media/shine/SHINE/b_64.bin";
    const char *filenameRco;
    filenameRco = s1.c_str();
    inRco.open(filenameRco, ios_base::in | ios::binary);
    if (!inRco.is_open())
    {
        cout << " the file b open fail" << endl;
        return 1;
    }
    for (int ii = 0; ii < od; ii++)
    {
        inRco.read((char *)&b[ii], sizeof(float));
    }

    inRco.close();

    //cout<<"filterdata:"<<b[0]<<endl;

    cudaStatus = cudaMemcpyToSymbol(db, b, sizeof(float) * od);

    if (cudaStatus != cudaSuccess)
    {
        cout << "center Fail to cudaMemcpyToSymbol on GPU" << endl;
        return;
    }
    //cudaStatus=cudaMemcpyToSymbol(da,a,sizeof(float)*31);

    //if (cudaStatus != cudaSuccess) {
    //  cout<<"center Fail to cudaMemcpyToSymbol on GPU"<<endl;
    // return ;
    // }

    ifstream inR;
    std::string s2 = "/media/shine/SHINE/UCTDat_2048T2048R_25M.bin";
    const char *filenameR;
    filenameR = s2.c_str();
    inR.open(filenameR, ios_base::in | ios::binary);
    if (!inR.is_open())
    {
        cout << " the file RIO0 open fail" << endl;
        return 1;
    }
    //

    //image line
    get_ele_position();

    cudaStatus = cudaMemcpyToSymbol(dev_ele_position_center, ele_position_center, sizeof(float) * ele_no);

    if (cudaStatus != cudaSuccess)
    {
        cout << "center Fail to cudaMemcpyToSymbol on GPU" << endl;
        return 1;
    }

    cudaMemcpyToSymbol(dev_ele_position_height, ele_position_height, sizeof(float) * ele_no);
    if (cudaStatus != cudaSuccess)
    {
        cout << "height Fail to cudaMemcpyToSymbol on GPU" << endl;
        return 1;
    }

    float *dev_sumdata;
    int *dev_sumpoint;

    cudaStatus = cudaMalloc((void **)(&dev_sumdata), N * N * sizeof(float));
    if (cudaStatus != cudaSuccess)
    {
        cout << "sumdata Fail to cudaMalloc on GPU" << endl;
        return 1;
        //return 1;
    }
    cudaStatus = cudaMalloc((void **)(&dev_sumpoint), N * N * sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        cout << "sumpoint Fail to cudaMalloc on GPU" << endl;
        return 1;
        //return 1;
    }

    cudaStatus = cudaMemcpy(dev_sumdata, allimage_data, N * N * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        cout << "sumdata Fail to cudaMemcpy on GPU" << endl;
        return 1;
    }
    cudaStatus = cudaMemcpy(dev_sumpoint, allpoint_count, N * N * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        cout << "sumpoint Fail to cudaMemcpy on GPU" << endl;
        return 1;
    }

    for (int trig_time = 0; trig_time < ele_no; trig_time++)
    //for (i=1;i<=1;i++)
    {
        cout << "Number of element : " << trig_time + 1 << endl;
        for (int ii = 0; ii < Nsample * ele_no; ii++)
        {
            //inR.read ( ( char * ) & uct_data_board0[ jj ][ ii ], sizeof(short));
            inR.read((char *)&uct_data_board0[ii], sizeof(short));
            //uct_data_board0[ ii]=before_convert_data;
        }
        // start=time(NULL);
        //for(int i=0;i<1024;i++){
        // if(trig_time%2==0)
        //{
        // continue;
        //}
        //cout<<uct_data_board0[0]<<endl;
        cudaStatus = precalcWithCuda(uct_data_board0, trig_time + 1, dev_sumdata, dev_sumpoint);
        //}
        // over=time(NULL);
        // cout<<"Running time is : "<<difftime(over,start)<<"s!"<<endl;
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "calcWithCuda failed!");
            return 1;
        }

        //////////////////
        //线

        // cudaError_t cudaStatus = calcWithCuda( i,dev_sumdata,dev_sumpoint,dev_filterdata);
    }
    cudaStatus = cudaMemcpy(allimage_data, dev_sumdata, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        cout << "allimagedata Fail to cudaMemcpy to CPU" << endl;
        return 1;
        //goto Error;
    }

    cudaStatus = cudaMemcpy(allpoint_count, dev_sumpoint, N * N * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        cout << "allpointcount Fail to cudaMemcpy to CPU" << endl;
        return 1;
        //goto Error;
    }
    //cout<<allimage_data[128200]<<endl;
    // cout<<allpoint_count[128200]<<endl;

    write_txtfile();
    over = time(NULL);
    cout << "Running time is : " << difftime(over, start) / 60 << "min!" << endl;
    inR.close();
    cudaFree(dev_sumdata);
    cudaFree(dev_sumpoint);
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
}

cudaError_t precalcWithCuda(short *uct_data_board0, int i, float *dev_sumdata, int *dev_sumpoint)
{
    cudaError_t cudaStatus;
    short *dev_uct_data_board0;
    float *dev_filterdata;

    cudaStatus = cudaMalloc((void **)(&dev_uct_data_board0), Nsample * ele_no * sizeof(short));
    if (cudaStatus != cudaSuccess)
    {
        cout << "uct_data_board0 Fail to cudaMalloc on GPU" << endl;
        //goto Error;
        return cudaStatus;
    }

    cudaStatus = cudaMalloc((void **)(&dev_filterdata), Nsample * ele_no * sizeof(float));
    if (cudaStatus != cudaSuccess)
    {
        cout << "uct_data_board0 Fail to cudaMalloc on GPU" << endl;
        //goto Error;
        return cudaStatus;
    }

    cudaStatus = cudaMemcpy(dev_uct_data_board0, uct_data_board0, Nsample * ele_no * sizeof(short), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        cout << "uct_data_board0 Fail to cudaMemcpy on GPU" << endl;
        //goto Error;
        return cudaStatus;
    }

    //kernel 1,kernel2 decode
    //kernel3 filter
    kernel3<<<8, 256>>>(dev_filterdata, dev_uct_data_board0);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        cout << "Kernel3 launch failed: " << cudaGetErrorString(cudaStatus);
        //goto Error;
        return cudaStatus;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel3!\n", cudaStatus);
        //goto Error;
        return cudaStatus;
    }

    //cudaStatus= cudaMemcpy( data1_1024_output,dev_filterdata , 5000*1024 * sizeof(float),cudaMemcpyDeviceToHost ) ;
    //  if (cudaStatus != cudaSuccess) {
    // cout<<"data_output Fail to cudaMemcpy to CPU"<<endl;
    //  goto Error;
    //goto Error;
    //  }

    float *dev_imagedata;
    // float *dev_trans_sdata;

    int *dev_pointcount;
    //  int dev_i;

    cudaStatus = cudaMalloc((void **)(&dev_imagedata), N * N * sizeof(float));
    if (cudaStatus != cudaSuccess)
    {
        cout << "imagedata Fail to cudaMalloc on GPU" << endl;
        //goto Error;
        return cudaStatus;
    }
    cudaStatus = cudaMalloc((void **)(&dev_pointcount), N * N * sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        cout << "pointcount Fail to cudaMalloc on GPU" << endl;
        //goto Error;
        return cudaStatus;
    }

    dim3 gridimage(N, N);
    //dim3 threads(M);
    kernel<<<gridimage, 2 * M>>>(i, dev_imagedata, dev_pointcount, dev_filterdata); //启动一个二维的N*N个block，每个block里面M个thread

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        cout << "calcKernel launch failed: " << cudaGetErrorString(cudaStatus);
        //goto Error;
        return cudaStatus;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching calcKernel!\n", cudaStatus);
        //goto Error;
        return cudaStatus;
    }
    // cudaDeviceSynchronize();

    //float *filter=(float*)malloc( N*N*sizeof(float));
    // cudaStatus= cudaMemcpy(filter,dev_imagedata ,  N*N * sizeof(float),cudaMemcpyDeviceToHost ) ;
    //  cout<<filter[128200]<<endl;
    //把所有的结果加到一起
    add<<<32, 32>>>(dev_sumdata, dev_sumpoint, dev_imagedata, dev_pointcount);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        cout << "addKernel launch failed: " << cudaGetErrorString(cudaStatus);
        //goto Error;
        return cudaStatus;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        //goto Error;
        return cudaStatus;
    }

Error:
    cudaFree(dev_uct_data_board0);
    cudaFree(dev_filterdata);
    cudaFree(dev_imagedata);
    cudaFree(dev_pointcount);
    return cudaStatus;
}

//cudaStatus= cudaMemcpy( data1_1024_output,dev_filterdata , 5000*1024 * sizeof(float),cudaMemcpyDeviceToHost ) ;
// if (cudaStatus != cudaSuccess) {
//cout<<"data_output Fail to cudaMemcpy to CPU"<<endl;
// goto Error;
//goto Error;
// }

void get_ele_position()
{
    //float rfocus=100.0/1000;
    //float first_one=2*PI*(45-39.1)/360;//第一个阵元角度
    // float ele_angle=(2*PI*39.1/(128-1))/360;//阵元间隔角度

    for (int i = 1; i < 257; i++)
    {
        ele_position_center[i - 1] = rfocus * cos(first_one + (i - 1) * ele_angle);
        ele_position_height[i - 1] = -rfocus * sin(first_one + (i - 1) * ele_angle);
    }
    for (int i = 257; i < 513; i++)
    {
        ele_position_center[i - 1] = rfocus * cos(first_one + (i - 257) * ele_angle + PI / 4);
        ele_position_height[i - 1] = -rfocus * sin(first_one + (i - 257) * ele_angle + PI / 4);
    }
    for (int i = 513; i < 769; i++)
    {
        ele_position_center[i - 1] = rfocus * cos(first_one + (i - 513) * ele_angle + PI / 2);
        ele_position_height[i - 1] = -rfocus * sin(first_one + (i - 513) * ele_angle + PI / 2);
    }
    for (int i = 769; i < 1025; i++)
    {
        ele_position_center[i - 1] = rfocus * cos(first_one + (i - 769) * ele_angle + 3 * PI / 4);
        ele_position_height[i - 1] = -rfocus * sin(first_one + (i - 769) * ele_angle + 3 * PI / 4);
    }
    for (int i = 1025; i < 1281; i++)
    {
        ele_position_center[i - 1] = rfocus * cos(first_one + (i - 1025) * ele_angle + PI);
        ele_position_height[i - 1] = -rfocus * sin(first_one + (i - 1025) * ele_angle + PI);
    }
    for (int i = 1281; i < 1537; i++)
    {
        ele_position_center[i - 1] = rfocus * cos(first_one + (i - 1281) * ele_angle + 5 * PI / 4);
        ele_position_height[i - 1] = -rfocus * sin(first_one + (i - 1281) * ele_angle + 5 * PI / 4);
    }
    for (int i = 1537; i < 1793; i++)
    {
        ele_position_center[i - 1] = rfocus * cos(first_one + (i - 1537) * ele_angle + 3 * PI / 2);
        ele_position_height[i - 1] = -rfocus * sin(first_one + (i - 1537) * ele_angle + 3 * PI / 2);
    }
    for (int i = 1793; i < 2049; i++)
    {
        ele_position_center[i - 1] = rfocus * cos(first_one + (i - 1793) * ele_angle + 7 * PI / 4);
        ele_position_height[i - 1] = -rfocus * sin(first_one + (i - 1793) * ele_angle + 7 * PI / 4);
    }
}
void write_txtfile()
{
    ofstream outfile("filter_das_2048element_052A_1520_224_40du_2048_tgc0014_25M.txt");
    if (!outfile.is_open())
    {
        cout << " the file open fail" << endl;
        exit(1);
    }

    for (int k = 0; k < N; k++)
    {
        for (int j = 0; j < N; j++)
        {
            if (allpoint_count[k * N + j] == 0)
                outfile << allimage_data[k * N + j] << " ";
            else
                outfile << allimage_data[k * N + j] / allpoint_count[k * N + j] << " ";
        }
        outfile << "\r\n";
    }

    outfile.close();
}
