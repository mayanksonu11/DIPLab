#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
using namespace std;
using namespace cv;
int image_slider_max;
int image_slider;
const int filter_slider_max = 8;
int filter_slider;
const int neighbour_slider_max = 3;
int neighbour_slider;
vector<vector<uchar>> image;
vector <Mat> image_stack;
Mat new_image;
void display(vector<uchar> &a){ //for debugging purpose
    for(int i=0;i<a.size();i++){
        cout << (int)a[i] << " " ;
    }
}

vector<vector<uchar>> convert_mat_to_array(Mat m, int row, int col){ // used to 2d matrix from Mat
    vector<vector<uchar>> image(row, vector<uchar>(col));
    for(int i=0;i<m.rows;i++){
        for(int j=0;j<m.cols;j++){
            image[i][j] = m.at<uchar>(i,j,0); 
        }
    }       

    return image;
}

vector<vector<uchar>> mean(vector<vector<uchar>>& image, int kernel_size)
{
    cout << "Mean filter "<< kernel_size << "x" << kernel_size << endl;
    int sum;
    vector<vector<uchar>> output = image;
    for(int i = 0; i< image.size(); i++)
    {
        for(int j = 0; j< image[0].size(); j++)
        {
            sum = 0;
            for(int m = -kernel_size/2; m<= kernel_size/2; m++)
            {
                for(int n = -kernel_size/2; n<= kernel_size/2; n++)
                {
                    if(i+m>=0 && i+m< image.size() && j+n >=0 && j+n < image[0].size())  sum += image[i+m][j+n];
                }
            }
            output[i][j] = sum/(kernel_size*kernel_size);
        }
    }
    return output;
}

vector<vector<uchar>> median(vector<vector<uchar>>& image, int kernel_size)
{
    cout << "Median Filter "<< kernel_size << "x" << kernel_size << endl;
    vector<vector<uchar>> output = image;
    for(int i = 0; i< image.size(); i++)
    {
        for(int j = 0; j< image[0].size(); j++)
        {
            vector<uchar> med;
            for(int m = -kernel_size/2; m<= kernel_size/2; m++)
            {
                for(int n = -kernel_size/2; n<= kernel_size/2; n++)
                {
                    if(i+m>=0 && i+m< image.size() && j+n >=0 && j+n < image[0].size())  med.push_back(image[i+m][j+n]);
                    else med.push_back(0);
                }
            }
            sort(med.begin(),med.end());
            output[i][j] = med[med.size()/2];
        }
    }
    return output;
}

vector<vector<uchar>> prewitt(vector<vector<uchar>>& image, int kernel_size)
{
    cout << "Prewitt Filter "<< kernel_size << "x" << kernel_size << endl;
    vector<vector<uchar>> output = image;
    int gx, gy;
    vector<vector<int>> prew_x(kernel_size, vector<int>(kernel_size,0));
    vector<vector<int>> prew_y = prew_x;
    for(int i = 0; i<kernel_size; i++)
    {
        for(int j = 0; j<kernel_size; j++)
        {
            if(i<kernel_size/2) prew_y[i][j] = 1;
            else if(i>kernel_size/2) prew_y[i][j] = -1;
            if(j<kernel_size/2) prew_x[i][j] = 1;
            else if(j>kernel_size/2) prew_x[i][j] = -1;
        }
    }
    for(int i = 0; i< image.size(); i++)
    {
        for(int j = 0; j< image[0].size(); j++)
        {
            gx = 0; gy = 0;
            for(int m = -kernel_size/2; m<= kernel_size/2; m++)
            {
                for(int n = -kernel_size/2; n<= kernel_size/2; n++)
                {
                    if(i+m>=0 && i+m< image.size() && j+n >=0 && j+n < image[0].size())
                    {
                        gx += prew_x[m+kernel_size/2][n+kernel_size/2]*image[i+m][j+n];
                        gy += prew_y[m+kernel_size/2][n+kernel_size/2]*image[i+m][j+n];
                    }
                }
            }
            output[i][j] = max(0,min(255,(int)sqrt(gx*gx + gy*gy)));
        }
    }
    return output;
}

vector<vector<uchar>> laplacian(vector<vector<uchar>>& image, int kernel_size)
{
    cout << "Laplacian Filter " << kernel_size << "x" << kernel_size <<endl; 
    int sum;
    vector<vector<uchar>> output = image;
    vector<vector<int>> lap(kernel_size, vector<int>(kernel_size,1));
    lap[kernel_size/2][kernel_size/2] = 1 - kernel_size*kernel_size;
    for(int i = 0; i< image.size(); i++)
    {
        for(int j = 0; j< image[0].size(); j++)
        {
            sum = 0;
            for(int m = -kernel_size/2; m<= kernel_size/2; m++)
            {
                for(int n = -kernel_size/2; n<= kernel_size/2; n++)
                {
                    if(i+m>=0 && i+m< image.size() && j+n >=0 && j+n < image[0].size())  sum += lap[m+kernel_size/2][n+kernel_size/2]*image[i+m][j+n];
                }
            }
            output[i][j] = (uchar)max(0,min(255,sum));
        }
    }
    return output;
}

vector<vector<uchar>> sobel_horizontal(vector<vector<uchar>>& image, int kernel_size)
{
    cout << "Sobel Horizontal Filter "<< kernel_size << "x" << kernel_size << endl;
    int sum;
    vector<vector<uchar>> output = image;
    vector<vector<int>> sobel_x(kernel_size, vector<int>(kernel_size,0));
    for(int i = 0; i<kernel_size; i++)
    {
        for(int j = 0; j<kernel_size; j++)
        {
            if(j<kernel_size/2) sobel_x[i][j] = min(i+1,kernel_size-i);
            else if(j>kernel_size/2) sobel_x[i][j] = -min(i+1,kernel_size-i);
        }
    }
    for(int i = 0; i< image.size(); i++)
    {
        for(int j = 0; j< image[0].size(); j++)
        {
            sum = 0;
            for(int m = -kernel_size/2; m<= kernel_size/2; m++)
            {
                for(int n = -kernel_size/2; n<= kernel_size/2; n++)
                {
                    if(i+m>=0 && i+m< image.size() && j+n >=0 && j+n < image[0].size())
                    {
                        sum += sobel_x[m+kernel_size/2][n+kernel_size/2]*image[i+m][j+n];
                    }
                }
            }
            output[i][j] = max(0,min(255,sum));
        }
    }
    return output;
}
vector<vector<uchar>> sobel_vertical(vector<vector<uchar>>& image, int kernel_size)
{
    cout << "Sobel Vertical Filter " << kernel_size << "x" << kernel_size<< endl;
    int sum;
    vector<vector<uchar>> output = image;
    vector<vector<int>> sobel_y(kernel_size, vector<int>(kernel_size,0));
    for(int i = 0; i<kernel_size; i++)
    {
        for(int j = 0; j<kernel_size; j++)
        {
            if(i<kernel_size/2) sobel_y[i][j] = min(j+1,kernel_size-j);
            else if(i>kernel_size/2) sobel_y[i][j] = -min(j+1,kernel_size-j);
        }
    }
    for(int i = 0; i< image.size(); i++)
    {
        for(int j = 0; j< image[0].size(); j++)
        {
            sum = 0;
            for(int m = -kernel_size/2; m<= kernel_size/2; m++)
            {
                for(int n = -kernel_size/2; n<= kernel_size/2; n++)
                {
                    if(i+m>=0 && i+m< image.size() && j+n >=0 && j+n < image[0].size())
                    {
                        sum += sobel_y[m+kernel_size/2][n+kernel_size/2]*image[i+m][j+n];
                    }
                }
            }
            output[i][j] = max(0,min(255,sum));
        }
    }
    return output;
}

vector<vector<uchar>> sobel_diagonal(vector<vector<uchar>>& image, int kernel_size)
{
    cout << "Sobel Diagonal Filter " << kernel_size << "x" << kernel_size << endl;
    int sum;
    vector<vector<uchar>> output = image;
    vector<vector<int>> sobel_d(kernel_size, vector<int>(kernel_size,0));
    for(int i = 0; i<kernel_size; i++)
    {
        for(int j = 0; j<kernel_size; j++)
        {
            if(i>j) sobel_d[i][j] = j-i;
            else if(i<j) sobel_d[i][j] = j-i;
        }
    }
    for(int i = 0; i< image.size(); i++)
    {
        for(int j = 0; j< image[0].size(); j++)
        {
            sum = 0;
            for(int m = -kernel_size/2; m<= kernel_size/2; m++)
            {
                for(int n = -kernel_size/2; n<= kernel_size/2; n++)
                {
                    if(i+m>=0 && i+m< image.size() && j+n >=0 && j+n < image[0].size())
                    {
                        sum += sobel_d[m+kernel_size/2][n+kernel_size/2]*image[i+m][j+n];
                    }
                }
            }
            output[i][j] = max(0,min(255,sum));
        }
    }
    return output;
}

vector<vector<uchar>> gaussian(vector<vector<uchar>>& image, int kernel_size)
{
    cout << "Gaussian Filter "<< kernel_size << "x" << kernel_size << endl;
    double sum;
    vector<vector<uchar>> output = image;
    vector<vector<double>> gauss(kernel_size, vector<double>(kernel_size,0));
    for(int i = 0; i<kernel_size; i++)
    {
        for(int j = 0; j<kernel_size; j++)
        {
            int x = kernel_size/2 - i, y = kernel_size/2 - j;
            gauss[i][j] = exp((x*x + y*y)/2.0);
            sum += gauss[i][j];
        }
    }
    for(int i = 0; i<kernel_size; i++)
    {
        for(int j = 0; j<kernel_size; j++)
        {
            gauss[i][j] /= sum;
        }
    }

    for(int i = 0; i< image.size(); i++)
    {
        for(int j = 0; j< image[0].size(); j++)
        {
            sum = 0;
            for(int m = -kernel_size/2; m<= kernel_size/2; m++)
            {
                for(int n = -kernel_size/2; n<= kernel_size/2; n++)
                {
                    if(i+m>=0 && i+m< image.size() && j+n >=0 && j+n < image[0].size())
                    {
                        sum += gauss[m+kernel_size/2][n+kernel_size/2]*image[i+m][j+n];
                    }
                }
            }
            output[i][j] = max(0,min(255,(int)round(sum)));
        }
    }
    return output;
}

vector<vector<uchar>> LoG(vector<vector<uchar>>& image, int kernel_size)
{
    cout << "Laplacian of Gaussian Filter "<< kernel_size << "x" << kernel_size << endl;
    double sum;
    vector<vector<uchar>> output = image;
    vector<vector<double>> lap_of_gauss(kernel_size, vector<double>(kernel_size,0));
    for(int i = 0; i<kernel_size; i++)
    {
        for(int j = 0; j<kernel_size; j++)
        {
            int x = kernel_size/2 - i, y = kernel_size/2 - j;
            lap_of_gauss[i][j] = ((x*x+y*y)/2.0 - 1)*exp((x*x + y*y)/2.0);
            sum += lap_of_gauss[i][j];
        }
    }
    for(int i = 0; i<kernel_size; i++)
    {
        for(int j = 0; j<kernel_size; j++)
        {
            lap_of_gauss[i][j] /= sum;
        }
    }

    for(int i = 0; i< image.size(); i++)
    {
        for(int j = 0; j< image[0].size(); j++)
        {
            sum = 0;
            for(int m = -kernel_size/2; m<= kernel_size/2; m++)
            {
                for(int n = -kernel_size/2; n<= kernel_size/2; n++)
                {
                    if(i+m>=0 && i+m< image.size() && j+n >=0 && j+n < image[0].size())
                    {
                        sum += lap_of_gauss[m+kernel_size/2][n+kernel_size/2]*image[i+m][j+n];
                    }
                }
            }
            output[i][j] = max(0,min(255,(int)round(sum)));
        }
    }
    return output;
}

Mat convert_array_to_mat(vector<vector<uchar>> output,int row,int col){
    Mat conv(row,col,CV_8U);
    for(int i=0;i<row;i++){
        for(int j=0;j<col;j++) {
            conv.at<uchar>(i,j,0) = output[i][j];
        }
    }
    return conv;
}
static void on_trackbar( int ,void* )
{
    int n;
    if(neighbour_slider == 0) n = 3;
    else if(neighbour_slider == 1) n = 5;
    else if(neighbour_slider == 2) n = 7;
    else if(neighbour_slider == 3) n = 9;
    
    int col = image_stack[image_slider].cols;
    int row = image_stack[image_slider].rows;
    image = convert_mat_to_array( image_stack[image_slider],row,col);
    new_image = Mat::zeros(row,col,CV_8U);

    if(filter_slider==0) new_image = convert_array_to_mat(mean(image,n), row, col);  
    else if(filter_slider == 1)  new_image = convert_array_to_mat(median(image,n),row,col);
    else if(filter_slider == 2)  new_image = convert_array_to_mat(prewitt(image,n),row,col);
    else if(filter_slider == 3)  new_image = convert_array_to_mat(laplacian(image,n),row,col);
    else if(filter_slider == 4)  new_image = convert_array_to_mat(sobel_horizontal(image,n),row,col);
    else if(filter_slider == 5)  new_image = convert_array_to_mat(sobel_vertical(image,n),row,col);
    else if(filter_slider == 6)  new_image = convert_array_to_mat(sobel_diagonal(image,n),row,col);
    else if(filter_slider == 7)  new_image = convert_array_to_mat(gaussian(image,n),row,col);
    else if(filter_slider == 8)  new_image = convert_array_to_mat(LoG(image,n),row,col);

    imshow("Input Image", image_stack[image_slider]);
    imshow("Filter Output", new_image);
    
}

int main(int argc, char** argv)
{
    vector<cv::String> fn;
    glob("images/*.jpg", fn, false);
    size_t count = fn.size(); 
    for (size_t i=0; i<count; i++)
        image_stack.push_back(imread(fn[i]));


    image_slider_max = image_stack.size()-1;
    image_slider = 0;
    neighbour_slider = 0;
    filter_slider = 0;
    namedWindow("Filter Output", WINDOW_AUTOSIZE); // Create Window
    char TrackbarName_image[50];
    char TrackbarName_filter[50];
    char TrackbarName_neighbour[50];
    sprintf( TrackbarName_image, "Image x %d", image_slider_max );
    createTrackbar( TrackbarName_image, "Filter Output", &image_slider, image_slider_max, on_trackbar );
    sprintf( TrackbarName_filter, "Filter x %d", filter_slider_max );
    createTrackbar( TrackbarName_filter, "Filter Output", &filter_slider, filter_slider_max, on_trackbar );
    sprintf( TrackbarName_neighbour, "Neighbour x %d", filter_slider_max );
    createTrackbar( TrackbarName_neighbour, "Filter Output", &neighbour_slider, neighbour_slider_max, on_trackbar );
    on_trackbar( image_slider, 0 );
    
    waitKey(0);
    return 0;
}