#include<iostream>
#include<immintrin.h>
#include<cmath>
#include<time.h>
#include<fstream>
#include<omp.h>



#define max_N 10000

using namespace std;

float A[max_N][max_N];
float B[max_N / 8][max_N * 8];

void f1(int m, int n, float a[][max_N]) {
	__m256 t1, t2, c;
	for (int k = 0; k < (m < n ? m : n); k++)
	{
		//选取k列绝对值最大的元素为主元,主元在row行
		float s = abs(a[k][k]);
		int row = k;
		for (int i = k + 1; i < m; i++)
		{
			float t = abs(a[i][k]);
			if (s < t)
			{
				s = t;
				row = i;
			}
		}
		//k列全为0则不需处理
		if (s == 0)
			continue;
		
		//交换row行与k行
		if (row != k)
			for (int j = 0; j < n; j++)
				swap(a[k][j], a[row][j]);
		for (int j = k + 1; j < n; j++)
			a[k][j] = a[k][j] / a[k][k];
		a[k][k] = 1;
		
		int j;
#pragma omp parallel for private(t1,t2,c,j) num_threads(6)
		for (int i = k + 1; i < m; i++)
		{
			c = _mm256_set1_ps(a[i][k]);
			for (j = k + 1; j <= n - 8; j += 8)
			{
				t1 = _mm256_loadu_ps(a[k] + j);
				t2 = _mm256_loadu_ps(a[i] + j);
				t1 = _mm256_mul_ps(t1, c);
				t2 = _mm256_sub_ps(t2, t1);
				_mm256_store_ps(a[i] + j, t2);
			}
		}
		
		for (int j = n - (n - k - 1) % 8; j < n; j++)
			for (int i = k + 1; i < m; i++)
				a[i][j] = a[i][j] - a[i][k] * a[k][j];
		for (int i = k + 1; i < m; i++)
			a[i][k] = 0;
		
	}
}


void f2(int m, int n, float a[][max_N],float b[][max_N*8]) {
	float tot = 0;
	time_t start, end;
	for(int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			b[j / 8][8 * i + j % 8] = a[i][j];
	__m256 t1, t2, t3, c;
	for (int k = 0; k < (m < n ? m : n); k++)
	{
		float s = abs(b[k / 8][8 * k + k % 8]);
		int row = k;
		for (int i = k + 1; i < m; i++)
		{
			float t = abs(b[k / 8][8 * i + k % 8]);
			if (s < t)
			{
				s = t;
				row = i;
			}
		}
		if (s == 0)
			continue;


		if (row != k)
			for (int j = 0; j < n; j++)
				swap(b[j / 8][8 * k + j % 8], b[j / 8][8 * row + j % 8]);
		for (int j = k + 1; j < n; j++)
			b[j / 8][8 * k + j % 8] = b[j / 8][8 * k + j % 8] / b[k / 8][8 * k + k % 8];
		b[k / 8][8 * k + k % 8] = 1;

		for (int i = 8 * (k + 1); i < 8 * m; i += 8)
			for (int j = k % 8 + 1; j < 8; j++)
				b[(k + 1) / 8][i + j] = b[(k + 1) / 8][i + j] - b[(k + 1) / 8][i + k % 8] * b[(k + 1) / 8][8 * k + j];
		int i, j;
		start = clock();
		#pragma omp parallel for private(t1,t2,t3,c,i,j) num_threads(6)
		for (i = ((k + 1) % 8 == 0 ? (k + 1) / 8 : 1 + (k + 1) / 8); i <= (n - 1) / 8; i++)
		{
			t1 = _mm256_loadu_ps(b[i] + 8 * k);
			for (j = 8 * (k + 1); j < 8 * m; j += 8)
			{
				t2 = _mm256_loadu_ps(b[i] + j);
				c = _mm256_set1_ps(b[k / 8][j + k % 8]);
				t3 = _mm256_mul_ps(t1, c);
				t2 = _mm256_sub_ps(t2, t3);
				_mm256_store_ps(b[i] + j, t2);
			}
		}
		end = clock();
		tot += (float)(end - start) / 1000;
		for (int i = k + 1; i < m; i++)
			b[k / 8][i * 8 + k % 8] = 0;
	}


	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			a[i][j] = b[j / 8][8 * i + j % 8];

}


void f3(int m, int n, float a[][max_N]) {
	__m256 t1, t2, c;
	for (int k = 0; k < (m < n ? m : n); k++)
	{
		//选取k列绝对值最大的元素为主元,主元在row行
		float s = abs(a[k][k]);
		int row = k;
		for (int i = k + 1; i < m; i++)
		{
			float t = abs(a[i][k]);
			if (s < t)
			{
				s = t;
				row = i;
			}
		}
		//k列全为0则不需处理
		if (s == 0)
			continue;
		//交换row行与k行
		if (row != k)
			for (int j = 0; j < n; j++)
				swap(a[k][j], a[row][j]);
		for (int j = k + 1; j < n; j++)
			a[k][j] = a[k][j] / a[k][k];
		a[k][k] = 1;
		int i, j;
		int T = 120;
		int d1 = (m - k - 1) % T == 0 ? (m - k - 1) / T : 1 + (m - k - 1) / T;
		int d2 = (n - k - 1) / 8;
		d2 = d2 % (T / 8) == 0 ? d2 / (T / 8) : 1 + d2 / (T / 8);
#pragma omp parallel for private(t1,t2,c,i,j) num_threads(6)
		for (int l = 0; l < d1*d2; l++)
		{
			int l1 = k + 1 + (l / d2)*T;
			int l2 = k + 1 + (l % d2)*T;
			for (i = l1; i < (l1 + T < m ? l1 + T : m); i++)
			{
				c = _mm256_set1_ps(a[i][k]);
				for (j = l2; j <= (l2 + T - 8 < n - 8 ? l2 + T - 8 : n - 8); j += 8)
				{
					t1 = _mm256_loadu_ps(a[k] + j);
					t2 = _mm256_loadu_ps(a[i] + j);
					t1 = _mm256_mul_ps(t1, c);
					t2 = _mm256_sub_ps(t2, t1);
					_mm256_store_ps(a[i] + j, t2);
				}
			}
		}

		for (int j = n - (n - k - 1) % 8; j < n; j++)
			for (int i = k + 1; i < m; i++)
				a[i][j] = a[i][j] - a[i][k] * a[k][j];
		for (int i = k + 1; i < m; i++)
			a[i][k] = 0;
	}
}


int main()
{
	ifstream in("data.txt");
	if (!in.is_open()) {
		cerr << "Failed to open file!" << endl;
		return -1;
	}
	int m, n;
	in >> m;
	in >> n;
	int i, j;
	for (i = 0; i < m; i++)
	{
		for (j = 0; j < n; j++)
		{
			in >> A[i][j];
		}
	}
	time_t start, end;
	/*
	start = clock();
	f1(m, n, A);
	end = clock();
	cout << "Time useage: " << (float)(end - start) / 1000 << " s" << endl;
	cout << A[m - 1][n - 1] << endl;
	*/
	/*
	start = clock();
	f2(m, n, A, B);
	end = clock();
	cout << "Time useage: " << (float)(end - start) / 1000 << " s" << endl;
	cout << A[m - 1][n - 1] << endl;
	*/
	
	start = clock();
	f3(m, n, A);
	end = clock();
	cout << "Time useage: " << (float)(end - start) / 1000 << " s" << endl;
	cout << A[m-1][n-1] << endl;
	
	/*
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
			cout << A[i][j] << " ";
			cout << endl;
	}*/
}