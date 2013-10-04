/*
	Procesas padaro atsitiktini skaiciu iteraciju
	Atsitiktine
*/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cuda.h>

#include <omp.h>
#include <string>
#include <fstream>
#include <vector>
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace std;

//struktura, kuri bus kopijuojama i GPU su rodykle i teksto eilute
struct GpuStruct
{
	char *pav;
	int kiekis;
	double kaina;
};

//pagrindine struktura, kuri bus sukuriama skaitant duomenis
class Struct
{
	string pav;
	int kiekis;
	double kaina;
	GpuStruct gpuStruct;
public:
	Struct(string input);
	~Struct(){cudaFree(gpuStruct.pav);}//destruktoriuje istrinama teksto eilute is GPU
	GpuStruct GetDev(){return gpuStruct;}
	string Print();
};

//is duomenu eilutes sukuriama struktura, sukuriama struktura kuri bus GPU, alokuojama teksto eilute GPU
Struct::Struct(string input)
{
	int start, end;
	start = 0;
	end = input.find(' ');
	pav = input.substr(0, end).c_str();
	start = end + 1;
	end = input.find(' ', start);
	kiekis = stoi(input.substr(start, end - start));
	start = end + 1;
	kaina = stod(input.substr(start));
	gpuStruct.kaina = kaina;
	gpuStruct.kiekis = kiekis;
	cudaMalloc(&gpuStruct.pav, pav.size() + 1);
	cudaMemcpy(gpuStruct.pav, pav.c_str(), pav.size() + 1, cudaMemcpyHostToDevice);
}

string Struct::Print()
{
	stringstream ss;
	ss << setw(15) << pav << setw(7) << kiekis << setw(20) << kaina;
	return ss.str();
}

vector<vector<Struct>> ReadStuff(string file);
vector<string> ReadLines(string file);

string Titles();
string Print(int nr, Struct &s);
void syncOut(vector<vector<Struct>>&);

__global__ void DevPrint(GpuStruct *data, int* starts);

int main()
{
	auto input = ReadStuff("LapunasD.txt");
	int count = 0;
	//suskaiciuojama kiek is viso yra duomenu
	for(auto &vec : input)
		count += vec.size();
	cout << "\nsinchroninis isvedimas\n\n";
	syncOut(input);
	cout << "\nasinchroninis isvedimas\n\n";
	cout << setw(10) << "Procesas" << setw(3) << "Nr" << Titles() << "\n\n";
	
	//procesu duomenu pradzios indeksai
	vector<int> starts;
	//lokalios GPU strukturu kopijos
	vector<GpuStruct> localStructs;
	
	int put = 0;
	for(auto &vec : input)
	{
		//proceso pradzia
		starts.push_back(put);
		for(auto &s : vec)
		{
			localStructs.push_back(s.GetDev());
			put++;
		}
	}
	starts.push_back(put);
	int *startsdev;
	//pradziu masyvas GPU
	cudaMalloc(&startsdev, sizeof(int) * starts.size());
	cudaMemcpy(startsdev, &starts[0], sizeof(int) * starts.size(), cudaMemcpyHostToDevice);
	GpuStruct *arr;
	//strukturu masyvas GPU
	cudaMalloc(&arr, sizeof(GpuStruct) * count);
	cudaMemcpy(arr, &localStructs[0], sizeof(GpuStruct) * count, cudaMemcpyHostToDevice);
	//GPU funkcija
	DevPrint<<<1, input.size()>>>(arr, startsdev);
	//palaukiam kol gpu baigs spausdint, "pause" uzrakina konsole
	cudaDeviceSynchronize();
	system("pause");
	//atlaisvinami pagrindiniai masyvai, teksto eilutes atlaisvinamos sunaikintant pagrindines strukturas - input
	cudaFree(arr);
	cudaFree(startsdev);
	return 0;
}

//is failo skaitomi duomenis i strukturu matrica
vector<vector<Struct>> ReadStuff(string file)
{
	//eiluciu vektorius
	auto lines = ReadLines(file);
	//duomenu matrica
	vector<vector<Struct>> ret;
	//laikinas vektorius vieno proceso duomenim
	vector<Struct> tmp;
	for(int i = 0; i < lines.size(); i++)
	{
		//tuscia eilute skiria procesus
		if(lines[i] == "")
		{
			ret.push_back(move(tmp));
		}
		else
		{
			tmp.emplace_back(lines[i]);
		}
	}
	return ret;
}

//failas skaitomas i eiluciu vektoriu
vector<string> ReadLines(string file)
{
	vector<string> ret;
	ifstream duom(file);
	while(!duom.eof())
	{
		string line;
		getline(duom, line);
		ret.push_back(line);
	}
	return ret;
}

string Titles()
{
	stringstream ss;
	ss << setw(15) << "Pavadiniams" << setw(7) << "Kiekis" << setw(20) << "Kaina";
	return ss.str();
}

//grazus sinchroninis spausdinimas
void syncOut(vector<vector<Struct>> &data)
{
	cout << setw(3) << "Nr" << Titles() << endl << endl;
	for(int i = 0; i < data.size(); i++)
	{
		auto &vec = data[i];
		cout << "Procesas_" << i << endl;
		for(int j = 0; j < vec.size(); j++)
		{
			cout << Print(j, vec[j]) << endl;
		}
	}
}

string Print(int nr, Struct &s)
{
	stringstream ss;
	ss << setw(3) << nr << s.Print();
	return ss.str();
}

//GPU funkcija spausdinimui
//gauna duomenis ir indeksus kur prasideda kieno duomenys
__global__ void DevPrint(GpuStruct *data, int *starts)
{
	int id = threadIdx.x;
	GpuStruct *d = data + starts[id];//proceso duomenu pradzia
	int count = starts[id+1] - starts[id];//proceso duomenu kiekis
	for(int i = 0; i < count; i++)
	{
		printf("Procesas_%i %2i %14s %6i %19f\n", id, i, d[i].pav, d[i].kiekis, d[i].kaina);
	}
}