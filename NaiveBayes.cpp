/*
@author Mitchell Mears
@title Naive Bayes
Program implements read_csv
by Ben Gorman
date: Jan, 16, 2019
availible https://www.gormanalysis.com/blog/reading-and-writing-csv-files-with-cpp/
*/

#include <string>
#include <fstream>
#include <vector>
#include <utility> 
#include <stdexcept> 
#include <sstream> 
#include <iostream>
#include <cmath>
#include <ctime>
# define M_PI           3.14159265358979323846  /* pi */
using namespace std;

vector<pair<string, vector<double>>> read_csv(string filename) {
	// Reads a CSV file into a vector of <string, vector<int>> pairs where
	// each pair represents <column name, column values>

	// Create a vector of <string, int vector> pairs to store the result
	vector<pair<string, vector<double>>> result;

	// Create an input filestream
	ifstream myFile(filename);

	// Make sure the file is open
	if (!myFile.is_open()) throw runtime_error("Could not open file");


	string line, colname;
	double val;

	// Read the column names
	if (myFile.good())
	{
		// Extract the first line in the file
		getline(myFile, line);

		// Create a stringstream from line
		stringstream ss(line);

		// Extract each column name
		while (getline(ss, colname, ',')) {

			// Initialize and add <colname, int vector> pairs to result
			result.push_back({ colname, vector<double> {} });
		}
	}

	// Read data, line by line
	while (getline(myFile, line))
	{
		// Create a stringstream of the current line
		stringstream ss(line);

		// Keep track of the current column index
		int colIdx = 0;

		// Extract each integer
		while (ss >> val) {

			// Add the current integer to the 'colIdx' column's values vector
			result.at(colIdx).second.push_back(val);

			// If the next token is a comma, ignore it and move on
			if (ss.peek() == ',') ss.ignore();

			// Increment the column index
			colIdx++;
		}
	}

	// Close file
	myFile.close();

	return result;
}


//computes apriori probability by counting initial instances of a chosen factor
double computeAprioriProb(string columnName, vector<double> column, int n = 0)
{
	int count = 0;
	int size = n == 0 ? column.size() : n;
	for (int i = 0; i < size; i++)
	{
		if (column[i] == 0)
		{
			count++;
		}
	}
	return (double)(size - count) / size;
}

vector<double> computeDependent(vector<double>& unknown, vector<double>& given, int givval, int ukval = -1)
{
	vector<double> dependents;
	for (int i = 0; i < unknown.size(); i++)
	{
		if ((ukval == -1 || unknown[i] == ukval) && given[i] == givval)
		{
			dependents.push_back(unknown[i]);
		}
	}
	return dependents;
}

//this function computes the likelihood of a vector based on another vector
vector<double> computeDependencyMatrix(
	vector<double>& col1, vector<double>& col2,
	double apriori_2, int low1, int high1, int val2, int n = 0)
{
	vector<double> likelihood_12;
	int size = n == 0 ? col1.size() : n;
	for (int i = low1; i <= high1; i++)
	{

		int count_i2 = computeDependent(col1, col2, val2, i).size();
		double ratio_ci2_2 = count_i2 / (apriori_2 * size);
		likelihood_12.push_back(ratio_ci2_2);
	}

	return likelihood_12;
}
//function calculates mean of a given column
double mean(vector<double> col, int n = 0) {
	int size = n == 0 ? col.size() : n;
	double sum = 0;
	for (int i = 0; i < size; i++)
	{
		sum += col[i];

	}
	return (sum / size);
}

//function calculates variance of a given vector
double variance(const vector<double>& col, double mean, int n = 0)
{

	int size = n == 0 ? col.size() : n;
	double sum = 0.0;
	for (int i = 0; i < size; ++i)
	{
		sum += (col[i] - mean) * (col[i] - mean);
	}
	return sum / (size - 1);
}

//1 / sqrt(2 * pi * var_v) * exp(-((v - mean_v) ^ 2) / (2 * var_v))

double ageLikelihood(double mean, double var, double age)
{
	double e = exp(-(age - mean)*(age - mean) / (2 * var));
	return 1 / sqrt(2 * M_PI* var) * e;
}
//This function calculates total probabilty of survival given age, sex, and passenger class. 
//has an option to offest. This is used to seperate training data and testing data.
//first holds all probabilities of survive second holds all died probabilities
pair<vector<double>, vector<double>> calc_raw_prob(vector<double> col1, vector<double> col2,
	vector<double> col3, vector<double> col4,
	double surviveApriori, double diedApriori,
	vector<double> pclassSurvive, vector <double>pclassDied,
	vector<double> sexSurvive, vector<double> sexDied,
	double ageMeanSurvive, double ageMeanDie, double ageVarDie, double ageVarSurvive, int n = 0, int offset = 0) {

	pair<vector<double>, vector<double>> rawProb;
	double prob;
	double denom = 0;
	double numeratorD = 0;
	double numeratorS = 0;
	int size = n == 0 ? col1.size() : n;

	for (int i = offset; i < size; i++)
	{
		denom = (pclassDied[col1[i] - 1] * sexDied[col3[i]] * diedApriori
			*ageLikelihood(ageMeanDie, ageVarDie, col4[i])
			+
			pclassSurvive[col1[i] - 1] * sexSurvive[col3[i]]
			* ageLikelihood(ageMeanSurvive, ageVarSurvive, col4[i])*surviveApriori);
		//sex = 0
		if (col3[i] == 0)
		{
			//pclass 1
			if (col1[i] == 1)
			{

				numeratorS = (pclassSurvive[0] * sexSurvive[0] * surviveApriori*ageLikelihood(ageMeanSurvive, ageVarSurvive, col4[i]));
				prob = numeratorS / denom;
				rawProb.first.push_back(prob);

				numeratorD = (pclassDied[0] * sexDied[0] * diedApriori*ageLikelihood(ageMeanDie, ageVarDie, col4[i]));
				prob = numeratorD / denom;
				rawProb.second.push_back(prob);
			}
			//pclass 2
			else if (col1[i] == 2)
			{

				numeratorS = (pclassSurvive[1] * sexSurvive[0] * surviveApriori*ageLikelihood(ageMeanSurvive, ageVarSurvive, col4[i]));
				prob = numeratorS / denom;
				rawProb.first.push_back(prob);

				numeratorD = (pclassDied[1] * sexDied[0] * diedApriori*ageLikelihood(ageMeanDie, ageVarDie, col4[i]));
				prob = numeratorD / denom;
				rawProb.second.push_back(prob);
			}
			//pclass 3
			else
			{

				numeratorS = (pclassSurvive[2] * sexSurvive[0] * surviveApriori*ageLikelihood(ageMeanSurvive, ageVarSurvive, col4[i]));
				prob = numeratorS / denom;
				rawProb.first.push_back(prob);

				numeratorD = (pclassDied[2] * sexDied[0] * diedApriori*ageLikelihood(ageMeanDie, ageVarDie, col4[i]));
				prob = numeratorD / denom;
				rawProb.second.push_back(prob);
			}
		}
		//sex = 1
		else
		{
			//pclass1
			if (col1[i] == 1)
			{

				numeratorS = (pclassSurvive[0] * sexSurvive[1] * surviveApriori*ageLikelihood(ageMeanSurvive, ageVarSurvive, col4[i]));
				prob = numeratorS / denom;

				numeratorD = (pclassDied[0] * sexDied[1] * diedApriori*ageLikelihood(ageMeanDie, ageVarDie, col4[i]));
				prob = numeratorD / denom;
				rawProb.second.push_back(prob);
			}
			//pclass 2
			else if (col1[i] == 2)
			{

				numeratorS = (pclassSurvive[1] * sexSurvive[1] * surviveApriori*ageLikelihood(ageMeanSurvive, ageVarSurvive, col4[i]));
				prob = numeratorS / denom;
				rawProb.first.push_back(prob);

				numeratorD = (pclassDied[1] * sexDied[1] * diedApriori*ageLikelihood(ageMeanDie, ageVarDie, col4[i]));
				prob = numeratorD / denom;
				rawProb.second.push_back(prob);
			}
			//pclass 3
			else
			{

				numeratorS = (pclassSurvive[2] * sexSurvive[1] * surviveApriori*ageLikelihood(ageMeanSurvive, ageVarSurvive, col4[i]));
				prob = numeratorS / denom;
				rawProb.first.push_back(prob);

				numeratorD = (pclassDied[2] * sexDied[1] * diedApriori*ageLikelihood(ageMeanDie, ageVarDie, col4[i]));
				prob = numeratorD / denom;
				rawProb.second.push_back(prob);
			}

		}
	}

	return rawProb;
}
//funstion predicts wether or not a person died based on age, sex, and pclass.
vector<double> predictDied(pair<vector<double>, vector<double>> toPredict, vector<double> actual, int n = 0, int offset = 0)
{
	vector<double> predictions;
	int truePos = 0;
	int trueNeg = 0;
	int falsePos = 0;
	int falseNeg = 0;
	int size = n == 0 ? toPredict.second.size() : n;
	for (int i = 0; i < size; i++)
	{

		if (toPredict.second[i] >= .50)
		{


			//positive case means did not survive = 0
			if (0 == actual[i + offset])
			{
				truePos++;
			}
			else
				falsePos++;
		}
		//predict negative case survive = 1
		else
		{

			//negative case means survive = 1
			if (1 == actual[i + offset])
			{
				trueNeg++;
			}
			else
				falseNeg++;
		}

	}
	//[0] true pos
	predictions.push_back(truePos);
	//[1] false pos
	predictions.push_back(falsePos);
	//[2] true neg
	predictions.push_back(trueNeg);
	//[3] false neg
	predictions.push_back(falseNeg);

	return predictions;
}
//predicts wether a person survived based on age, sex, and pclass.
vector<double> predictSurvive(pair<vector<double>, vector<double>> toPredict, vector<double> actual, int n = 0, int offset = 0)
{
	vector<double> predictions;
	int countDied = 0;
	int countSurvived = 0;
	int truePos = 0;
	int trueNeg = 0;
	int falsePos = 0;
	int falseNeg = 0;
	int size = n == 0 ? toPredict.first.size() : n;
	for (int i = 0; i < size; i++)
	{

		if (toPredict.first[i] >= .5)
		{

			countDied++;
			//positive case means did not survive = 0
			if (0 == actual[i + offset])
			{
				truePos++;
			}
			else
				falsePos++;
		}
		//predict negative case survive = 1
		else
		{
			countSurvived++;
			//negative case means survive = 1
			if (1 == actual[i + offset])
			{
				trueNeg++;
			}
			else
				falseNeg++;
		}

	}
	//[0] true pos
	predictions.push_back(truePos);
	//[1] false pos
	predictions.push_back(falsePos);
	//[2] true neg
	predictions.push_back(trueNeg);
	//[3] false neg
	predictions.push_back(falseNeg);

	return predictions;
}
//computes accuracy. true pos + true neg / all outcomes
double accuracy(vector<double> confusionMatrix) {
	return (confusionMatrix[0] + confusionMatrix[2]) / (confusionMatrix[0] + confusionMatrix[1] + confusionMatrix[2] + confusionMatrix[3]);
}
//computes specificity. True negative/true negative + false positive
double specificity(vector<double> confusionMatrix) {
	return (confusionMatrix[0]) / (confusionMatrix[0] + confusionMatrix[1]);
}
//computes sensitivity. True pos/true pos + false neg
double sensitivity(vector<double> confusionMatrix) {
	return (confusionMatrix[0]) / (confusionMatrix[0] + confusionMatrix[3]);
}

int main() {
	//read in data set.
	vector<pair<string, vector<double>>> df = read_csv("titanic_project.csv");
	//instantiate vectors for different columns
	vector<double> pclassCol = df[0].second;
	vector<double> survivedCol = df[1].second;
	vector<double> sexCol = df[2].second;
	vector<double> ageCol = df[3].second;

	clock_t cstart;
	cstart = clock();
	//survival aproiori data and its complement
	double surviveAprioriProb = computeAprioriProb(df[1].first, df[1].second);
	double diedAprioriProb = 1 - surviveAprioriProb;

	//instantiate likelihood vectors for different columns
	vector<double> likelihood_pclass_survived;
	vector<double> likelihood_pclass_died;
	vector<double> likelihood_sex_survived;
	vector<double> likelihood_sex_died;
	likelihood_pclass_survived = computeDependencyMatrix(pclassCol, survivedCol, surviveAprioriProb, 1, 3, 1);
	likelihood_pclass_died = computeDependencyMatrix(pclassCol, survivedCol, diedAprioriProb, 1, 3, 0);
	likelihood_sex_survived = computeDependencyMatrix(sexCol, survivedCol, surviveAprioriProb, 0, 1, 1);
	likelihood_sex_died = computeDependencyMatrix(sexCol, survivedCol, diedAprioriProb, 0, 1, 0);

	//Various variables to be used later like likelihood
	vector<double> survived_byAge = computeDependent(ageCol, survivedCol, 1);
	vector<double> died_byAge = computeDependent(ageCol, survivedCol, 0);
	double ageMeanSurvived = mean(survived_byAge);
	double ageMeanDied = mean(died_byAge);
	double ageVarSurvived = variance(survived_byAge, ageMeanSurvived);
	double ageVarDied = variance(died_byAge, ageMeanDied);


	//run Naive Bayes on test data
	pair<vector<double>, vector<double>> testNB = calc_raw_prob(pclassCol, survivedCol, sexCol, ageCol,
		surviveAprioriProb, diedAprioriProb,
		likelihood_pclass_survived, likelihood_pclass_died,
		likelihood_sex_survived, likelihood_sex_died,
		ageMeanSurvived, ageMeanDied, ageVarDied, ageVarSurvived, 1046, 900);

	vector<double> confusionMatrix = predictDied(testNB, survivedCol, 0, 900);

	//stop time test
	clock_t cend = clock() - cstart;
	cout << "Execution time:" << (float)cend / CLOCKS_PER_SEC << "seconds" << endl;

	cout << "1: Probability Survived  2: Probability Perished\n";
	//print out first 5 test data elements
	for (int i = 0; i < 5; i++)
	{
		cout << i << ": Raw prob Test: " << "1:" << testNB.first[i] << " 2:" << testNB.second[i] << "\n";
	}



	cout << "True Pos " << confusionMatrix[0] << "\n";
	cout << "False Pos " << confusionMatrix[1] << "\n";
	cout << "True Neg " << confusionMatrix[2] << "\n";
	cout << "False Neg " << confusionMatrix[3] << "\n";

	cout << "Accuracy: " << accuracy(confusionMatrix)*100 << "\n";
	cout << "Specificity: " << specificity(confusionMatrix)*100 << "\n";
	cout << "Sensitivity: " << sensitivity(confusionMatrix)*100 << "\n";



	return 0;
}
