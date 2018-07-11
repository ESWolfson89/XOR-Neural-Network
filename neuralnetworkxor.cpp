// neural network XOR training simulation

// Eric Wolfson (sometime between 2014 and 2016)

#include <iostream>
#include <math.h>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <limits.h>

#define INP_COMBINATIONS 4
#define NUM_INP_NEURONS 3
#define MAX_ITERATIONS 100000

#define MIN_HIDDEN_NEURONS 3
#define MAX_HIDDEN_NEURONS 29

static const double euler_const = 2.71828182845904523536;
static const double learn_rate = 0.7;
static const double momentum = 0.5;
static const double end_error = 0.01;
static const double ideal_vector_XOR[4] = {0.0,1.0,1.0,0.0};

enum fn_type
{
    ACT_SIGMOID,
    ACT_TANH
};

double compute_activation(double, fn_type);
double compute_derivative(double, fn_type);

int string2Int(std::string);
bool isConvertibleToNumber(std::string);

class edge
{
public:
    edge();
    edge(double);
    double getWeight();
    double getWeightDelta();
    double getGradient();
    void incGradient(double);
    void zeroGradient();
    void setWeight(double);
    void setWeightDelta(double);
private:
    double weight;
    double weight_delta;
    double batch_gradient;
};

class neuron
{
  public:
    neuron();
    neuron(bool);
    void setSum(double);
    void computeOutput(fn_type);
    void setOutput(double);
    void addEdge(edge e);
    bool getBiasFlag();
    edge *getEdge(int);
    int getID();
    int getNumEdges();
    double getSum();
    double getOutput();
  private:
    bool is_bias;
    double sum;
    double output;
    std::vector<edge> edges_out;
};

class network
{
  public:
      network();
      void initNetwork();
      void initVectors();
      void initUserVariables();
      void initNodes();
      void initEdges();
      void initFixedBiases();
      void performTraining();
      void feedForward(double,double);
      void feedForwardGivenInput(int);
      void zeroAllGradients();
      void computeErrorDifferential(int);
      void computeDeltas();
      void incGradientsInBatch();
      void updateWeights();
      void calculateFinalAccuracy();
      void reportResults();
      void showInitialData();
      void cleanUp();
  private:
      double ideal_vector[INP_COMBINATIONS];
      double actual_vector[INP_COMBINATIONS];
      double hidden_delta[MAX_HIDDEN_NEURONS - 1];
      double error_diff;
      double output_delta;
      double new_weight_delta;
      int iterations;
      int num_tests_passed;
      int num_hidden_neurons;
      fn_type activation_function_used;
      std::vector<neuron> input_neurons;
      std::vector<neuron> hidden_neurons;
      std::vector<neuron> output_neurons;
};

int main()
{
    srand(time(NULL));

    network nn;

    std::cout << "<<<XOR neural network>>>\n";

    nn.initNetwork();

    nn.performTraining();

    nn.cleanUp();

    return 0;
}

// divide by 0 safe since it has no real solutions such that
// denominator equals 0
double compute_activation(double x, fn_type fn)
{
    switch(fn)
    {
	case(ACT_SIGMOID):
    	     return 1.0/(1.0 + pow(euler_const,-x));
	default:
	     break;
    }
    return (2.0/(1+pow(euler_const,-2.0*x))) - 1.0;
}

double compute_derivative(double x, fn_type fn)
{
    double exp_temp;
    switch(fn)
    {
	case(ACT_SIGMOID):
    	     exp_temp = pow(euler_const,x);
    	     return exp_temp/((1.0+exp_temp)*(1.0+exp_temp));
	default:
	     break;
    }
    exp_temp = pow(euler_const,2.0*x);
    return (4.0 * exp_temp) / ((exp_temp + 1) * (exp_temp + 1));
}

// network class implementation

network::network()
{
    num_hidden_neurons = 4;
    activation_function_used = ACT_SIGMOID;
}

void network::initNetwork()
{
    initUserVariables();
    initVectors();
    initNodes();
    initEdges();
    initFixedBiases();
    showInitialData();
}

void network::initUserVariables()
{
    std::string s;
    bool is_valid_number;
    int number = 4;
    // prompt for activation function
    std::cout << "Enter the activation function the neurons should use\n";
    std::cout << "(\"tanh\" for hyperbolic tangent; anything else is default sigmoid)\n";
    std::cout << "followed by the number of neurons in the hidden layer(3-29)\n";
    std::cout << "(for example: tanh 5): ";
    std::cin >> s;
    if (s == "tanh")
        activation_function_used = ACT_TANH;
    // prompt for number of hidden layer neurons
    std::cin >> s;
    is_valid_number = isConvertibleToNumber(s);
    if (is_valid_number)
    {
        number = string2Int(s);
        if (number >= MIN_HIDDEN_NEURONS && number <= MAX_HIDDEN_NEURONS)
        {
            num_hidden_neurons = number;
            return;
        }
    }

    std::cout << "Defaulting to 4...\n";
}

void network::showInitialData()
{
    std::cout << "Network has 3 input neurons, " << hidden_neurons.size() << " hidden neurons, and 1 output neuron:\n";

    for (int i = 0; i < (int)std::max(0,(int)hidden_neurons.size() - (int)input_neurons.size() - 1); ++i)
         std::cout << " ";
    for (int i = 0; i < input_neurons.size(); ++i)
         std::cout << " * ";
    std::cout << "\n";
    for (int i = 0; i < (int)std::max(0,(int)hidden_neurons.size() - (int)input_neurons.size() - 1); ++i)
         std::cout << " ";
    for (int i = 0; i < input_neurons.size(); ++i)
         std::cout << "/|\\";
    std::cout << "\n ";
    if (hidden_neurons.size() == MIN_HIDDEN_NEURONS)
        std::cout << " ";
    for (int i = 0; i < hidden_neurons.size(); ++i)
         std::cout << "* ";
    std::cout << "\n ";
    if (hidden_neurons.size() == MIN_HIDDEN_NEURONS)
        std::cout << " ";
    std::cout << "\\ ";
    for (int i = 0; i < hidden_neurons.size() - 2; ++i)
         std::cout << "| ";
    std::cout << "/";
    std::cout << "\n";
    if (hidden_neurons.size() == MIN_HIDDEN_NEURONS)
        std::cout << " ";
    for (int i = 0; i < hidden_neurons.size(); ++i)
         std::cout << " ";
    std::cout << "*\n";

    std::cout << "Max batch training iterations is " << MAX_ITERATIONS << ".\n";

    std::cout << "Initial weights are randomized from -1.0 to 1.0 inclusive in increments of 0.1\n";

    if (activation_function_used == ACT_SIGMOID)
        std::cout << "Activation function used: f(x) = 1/(1+e^-x)\n";
    else
        std::cout << "Activation function used: f(x) = tanh\n";

    std::cout << "Training network to compute XOR...\n";
}

void network::initVectors()
{
    for (int i = 0; i < INP_COMBINATIONS; ++i)
    {
      ideal_vector[i] = ideal_vector_XOR[i];
    }

    // hidden_delta will contain more array slots
    // than will be used unless num_hidden_neurons == MAX_HIDDEN_NEURONS
    for (int i = 0; i < MAX_HIDDEN_NEURONS - 1; ++i)
    {
      hidden_delta[i] = 0.0;
    }
}

// IN ORDER FOR THIS PROGRAM TO WORK:
// BIASES MUST TRAIL NON BIASES IN INPUT AND HIDDEN LAYERS
// FOR INSTANCE: I I I I I I B  IS A VALID INPUT LAYER
//               I I B I I I I  IS *NOT* A VALID INPUT LAYER
// where I = non input bias and B = input bias

void network::initNodes()
{
    for (int i = 0; i < NUM_INP_NEURONS; ++i)
    {
      input_neurons.push_back(neuron(i == NUM_INP_NEURONS - 1 ? true : false));
    }

    for (int i = 0; i < num_hidden_neurons; ++i)
    {
      hidden_neurons.push_back(neuron(i == num_hidden_neurons - 1 ? true : false));
    }

    output_neurons.push_back(neuron(false));
}

void network::initEdges()
{
    for (int in = 0; in < input_neurons.size(); ++in)
    {
        for (int ie = 0; ie < hidden_neurons.size() - 1; ++ie)
        {
            input_neurons[in].addEdge(edge((double)((rand() % 21) - 10)/10.0));
        }
    }

    for (int h = 0; h < hidden_neurons.size(); ++h)
    {
        hidden_neurons[h].addEdge(edge((double)((rand() % 21) - 10)/10.0));
    }
}

void network::initFixedBiases()
{
    input_neurons[input_neurons.size() - 1].setOutput(1.0);
    hidden_neurons[hidden_neurons.size() - 1].setOutput(1.0);
}

void network::feedForward(double inp1, double inp2)
{
    input_neurons[0].setOutput(inp1);
    input_neurons[1].setOutput(inp2);

    double sum;

    for (int n = 0; n < hidden_neurons.size(); ++n)
    {
        if (!hidden_neurons[n].getBiasFlag())
        {
            sum = 0.0;

            for (int i = 0; i < input_neurons.size(); ++i)
            {
                sum += (input_neurons[i].getOutput() * input_neurons[i].getEdge(n)->getWeight());
            }

            hidden_neurons[n].setSum(sum);
            hidden_neurons[n].computeOutput(activation_function_used);
        }
    }

    sum = 0.0;

    for (int i = 0; i < hidden_neurons.size(); ++i)
    {
        sum += (hidden_neurons[i].getOutput() * hidden_neurons[i].getEdge(0)->getWeight());
    }

    output_neurons[0].setSum(sum);
    output_neurons[0].computeOutput(activation_function_used);
}

void network::zeroAllGradients()
{
    for (int in = 0; in < input_neurons.size(); ++in)
    for (int ie = 0; ie < input_neurons[in].getNumEdges(); ++ie)
         input_neurons[in].getEdge(ie)->zeroGradient();

    for (int on = 0; on < hidden_neurons.size(); ++on)
         hidden_neurons[on].getEdge(0)->zeroGradient();
}

void network::feedForwardGivenInput(int i)
{
    switch(i)
    {
        case(0):
            feedForward(0,0);
            break;
        case(1):
            feedForward(0,1);
            break;
        case(2):
            feedForward(1,0);
            break;
        case(3):
            feedForward(1,1);
            break;
        default:
            std::cerr << "\nBAD INPUT VECTOR INDEX!\n";
    }
}

void network::computeErrorDifferential(int i)
{
    actual_vector[i] = output_neurons[0].getOutput();
    error_diff = actual_vector[i] - ideal_vector[i];
}

void network::computeDeltas()
{
    output_delta = -error_diff * compute_derivative(output_neurons[0].getSum(),activation_function_used);

    for (int h = 0; h < hidden_neurons.size() - 1; ++h)
    {
        hidden_delta[h] = compute_derivative(hidden_neurons[h].getSum(),activation_function_used) *
                          hidden_neurons[h].getEdge(0)->getWeight() *
                          output_delta;
    }
}

void network::incGradientsInBatch()
{
    for (int in = 0; in < input_neurons.size(); ++in)
    {
        for (int ie = 0; ie < input_neurons[in].getNumEdges(); ++ie)
        {
            input_neurons[in].getEdge(ie)->incGradient(hidden_delta[ie] * input_neurons[in].getOutput());
        }
    }

    for (int h = 0; h < hidden_neurons.size(); ++h)
    {
        hidden_neurons[h].getEdge(0)->incGradient(output_delta * hidden_neurons[h].getOutput());
    }
}

void network::updateWeights()
{
    new_weight_delta = 0.0;

    for (int in = 0; in < input_neurons.size(); ++in)
    for (int ie = 0; ie < input_neurons[in].getNumEdges(); ++ie)
    {
        new_weight_delta = (learn_rate * input_neurons[in].getEdge(ie)->getGradient()) +
                           (momentum * input_neurons[in].getEdge(ie)->getWeightDelta());
        input_neurons[in].getEdge(ie)->setWeight(input_neurons[in].getEdge(ie)->getWeight() + new_weight_delta);
        input_neurons[in].getEdge(ie)->setWeightDelta(new_weight_delta);
    }

    for (int h = 0; h < hidden_neurons.size(); ++h)
    {
        new_weight_delta = (learn_rate * hidden_neurons[h].getEdge(0)->getGradient()) +
                           (momentum * hidden_neurons[h].getEdge(0)->getWeightDelta());
        hidden_neurons[h].getEdge(0)->setWeight(hidden_neurons[h].getEdge(0)->getWeight() + new_weight_delta);
        hidden_neurons[h].getEdge(0)->setWeightDelta(new_weight_delta);
    }
}

void network::calculateFinalAccuracy()
{
    num_tests_passed = 0;

    for (int r = 0; r < INP_COMBINATIONS; ++r)
    {
        if (fabs(ideal_vector[r] - actual_vector[r]) <= end_error)
        {
            num_tests_passed++;
        }
    }
}

void network::performTraining()
{
    num_tests_passed = 0;
    iterations = 0;

    do
    {
        iterations++;
        zeroAllGradients();

        for (int i = 0; i < INP_COMBINATIONS; ++i)
        {
            feedForwardGivenInput(i);
            computeErrorDifferential(i);
            computeDeltas();
            incGradientsInBatch();
        }

        updateWeights();
        calculateFinalAccuracy();

    } while(num_tests_passed < INP_COMBINATIONS && iterations < MAX_ITERATIONS);

    reportResults();
}

void network::reportResults()
{
    if (iterations == MAX_ITERATIONS)
        std::cout << "Training failed!\n";
    else
        std::cout << "Training succeeded!\n";

    std::cout << "Number of iterations: " << iterations << "\n";
    std::cout << "Final results (should converge to ideal outputs on an XOR truth table):\n";
    for (int i = 0; i < INP_COMBINATIONS; ++i)
    {
        feedForwardGivenInput(i);
        std::cout << "Input: " << i << " combination -> " << output_neurons[0].getOutput() << "\n";
    }

    std::cout << "-------final weights-------\n";

    for (int in = 0; in < input_neurons.size(); ++in)
    for (int ie = 0; ie < input_neurons[in].getNumEdges(); ++ie)
    {
        std::cout << "edge weight: input node " << in << " to hidden node " << ie << " -> " << input_neurons[in].getEdge(ie)->getWeight() << "\n";
    }

    for (int h = 0; h < hidden_neurons.size(); ++h)
    {
        std::cout << "edge weight: hidden node " << h << " to output node -> " << hidden_neurons[h].getEdge(0)->getWeight() << "\n";
    }

    std::cin.get();
    std::cin.get();

    std::cout << "Termination successful.\n";
}

void network::cleanUp()
{
    std::vector<neuron>().swap(input_neurons);
    std::vector<neuron>().swap(hidden_neurons);
    std::vector<neuron>().swap(output_neurons);
}

// edge class implementation

edge::edge()
{

}

edge::edge(double w)
{
    weight = w;
    weight_delta = 0.0;
    batch_gradient = 0.0;
}

double edge::getWeight()
{
    return weight;
}

double edge::getWeightDelta()
{
    return weight_delta;
}

double edge::getGradient()
{
    return batch_gradient;
}

void edge::setWeight(double d)
{
    weight = d;
}

void edge::setWeightDelta(double wd)
{
    weight_delta = wd;
}

void edge::incGradient(double g)
{
    batch_gradient += g;
}

void edge::zeroGradient()
{
    batch_gradient = 0.0;
}

// neuron class implementation

neuron::neuron()
{

}

neuron::neuron(bool b)
{
    is_bias = b;
    sum = -1.0;
    output = -1.0;
}

void neuron::setSum(double s)
{
    sum = s;
}

void neuron::computeOutput(fn_type fn_used)
{
    output = compute_activation(sum,fn_used);
}

double neuron::getSum()
{
    return sum;
}

double neuron::getOutput()
{
    return output;
}

void neuron::addEdge(edge e)
{
    edges_out.push_back(e);
}

edge *neuron::getEdge(int i)
{
    return &edges_out[i];
}

int neuron::getNumEdges()
{
    return edges_out.size();
}

void neuron::setOutput(double d)
{
    output = d;
}

bool neuron::getBiasFlag()
{
    return is_bias;
}

// utility functions
int string2Int(std::string s)
{
    int result = 0;
    int digit = 0;
    int factor = 1;
    int ssize = s.size();
    for (int i = ssize-1; i >= 0; --i)
    {
        digit = (int)s[ssize-i-1] - 48;
        factor = 1;
        for (int j = 0; j < i; ++j)
        {
            factor *= 10;
        }
        result += (digit * factor);
    }
    return result;
}

bool isConvertibleToNumber(std::string s)
{
    for (int i = 0; i < s.size(); ++i)
    {
        if (s[i] < '0' || s[i] > '9')
            return false;
    }
    return true;
}
